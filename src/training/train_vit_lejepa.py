from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.data.dataloaders import build_loaders
from src.networks.factory import build_linear_probe_model, build_lejepa_model
from src.networks.vit import LeJEPAViTCifar
from src.training.checkpointing import load_checkpoint, save_checkpoint
from src.training.losses import MultiViewLeJEPASIGRegLoss, build_sigreg_from_config
from src.training.train_vit_supervised import evaluate
from src.utils import get_device, load_yaml, set_seed


def _as_view_list(views) -> list[torch.Tensor]:
    if isinstance(views, torch.Tensor):
        raise ValueError("Self-supervised loader returned a single tensor. Expected multiple views.")
    if isinstance(views, (list, tuple)):
        return list(views)
    raise TypeError(f"Unsupported views type: {type(views)}")


def _build_vit_lejepa_loss(cfg: dict[str, Any]) -> MultiViewLeJEPASIGRegLoss:
    lejepa_cfg = cfg.get("lejepa", {})
    sigreg = build_sigreg_from_config(cfg)
    return MultiViewLeJEPASIGRegLoss(
        sigreg=sigreg,
        prediction_weight=float(lejepa_cfg.get("prediction_weight", 1.0)),
        sigreg_weight=float(lejepa_cfg.get("sigreg_weight", cfg.get("training", {}).get("sigreg_weight", 0.05))),
        normalize_prediction=bool(lejepa_cfg.get("normalize_prediction", False)),
    )


def pretrain_vit_lejepa(cfg: dict[str, Any], device: torch.device) -> LeJEPAViTCifar:
    loaders = build_loaders(cfg, self_supervised=True)
    model = build_lejepa_model(cfg).to(device)
    if not isinstance(model, LeJEPAViTCifar):
        raise TypeError(f"Expected LeJEPAViTCifar, got {type(model)}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg["training"]["pretrain_epochs"]),
    )
    criterion = _build_vit_lejepa_loss(cfg)
    use_amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    best_loss = float("inf")
    best_path = cfg["checkpoints"]["backbone_best_path"]

    print("LeJEPA ViT pretraining method:")
    print(f"  architecture: {cfg['model'].get('architecture')}")
    print(f"  sigreg_implementation: {cfg.get('lejepa', {}).get('sigreg_implementation', 'official')}")
    print(f"  num_global_views: {cfg.get('lejepa_views', {}).get('num_global_views', 'config/default')}")
    print(f"  num_local_views: {cfg.get('lejepa_views', {}).get('num_local_views', 'config/default')}")
    print("  stop_gradient: false")
    print("  teacher_student_or_ema: false")

    for epoch in range(1, int(cfg["training"]["pretrain_epochs"]) + 1):
        model.train()
        total_loss = 0.0
        total_prediction = 0.0
        total_sigreg = 0.0
        seen = 0

        progress = tqdm(loaders.train, desc=f"vit lejepa pretrain epoch {epoch}", leave=False)
        for views, _ in progress:
            view_list = [v.to(device, non_blocking=True) for v in _as_view_list(views)]
            batch_size = view_list[0].shape[0]

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                outputs = model.forward_views(view_list)
                losses = criterion(
                    projections=outputs["z"],
                    predictions=outputs["p"],
                    embeddings_for_sigreg=outputs["embeddings_for_sigreg"],
                )
                loss = losses["loss"]

            scaler.scale(loss).backward()
            if "grad_clip_norm" in cfg["training"]:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg["training"]["grad_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item()) * batch_size
            total_prediction += float(losses["prediction_loss"].item()) * batch_size
            total_sigreg += float(losses["sigreg_loss"].item()) * batch_size
            seen += batch_size
            progress.set_postfix(
                loss=total_loss / max(seen, 1),
                pred=total_prediction / max(seen, 1),
                sig=total_sigreg / max(seen, 1),
            )

        scheduler.step()
        avg_loss = total_loss / max(seen, 1)
        avg_pred = total_prediction / max(seen, 1)
        avg_sig = total_sigreg / max(seen, 1)
        print(f"epoch={epoch:03d} vit_lejepa_loss={avg_loss:.4f} pred={avg_pred:.4f} sigreg={avg_sig:.4f}")

        payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "backbone_state": model.backbone.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "pretrain_loss": avg_loss,
            "prediction_loss": avg_pred,
            "sigreg_loss": avg_sig,
            "method": {
                "name": "LeJEPA faithful-method reproduction",
                "official_sigreg": str(cfg.get("lejepa", {}).get("sigreg_implementation", "official")).lower() in {"official", "lejepa"},
                "stop_gradient": False,
                "teacher_student_or_ema": False,
                "multi_crop_views": cfg.get("lejepa_views", {}),
                "deviations": cfg.get("method", {}).get("deviations", []),
            },
            "config": cfg,
        }
        save_checkpoint(cfg["checkpoints"]["backbone_last_path"], payload)
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(best_path, payload)

    # Load best pretraining checkpoint before probe training.
    best_ckpt = load_checkpoint(best_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    return model


def train_vit_linear_probe(cfg: dict[str, Any], pretrained: LeJEPAViTCifar, device: torch.device) -> nn.Module:
    loaders = build_loaders(cfg, self_supervised=False)
    model = build_linear_probe_model(cfg, pretrained.backbone).to(device)

    for p in model.backbone.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=float(cfg["training"].get("probe_learning_rate", 0.001)),
        weight_decay=float(cfg["training"].get("probe_weight_decay", cfg["training"].get("weight_decay", 0.0))),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg["training"].get("probe_epochs", 50)),
    )
    loss_fn = nn.CrossEntropyLoss()
    use_amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, int(cfg["training"].get("probe_epochs", 50)) + 1):
        model.train()
        model.backbone.eval()
        total_loss = 0.0
        correct = 0
        seen = 0
        progress = tqdm(loaders.train, desc=f"vit linear probe epoch {epoch}", leave=False)

        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                logits = model(images)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item()) * images.size(0)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            seen += images.size(0)
            progress.set_postfix(loss=total_loss / max(seen, 1), acc=correct / max(seen, 1))

        scheduler.step()
        val = evaluate(model, loaders.val, device)
        train_loss = total_loss / max(seen, 1)
        train_acc = correct / max(seen, 1)
        print(
            f"probe_epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} val_loss={val.loss:.4f} val_acc={val.accuracy:.4f}"
        )

        payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "backbone_state": model.backbone.state_dict(),
            "classifier_state": model.classifier.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_accuracy": val.accuracy,
            "config": cfg,
        }
        save_checkpoint(cfg["checkpoints"]["probe_last_path"], payload)
        if val.accuracy > best_val_acc:
            best_val_acc = val.accuracy
            best_epoch = epoch
            save_checkpoint(cfg["checkpoints"]["probe_best_path"], payload)

    # Evaluate best probe.
    best_ckpt = load_checkpoint(cfg["checkpoints"]["probe_best_path"], map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    test = evaluate(model, loaders.test, device)
    print(
        f"vit_linear_probe_test_loss={test.loss:.4f} "
        f"vit_linear_probe_test_acc={test.accuracy:.4f} "
        f"best_val_acc={best_val_acc:.4f} best_epoch={best_epoch}"
    )
    return model


def train_vit_lejepa(config_path: str) -> dict[str, float]:
    cfg = load_yaml(config_path)
    set_seed(int(cfg["project"]["seed"]))
    device = get_device()
    pretrained = pretrain_vit_lejepa(cfg, device)
    probe = train_vit_linear_probe(cfg, pretrained, device)
    return {"status": "completed", "architecture": str(cfg["model"].get("architecture"))}
