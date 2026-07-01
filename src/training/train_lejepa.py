from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from src.data.dataloaders import build_loaders
from src.networks.factory import build_lejepa_model, build_linear_probe_model
from src.networks.resnet import LeJEPAResNet
from src.training.checkpointing import load_checkpoint, save_checkpoint
from src.training.losses import MultiViewLeJEPASIGRegLoss, build_sigreg_from_config
from src.training.train_supervised import evaluate
from src.utils import ensure_dir, get_device, load_yaml, set_seed


def _as_view_list(views) -> list[torch.Tensor]:
    if isinstance(views, torch.Tensor):
        raise ValueError("Self-supervised loader returned a single tensor. Expected multi-crop views.")
    if isinstance(views, (list, tuple)):
        return list(views)
    raise TypeError(f"Unsupported views type: {type(views)}")


def _build_lejepa_loss(cfg: dict[str, Any]) -> MultiViewLeJEPASIGRegLoss:
    lejepa_cfg = cfg.get("lejepa", {})
    sigreg = build_sigreg_from_config(cfg)
    return MultiViewLeJEPASIGRegLoss(
        sigreg=sigreg,
        prediction_weight=float(lejepa_cfg.get("prediction_weight", 1.0)),
        sigreg_weight=float(lejepa_cfg.get("sigreg_weight", cfg.get("training", {}).get("sigreg_weight", 0.05))),
        normalize_prediction=bool(lejepa_cfg.get("normalize_prediction", False)),
    )


def build_warmup_cosine_scheduler(
    optimizer: Optimizer,
    warmup_epochs: int,
    total_epochs: int,
) -> LambdaLR:
    """Linear warmup followed by cosine decay."""
    warmup_epochs = int(max(0, warmup_epochs))
    total_epochs = int(max(1, total_epochs))

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)

        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def _as_list_of_ints(value: Any, default: list[int]) -> list[int]:
    if value is None:
        return default
    if isinstance(value, list):
        return [int(v) for v in value]
    if isinstance(value, str):
        return [int(v.strip()) for v in value.split(",") if v.strip()]
    return [int(value)]


def _unique_existing(paths: list[str | Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()

    for path in paths:
        p = Path(path)
        key = str(p)
        if key in seen or not p.exists():
            continue
        seen.add(key)
        out.append(p)

    return out


@torch.no_grad()
def _extract_backbone_features(
    backbone: torch.nn.Module,
    loader,
    device: torch.device,
    max_samples: int | None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]:
    backbone.eval()

    features: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    std_means: list[torch.Tensor] = []
    std_mins: list[torch.Tensor] = []
    norm_means: list[torch.Tensor] = []
    total = 0

    for images, labels in tqdm(loader, desc="selection extract", leave=False):
        if max_samples is not None and total >= max_samples:
            break

        if max_samples is not None and total + images.shape[0] > max_samples:
            remaining = max_samples - total
            images = images[:remaining]
            labels = labels[:remaining]

        images = images.to(device, non_blocking=True)
        raw = backbone(images).float()

        batch_std = raw.std(dim=0, unbiased=False)
        std_means.append(batch_std.mean().detach().cpu())
        std_mins.append(batch_std.min().detach().cpu())
        norm_means.append(raw.norm(dim=1).mean().detach().cpu())

        feats = F.normalize(raw, dim=1)
        features.append(feats.detach().cpu())
        labels_all.append(labels.detach().cpu())
        total += int(labels.numel())

    features_t = torch.cat(features, dim=0)
    labels_t = torch.cat(labels_all, dim=0).long()

    diagnostics = {
        "num_samples": int(features_t.shape[0]),
        "feature_dim": int(features_t.shape[1]),
        "std_mean": float(torch.stack(std_means).mean().item()),
        "std_min": float(torch.stack(std_mins).mean().item()),
        "norm_mean": float(torch.stack(norm_means).mean().item()),
    }

    return features_t, labels_t, diagnostics


@torch.no_grad()
def _knn_accuracy(
    bank_features: torch.Tensor,
    bank_labels: torch.Tensor,
    query_features: torch.Tensor,
    query_labels: torch.Tensor,
    num_classes: int,
    k: int,
    temperature: float,
    query_chunk_size: int,
    device: torch.device,
) -> dict[str, float | int]:
    bank_features = bank_features.to(device=device, dtype=torch.float32)
    bank_labels = bank_labels.to(device=device, dtype=torch.long)
    query_features = query_features.to(device=device, dtype=torch.float32)
    query_labels = query_labels.to(device=device, dtype=torch.long)

    k = min(int(k), int(bank_features.shape[0]))
    correct = 0
    total = 0

    for start in tqdm(range(0, query_features.shape[0], query_chunk_size), desc=f"selection kNN k={k}", leave=False):
        end = min(start + query_chunk_size, query_features.shape[0])
        q = query_features[start:end]
        labels = query_labels[start:end]

        sim = q @ bank_features.T
        values, indices = sim.topk(k=k, dim=1)
        topk_labels = bank_labels[indices]
        weights = torch.exp(values / temperature)

        votes = torch.zeros((q.shape[0], num_classes), dtype=torch.float32, device=device)
        for i in range(q.shape[0]):
            votes[i].index_add_(0, topk_labels[i], weights[i])

        preds = votes.argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total += int(labels.numel())

    return {
        "k": k,
        "accuracy": correct / max(total, 1),
        "correct": correct,
        "total": total,
    }


def maybe_select_checkpoint_by_knn(
    cfg: dict[str, Any],
    model: torch.nn.Module,
    candidate_paths: list[str | Path],
    best_path: str | Path,
    device: torch.device,
) -> None:
    """
    Optionally replace backbone_best_path with the candidate checkpoint that has
    the best frozen-backbone kNN accuracy.

    This is controlled by:
      checkpoint_selection:
        enabled: true
        select_backbone_by: knn
    """
    selection_cfg = cfg.get("checkpoint_selection", {}) or {}
    if not bool(selection_cfg.get("enabled", False)):
        return

    metric = str(selection_cfg.get("select_backbone_by", "pretrain_loss")).lower()
    if metric not in {"knn", "pretrain_loss", "loss"}:
        raise ValueError(f"Unsupported checkpoint_selection.select_backbone_by={metric!r}")

    if metric in {"pretrain_loss", "loss"}:
        print("checkpoint_selection enabled, but select_backbone_by=pretrain_loss; keeping loss-best backbone.")
        return

    candidates = _unique_existing(candidate_paths)
    if not candidates:
        raise FileNotFoundError("No existing candidate checkpoints found for kNN selection.")

    old_batch_size = cfg.get("data", {}).get("batch_size")
    cfg.setdefault("data", {})
    cfg["data"]["batch_size"] = int(
        selection_cfg.get(
            "batch_size",
            cfg.get("evaluation", {}).get("representation_batch_size", 256),
        )
    )

    loaders = build_loaders(cfg, self_supervised=False)

    if old_batch_size is not None:
        cfg["data"]["batch_size"] = old_batch_size

    max_bank_samples = selection_cfg.get("max_bank_samples", 10000)
    max_test_samples = selection_cfg.get("max_test_samples", 2000)
    max_bank_samples = None if max_bank_samples in {None, "none", "None", "null"} else int(max_bank_samples)
    max_test_samples = None if max_test_samples in {None, "none", "None", "null"} else int(max_test_samples)

    knn_ks = _as_list_of_ints(selection_cfg.get("knn_ks", selection_cfg.get("knn_k", 20)), [20])
    temperature = float(selection_cfg.get("temperature", cfg.get("evaluation", {}).get("knn_temperature", 0.07)))
    query_chunk_size = int(selection_cfg.get("query_chunk_size", cfg.get("evaluation", {}).get("knn_query_chunk_size", 256)))
    num_classes = int(cfg["model"]["num_classes"])

    rows: list[dict[str, Any]] = []

    print("Backbone checkpoint selection by frozen-backbone kNN:")
    print(f"  candidates: {len(candidates)}")
    print(f"  k values: {knn_ks}")
    print(f"  max_bank_samples: {max_bank_samples}")
    print(f"  max_test_samples: {max_test_samples}")

    for checkpoint_path in candidates:
        print(f"\nEvaluating candidate: {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()

        bank_features, bank_labels, bank_diag = _extract_backbone_features(
            backbone=model.backbone,
            loader=loaders.train_eval,
            device=device,
            max_samples=max_bank_samples,
        )
        query_features, query_labels, query_diag = _extract_backbone_features(
            backbone=model.backbone,
            loader=loaders.test,
            device=device,
            max_samples=max_test_samples,
        )

        for k in knn_ks:
            knn = _knn_accuracy(
                bank_features=bank_features,
                bank_labels=bank_labels,
                query_features=query_features,
                query_labels=query_labels,
                num_classes=num_classes,
                k=k,
                temperature=temperature,
                query_chunk_size=query_chunk_size,
                device=device,
            )
            rows.append(
                {
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
                    "pretrain_loss": float(checkpoint.get("pretrain_loss", float("nan"))),
                    "prediction_loss": float(checkpoint.get("prediction_loss", float("nan"))),
                    "sigreg_loss": float(checkpoint.get("sigreg_loss", float("nan"))),
                    "knn_k": int(knn["k"]),
                    "knn_accuracy": float(knn["accuracy"]),
                    "knn_correct": int(knn["correct"]),
                    "knn_total": int(knn["total"]),
                    "bank_samples": int(bank_features.shape[0]),
                    "query_samples": int(query_features.shape[0]),
                    "feature_dim": int(bank_features.shape[1]),
                    "bank_std_mean": float(bank_diag["std_mean"]),
                    "bank_std_min": float(bank_diag["std_min"]),
                    "bank_norm_mean": float(bank_diag["norm_mean"]),
                    "query_std_mean": float(query_diag["std_mean"]),
                    "query_std_min": float(query_diag["std_min"]),
                    "query_norm_mean": float(query_diag["norm_mean"]),
                }
            )

    if not rows:
        raise RuntimeError("Checkpoint selection produced no kNN rows.")

    best_row = max(rows, key=lambda r: float(r["knn_accuracy"]))
    metrics_dir = Path(cfg.get("_experiment", {}).get("metrics_dir", "outputs/metrics"))
    output_dir = ensure_dir(metrics_dir / "checkpoint_selection")
    csv_path = output_dir / "backbone_checkpoint_selection_knn.csv"
    yaml_path = output_dir / "backbone_checkpoint_selection_knn.yaml"

    pd.DataFrame(rows).to_csv(csv_path, index=False)

    result = {
        "enabled": True,
        "select_backbone_by": "knn",
        "selected_checkpoint_path": str(best_row["checkpoint_path"]),
        "selected_epoch": int(best_row["checkpoint_epoch"]),
        "best_knn_accuracy": float(best_row["knn_accuracy"]),
        "knn_k": int(best_row["knn_k"]),
        "num_candidates": len(candidates),
        "candidate_paths": [str(p) for p in candidates],
        "selection_csv": str(csv_path),
        "selection_yaml": str(yaml_path),
        "settings": {
            "max_bank_samples": max_bank_samples,
            "max_test_samples": max_test_samples,
            "knn_ks": knn_ks,
            "temperature": temperature,
            "query_chunk_size": query_chunk_size,
        },
        "best_row": best_row,
    }

    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(result, f, sort_keys=False)

    selected_checkpoint = load_checkpoint(result["selected_checkpoint_path"], map_location=device)
    selected_checkpoint["checkpoint_selection"] = result
    save_checkpoint(best_path, selected_checkpoint)

    print("\nBackbone checkpoint selection completed.")
    print(f"  selected: {result['selected_checkpoint_path']}")
    print(f"  best kNN: {result['best_knn_accuracy']:.4f} at k={result['knn_k']}")
    print(f"  saved CSV: {csv_path}")
    print(f"  overwritten backbone_best_path: {best_path}")


def _build_probe_optimizer(cfg: dict[str, Any], parameters) -> Optimizer:
    probe_optimizer = str(cfg["training"].get("probe_optimizer", "adamw")).lower()
    probe_lr = float(cfg["training"].get("probe_learning_rate", 0.001))
    probe_weight_decay = float(cfg["training"].get("probe_weight_decay", cfg["training"].get("weight_decay", 0.0)))

    if probe_optimizer == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=probe_lr,
            momentum=float(cfg["training"].get("probe_momentum", 0.9)),
            weight_decay=probe_weight_decay,
            nesterov=bool(cfg["training"].get("probe_nesterov", True)),
        )

    if probe_optimizer == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=probe_lr,
            weight_decay=probe_weight_decay,
        )

    raise ValueError(f"Unsupported probe_optimizer={probe_optimizer!r}")


def pretrain_lejepa(cfg: dict[str, Any], device: torch.device) -> nn.Module:
    loaders = build_loaders(cfg, self_supervised=True)
    model = build_lejepa_model(cfg).to(device)
    if not hasattr(model, "forward_views"):
        raise TypeError(f"LeJEPA model must implement forward_views(); got {type(model)}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        warmup_epochs=int(cfg["training"].get("warmup_epochs", 0)),
        total_epochs=int(cfg["training"]["pretrain_epochs"]),
    )
    criterion = _build_lejepa_loss(cfg)
    use_amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    best_loss = float("inf")
    best_path = cfg["checkpoints"]["backbone_best_path"]

    selection_cfg = cfg.get("checkpoint_selection", {}) or {}
    selection_enabled = bool(selection_cfg.get("enabled", False))
    selection_metric = str(selection_cfg.get("select_backbone_by", "pretrain_loss")).lower()
    loss_best_path = cfg["checkpoints"].get("backbone_loss_best_path", best_path)
    loss_best_target = loss_best_path if selection_enabled and selection_metric == "knn" else best_path
    save_candidate_every = int(selection_cfg.get("save_candidate_every_epochs", 0) or 0)
    candidate_paths: list[str] = []

    print("LeJEPA pretraining method:")
    print(f"  architecture: {cfg['model'].get('architecture')}")
    print(f"  sigreg_implementation: {cfg.get('lejepa', {}).get('sigreg_implementation', 'official')}")
    print(f"  num_global_views: {cfg.get('lejepa_views', {}).get('num_global_views', 'config/default')}")
    print(f"  num_local_views: {cfg.get('lejepa_views', {}).get('num_local_views', 'config/default')}")
    print(f"  scheduler: warmup_cosine warmup_epochs={int(cfg['training'].get('warmup_epochs', 0))}")
    print(f"  checkpoint_selection: enabled={selection_enabled} metric={selection_metric}")
    print("  stop_gradient: false")
    print("  teacher_student_or_ema: false")

    for epoch in range(1, int(cfg["training"]["pretrain_epochs"]) + 1):
        model.train()
        total_loss = 0.0
        total_prediction = 0.0
        total_sigreg = 0.0
        seen = 0

        progress = tqdm(loaders.train, desc=f"lejepa pretrain epoch {epoch}", leave=False)
        for views, _ in progress:
            view_list = [v.to(device, non_blocking=True) for v in _as_view_list(views)]
            batch_size = int(view_list[0].shape[0])

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
        lr = float(optimizer.param_groups[0]["lr"])
        print(f"epoch={epoch:03d} lejepa_loss={avg_loss:.4f} pred={avg_pred:.4f} sigreg={avg_sig:.4f} lr={lr:.6g}")

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

        last_path = cfg["checkpoints"]["backbone_last_path"]
        save_checkpoint(last_path, payload)
        if str(last_path) not in candidate_paths:
            candidate_paths.append(str(last_path))

        if save_candidate_every > 0 and epoch % save_candidate_every == 0:
            last_path_obj = Path(last_path)
            candidate_path = last_path_obj.with_name(
                last_path_obj.stem.replace("_last", "") + f"_epoch_{epoch:03d}" + last_path_obj.suffix
            )
            save_checkpoint(candidate_path, payload)
            if str(candidate_path) not in candidate_paths:
                candidate_paths.append(str(candidate_path))

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(loss_best_target, payload)
            if str(loss_best_target) not in candidate_paths:
                candidate_paths.append(str(loss_best_target))

    maybe_select_checkpoint_by_knn(
        cfg=cfg,
        model=model,
        candidate_paths=candidate_paths,
        best_path=best_path,
        device=device,
    )

    best_ckpt = load_checkpoint(best_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    return model


def train_linear_probe(cfg: dict[str, Any], pretrained: nn.Module, device: torch.device) -> nn.Module:
    loaders = build_loaders(cfg, self_supervised=False)
    model = build_linear_probe_model(cfg, pretrained.backbone).to(device)

    for p in model.backbone.parameters():
        p.requires_grad_(False)

    optimizer = _build_probe_optimizer(cfg, model.classifier.parameters())
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        warmup_epochs=int(cfg["training"].get("probe_warmup_epochs", 0)),
        total_epochs=int(cfg["training"].get("probe_epochs", 50)),
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
        progress = tqdm(loaders.train, desc=f"linear probe epoch {epoch}", leave=False)

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
        lr = float(optimizer.param_groups[0]["lr"])
        print(
            f"probe_epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} val_loss={val.loss:.4f} val_acc={val.accuracy:.4f} lr={lr:.6g}"
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

    best_ckpt = load_checkpoint(cfg["checkpoints"]["probe_best_path"], map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    test = evaluate(model, loaders.test, device)
    print(
        f"linear_probe_test_loss={test.loss:.4f} linear_probe_test_acc={test.accuracy:.4f} "
        f"best_val_acc={best_val_acc:.4f} best_epoch={best_epoch}"
    )
    return model


def train_lejepa(config_path: str) -> dict[str, float | str]:
    cfg = load_yaml(config_path)
    set_seed(int(cfg["project"]["seed"]))
    device = get_device()
    pretrained = pretrain_lejepa(cfg, device)
    train_linear_probe(cfg, pretrained, device)
    return {"status": "completed", "architecture": str(cfg["model"].get("architecture"))}
