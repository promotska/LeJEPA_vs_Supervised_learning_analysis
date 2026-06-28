from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.data.dataloaders import build_loaders
from src.networks.vit import SupervisedViTCifar
from src.training.checkpointing import save_checkpoint
from src.utils import get_device, load_yaml, set_seed


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float


def build_vit_from_config(cfg: dict[str, Any]) -> SupervisedViTCifar:
    model_cfg = cfg["model"]

    return SupervisedViTCifar(
        image_size=int(model_cfg.get("image_size", 32)),
        patch_size=int(model_cfg.get("patch_size", 4)),
        num_classes=int(model_cfg["num_classes"]),
        embed_dim=int(model_cfg.get("embed_dim", 192)),
        depth=int(model_cfg.get("depth", 6)),
        num_heads=int(model_cfg.get("num_heads", 3)),
        mlp_ratio=float(model_cfg.get("mlp_ratio", 4.0)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        attn_dropout=float(model_cfg.get("attn_dropout", 0.1)),
    )


def evaluate(model: nn.Module, loader, device: torch.device) -> EpochMetrics:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = loss_fn(logits, labels)

            total_loss += loss.item() * images.size(0)
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_count += images.size(0)

    return EpochMetrics(
        loss=total_loss / max(total_count, 1),
        accuracy=total_correct / max(total_count, 1),
    )


def train_vit_supervised(config_path: str) -> dict[str, float]:
    cfg = load_yaml(config_path)
    set_seed(int(cfg["project"]["seed"]))
    device = get_device()

    loaders = build_loaders(cfg, self_supervised=False)

    model = build_vit_from_config(cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg["training"]["epochs"]),
    )

    loss_fn = nn.CrossEntropyLoss()
    use_amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        model.train()

        running_loss = 0.0
        correct = 0
        seen = 0

        progress = tqdm(
            loaders.train,
            desc=f"vit supervised epoch {epoch}",
            leave=False,
        )

        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                logits = model(images)
                loss = loss_fn(logits, labels)

            scaler.scale(loss).backward()

            if "grad_clip_norm" in cfg["training"]:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(cfg["training"]["grad_clip_norm"]),
                )

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            seen += images.size(0)

            progress.set_postfix(
                loss=running_loss / max(seen, 1),
                acc=correct / max(seen, 1),
            )

        scheduler.step()

        train_loss = running_loss / max(seen, 1)
        train_acc = correct / max(seen, 1)

        val = evaluate(model, loaders.val, device)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_loss={val.loss:.4f} "
            f"val_acc={val.accuracy:.4f}"
        )

        payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_accuracy": val.accuracy,
            "config": cfg,
        }

        save_checkpoint(cfg["checkpoints"]["last_path"], payload)

        if val.accuracy > best_val_acc:
            best_val_acc = val.accuracy
            best_epoch = epoch
            save_checkpoint(cfg["checkpoints"]["best_path"], payload)

    test = evaluate(model, loaders.test, device)

    print(
        f"final_test_loss={test.loss:.4f} "
        f"final_test_acc={test.accuracy:.4f} "
        f"best_val_acc={best_val_acc:.4f} "
        f"best_epoch={best_epoch}"
    )

    return {
        "best_val_accuracy": best_val_acc,
        "best_epoch": float(best_epoch),
        "final_test_loss": test.loss,
        "final_test_accuracy": test.accuracy,
    }