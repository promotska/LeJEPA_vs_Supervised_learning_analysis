from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.data.dataloaders import build_loaders
from src.networks.factory import build_supervised_model
from src.training.checkpointing import save_checkpoint
from src.utils import get_device, load_yaml, set_seed


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> EpochMetrics:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = loss_fn(logits, labels)

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_count += batch_size

    return EpochMetrics(
        loss=total_loss / max(total_count, 1),
        accuracy=total_correct / max(total_count, 1),
    )


def _build_optimizer(cfg: dict[str, Any], model: nn.Module) -> torch.optim.Optimizer:
    training_cfg = cfg["training"]
    optimizer_name = str(training_cfg.get("optimizer", "adamw")).lower()
    lr = float(training_cfg["learning_rate"])
    weight_decay = float(training_cfg.get("weight_decay", 0.0))

    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(training_cfg.get("momentum", 0.9)),
            weight_decay=weight_decay,
            nesterov=bool(training_cfg.get("nesterov", True)),
        )

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def _build_scheduler(cfg: dict[str, Any], optimizer: torch.optim.Optimizer):
    training_cfg = cfg["training"]
    scheduler_name = str(training_cfg.get("scheduler", "cosine")).lower()

    if scheduler_name in {"none", "off", "false"}:
        return None

    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(training_cfg["epochs"]),
        )

    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def train_supervised(config_path: str) -> dict[str, float]:
    """
    Generic supervised training entry point.

    It is intentionally model/dataset-agnostic. The exact architecture is read
    from config["model"]["architecture"] and built by src.networks.factory.

    Supported examples:
      - resnet18_cifar on CIFAR-10 / CIFAR-100
      - resnet18_imagenet / resnet50_imagenet on ImageFolder/ImageNet-100
      - vit_tiny_cifar / vit_small_patch16_224 through the same factory

    This replaces the old hardcoded SupervisedResNet18Cifar path.
    """
    cfg = load_yaml(config_path)
    set_seed(int(cfg["project"]["seed"]))
    device = get_device()

    loaders = build_loaders(cfg, self_supervised=False)
    model = build_supervised_model(cfg).to(device)

    optimizer = _build_optimizer(cfg, model)
    scheduler = _build_scheduler(cfg, optimizer)
    loss_fn = nn.CrossEntropyLoss()

    use_amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    best_val_acc = 0.0
    best_epoch = 0
    last_train_loss = 0.0
    last_train_acc = 0.0

    print("Supervised training settings:")
    print(f"  config: {config_path}")
    print(f"  architecture: {cfg['model'].get('architecture')}")
    print(f"  dataset: {cfg['data'].get('dataset')}")
    print(f"  device: {device}")
    print(f"  amp: {use_amp}")
    print(f"  epochs: {cfg['training']['epochs']}")

    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        model.train()

        running_loss = 0.0
        correct = 0
        seen = 0

        progress = tqdm(loaders.train, desc=f"supervised epoch {epoch}", leave=False)

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

            batch_size = images.size(0)
            running_loss += float(loss.item()) * batch_size
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            seen += batch_size

            progress.set_postfix(
                loss=running_loss / max(seen, 1),
                acc=correct / max(seen, 1),
            )

        if scheduler is not None:
            scheduler.step()

        last_train_loss = running_loss / max(seen, 1)
        last_train_acc = correct / max(seen, 1)
        val = evaluate(model, loaders.val, device)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={last_train_loss:.4f} "
            f"train_acc={last_train_acc:.4f} "
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
        "best_val_accuracy": float(best_val_acc),
        "best_epoch": float(best_epoch),
        "final_test_loss": float(test.loss),
        "final_test_accuracy": float(test.accuracy),
        "last_train_loss": float(last_train_loss),
        "last_train_accuracy": float(last_train_acc),
    }
