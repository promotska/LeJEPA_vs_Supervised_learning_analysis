from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.data.dataloaders import build_loaders
from src.networks.resnet import SupervisedResNet18Cifar
from src.training.checkpointing import save_checkpoint
from src.utils import get_device, load_yaml, set_seed


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float


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
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_count += images.size(0)
    return EpochMetrics(loss=total_loss / total_count, accuracy=total_correct / total_count)


def train_supervised(config_path: str) -> None:
    cfg = load_yaml(config_path)
    set_seed(int(cfg["project"]["seed"]))
    device = get_device()

    loaders = build_loaders(cfg, self_supervised=False)

    model = SupervisedResNet18Cifar(num_classes=int(cfg["model"]["num_classes"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg["training"]["epochs"]))
    loss_fn = nn.CrossEntropyLoss()
    use_amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    best_val_acc = 0.0
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
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            seen += images.size(0)
            progress.set_postfix(loss=running_loss / seen, acc=correct / seen)

        scheduler.step()
        val = evaluate(model, loaders.val, device)
        print(f"epoch={epoch:03d} train_loss={running_loss/seen:.4f} train_acc={correct/seen:.4f} val_loss={val.loss:.4f} val_acc={val.accuracy:.4f}")

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
            save_checkpoint(cfg["checkpoints"]["best_path"], payload)

    test = evaluate(model, loaders.test, device)
    print(f"final_test_loss={test.loss:.4f} final_test_acc={test.accuracy:.4f}")
