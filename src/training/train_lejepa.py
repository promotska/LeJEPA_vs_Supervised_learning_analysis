from __future__ import annotations

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.data.dataloaders import build_loaders
from src.networks.resnet import LeJEPAResNet18Cifar, LinearProbeResNet18Cifar
from src.training.checkpointing import save_checkpoint
from src.training.losses import LeJEPALoss
from src.training.train_supervised import evaluate
from src.utils import get_device, load_yaml, set_seed


def pretrain_lejepa(cfg: dict, device: torch.device) -> LeJEPAResNet18Cifar:
    loaders = build_loaders(cfg, self_supervised=True)

    model = LeJEPAResNet18Cifar(
        feature_dim=int(cfg["model"]["feature_dim"]),
        projection_dim=int(cfg["model"]["projection_dim"]),
        prediction_dim=int(cfg["model"]["prediction_dim"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg["training"]["pretrain_epochs"]))
    criterion = LeJEPALoss(sigreg_weight=float(cfg["training"].get("sigreg_weight", 0.05)))
    use_amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    best_loss = float("inf")
    for epoch in range(1, int(cfg["training"]["pretrain_epochs"]) + 1):
        model.train()
        total_loss = 0.0
        total_prediction = 0.0
        total_sigreg = 0.0
        seen = 0
        progress = tqdm(loaders.train, desc=f"lejepa pretrain epoch {epoch}", leave=False)
        for views, _ in progress:
            x1, x2 = views
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                outputs = model(x1, x2)
                losses = criterion(**outputs)
                loss = losses["loss"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = x1.size(0)
            total_loss += loss.item() * batch_size
            total_prediction += losses["prediction_loss"].item() * batch_size
            total_sigreg += losses["sigreg_loss"].item() * batch_size
            seen += batch_size
            progress.set_postfix(loss=total_loss / seen, pred=total_prediction / seen, sig=total_sigreg / seen)

        scheduler.step()
        avg_loss = total_loss / seen
        print(f"epoch={epoch:03d} lejepa_loss={avg_loss:.4f} pred={total_prediction/seen:.4f} sigreg={total_sigreg/seen:.4f}")

        payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "backbone_state": model.backbone.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "pretrain_loss": avg_loss,
            "config": cfg,
        }
        save_checkpoint(cfg["checkpoints"]["backbone_last_path"], payload)
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(cfg["checkpoints"]["backbone_best_path"], payload)

    return model


def train_linear_probe(cfg: dict, pretrained: LeJEPAResNet18Cifar, device: torch.device) -> LinearProbeResNet18Cifar:
    loaders = build_loaders(cfg, self_supervised=False)

    model = LinearProbeResNet18Cifar(pretrained.backbone, num_classes=int(cfg["model"]["num_classes"])).to(device)
    for p in model.backbone.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=float(cfg["training"]["probe_learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg["training"]["probe_epochs"]))
    loss_fn = nn.CrossEntropyLoss()
    use_amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    best_val_acc = 0.0
    for epoch in range(1, int(cfg["training"]["probe_epochs"]) + 1):
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

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            seen += images.size(0)
            progress.set_postfix(loss=total_loss / seen, acc=correct / seen)

        scheduler.step()
        val = evaluate(model, loaders.val, device)
        print(f"probe_epoch={epoch:03d} train_loss={total_loss/seen:.4f} train_acc={correct/seen:.4f} val_acc={val.accuracy:.4f}")

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
            save_checkpoint(cfg["checkpoints"]["probe_best_path"], payload)

    test = evaluate(model, loaders.test, device)
    print(f"linear_probe_test_loss={test.loss:.4f} linear_probe_test_acc={test.accuracy:.4f}")
    return model


def train_lejepa(config_path: str) -> None:
    cfg = load_yaml(config_path)
    set_seed(int(cfg["project"]["seed"]))
    device = get_device()
    pretrained = pretrain_lejepa(cfg, device)
    train_linear_probe(cfg, pretrained, device)
