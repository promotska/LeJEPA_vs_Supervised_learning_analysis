from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.networks.resnet import (
    LeJEPAResNet18Cifar,
    LinearProbeResNet18Cifar,
    ResNet18CifarBackbone,
    SupervisedResNet18Cifar,
)
from src.networks.vit import LeJEPAViTCifar, LinearProbeViTCifar, SupervisedViTCifar, ViTCifarBackbone
from src.training.checkpointing import load_checkpoint


def architecture_name(cfg: dict[str, Any]) -> str:
    return str(cfg.get("model", {}).get("architecture", "resnet18_cifar")).lower()


def build_supervised_model(cfg: dict[str, Any]) -> nn.Module:
    arch = architecture_name(cfg)
    model_cfg = cfg["model"]

    if arch == "resnet18_cifar":
        return SupervisedResNet18Cifar(num_classes=int(model_cfg["num_classes"]))

    if arch == "vit_tiny_cifar":
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

    raise ValueError(f"Unsupported supervised architecture: {arch}")


def build_lejepa_model(cfg: dict[str, Any]) -> nn.Module:
    arch = architecture_name(cfg)
    model_cfg = cfg["model"]

    if arch == "resnet18_cifar":
        return LeJEPAResNet18Cifar(
            feature_dim=int(model_cfg.get("feature_dim", 512)),
            projection_dim=int(model_cfg.get("projection_dim", 256)),
            prediction_dim=int(model_cfg.get("prediction_dim", 256)),
        )

    if arch == "vit_tiny_cifar":
        return LeJEPAViTCifar(
            image_size=int(model_cfg.get("image_size", 32)),
            patch_size=int(model_cfg.get("patch_size", 4)),
            embed_dim=int(model_cfg.get("embed_dim", 192)),
            depth=int(model_cfg.get("depth", 6)),
            num_heads=int(model_cfg.get("num_heads", 3)),
            mlp_ratio=float(model_cfg.get("mlp_ratio", 4.0)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            attn_dropout=float(model_cfg.get("attn_dropout", 0.1)),
            projection_dim=int(model_cfg.get("projection_dim", 256)),
            prediction_dim=int(model_cfg.get("prediction_dim", 512)),
        )

    raise ValueError(f"Unsupported LeJEPA architecture: {arch}")


def build_linear_probe_model(cfg: dict[str, Any], backbone: nn.Module) -> nn.Module:
    arch = architecture_name(cfg)
    num_classes = int(cfg["model"]["num_classes"])

    if arch == "resnet18_cifar":
        if not isinstance(backbone, ResNet18CifarBackbone):
            raise TypeError(f"Expected ResNet18CifarBackbone, got {type(backbone)}")
        return LinearProbeResNet18Cifar(backbone=backbone, num_classes=num_classes)

    if arch == "vit_tiny_cifar":
        if not isinstance(backbone, ViTCifarBackbone):
            raise TypeError(f"Expected ViTCifarBackbone, got {type(backbone)}")
        return LinearProbeViTCifar(backbone=backbone, num_classes=num_classes)

    raise ValueError(f"Unsupported probe architecture: {arch}")


def build_classifier_for_mode(cfg: dict[str, Any], mode: str) -> nn.Module:
    mode = mode.lower()
    arch = architecture_name(cfg)
    if mode in {"supervised", "vit_supervised"}:
        return build_supervised_model(cfg)
    if mode in {"lejepa", "vit_lejepa"}:
        if arch == "resnet18_cifar":
            backbone = ResNet18CifarBackbone()
        elif arch == "vit_tiny_cifar":
            model_cfg = cfg["model"]
            backbone = ViTCifarBackbone(
                image_size=int(model_cfg.get("image_size", 32)),
                patch_size=int(model_cfg.get("patch_size", 4)),
                in_channels=3,
                embed_dim=int(model_cfg.get("embed_dim", 192)),
                depth=int(model_cfg.get("depth", 6)),
                num_heads=int(model_cfg.get("num_heads", 3)),
                mlp_ratio=float(model_cfg.get("mlp_ratio", 4.0)),
                dropout=float(model_cfg.get("dropout", 0.1)),
                attn_dropout=float(model_cfg.get("attn_dropout", 0.1)),
            )
        else:
            raise ValueError(f"Unsupported architecture for LeJEPA classifier: {arch}")
        return build_linear_probe_model(cfg, backbone=backbone)
    raise ValueError(f"Unsupported mode: {mode}")


def load_classifier_from_checkpoint(
    cfg: dict[str, Any],
    mode: str,
    device: torch.device,
    checkpoint_path: str | Path | None = None,
    requires_grad: bool = False,
) -> nn.Module:
    ckpt_path = Path(checkpoint_path or cfg["evaluation"]["checkpoint_path"])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = load_checkpoint(ckpt_path, map_location=device)

    model = build_classifier_for_mode(cfg, mode=mode)
    model.load_state_dict(checkpoint["model_state"])

    for p in model.parameters():
        p.requires_grad_(requires_grad)

    return model.to(device).eval()
