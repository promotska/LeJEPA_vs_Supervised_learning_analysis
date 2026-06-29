from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18, resnet50

from src.networks.heads import MLPHead


class ResNetBackbone(nn.Module):
    """ResNet backbone with CIFAR or ImageNet stem.

    For CIFAR experiments, use model_name='resnet18' and cifar_stem=True.
    For ImageNet-100/224 experiments, use cifar_stem=False to keep the standard
    7x7 stride-2 convolution and maxpool.
    """

    def __init__(self, model_name: str = "resnet18", cifar_stem: bool = True):
        super().__init__()
        model_name = model_name.lower()
        if model_name == "resnet18":
            net = resnet18(weights=None)
            self.feature_dim = 512
        elif model_name == "resnet50":
            net = resnet50(weights=None)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported ResNet model_name={model_name!r}")

        if cifar_stem:
            net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            net.maxpool = nn.Identity()

        net.fc = nn.Identity()
        self.net = net
        self.model_name = model_name
        self.cifar_stem = bool(cifar_stem)

        # Expose canonical children for Grad-CAM hook paths.
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.avgpool = net.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResNet18CifarBackbone(ResNetBackbone):
    """Backward-compatible CIFAR ResNet-18 backbone."""

    def __init__(self):
        super().__init__(model_name="resnet18", cifar_stem=True)


class SupervisedResNet(nn.Module):
    def __init__(self, num_classes: int, model_name: str = "resnet18", cifar_stem: bool = True):
        super().__init__()
        self.backbone = ResNetBackbone(model_name=model_name, cifar_stem=cifar_stem)
        self.classifier = nn.Linear(self.backbone.feature_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))


class SupervisedResNet18Cifar(SupervisedResNet):
    def __init__(self, num_classes: int = 10):
        super().__init__(num_classes=num_classes, model_name="resnet18", cifar_stem=True)


class LeJEPAResNet(nn.Module):
    """ResNet backbone + projector/predictor for faithful LeJEPA-SIGReg training."""

    def __init__(
        self,
        model_name: str = "resnet18",
        cifar_stem: bool = True,
        feature_dim: int | None = None,
        projection_dim: int = 256,
        prediction_dim: int = 512,
    ):
        super().__init__()
        self.backbone = ResNetBackbone(model_name=model_name, cifar_stem=cifar_stem)
        resolved_feature_dim = int(feature_dim or self.backbone.feature_dim)
        if resolved_feature_dim != self.backbone.feature_dim:
            raise ValueError(
                f"feature_dim={resolved_feature_dim} does not match backbone feature_dim={self.backbone.feature_dim}"
            )
        self.projector = MLPHead(resolved_feature_dim, prediction_dim, projection_dim)
        self.predictor = MLPHead(projection_dim, prediction_dim, projection_dim)

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward_projected(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(self.forward_backbone(x))

    def forward_views(self, views: list[torch.Tensor]) -> dict[str, list[torch.Tensor] | torch.Tensor]:
        z = [self.forward_projected(v) for v in views]
        p = [self.predictor(zi) for zi in z]
        return {"z": z, "p": p, "embeddings_for_sigreg": torch.cat(z, dim=0)}

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> dict[str, torch.Tensor]:
        # Backward-compatible two-view API for old code/checks.
        z1 = self.forward_projected(x1)
        z2 = self.forward_projected(x2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        return {"z1": z1, "z2": z2, "p1": p1, "p2": p2}


class LeJEPAResNet18Cifar(LeJEPAResNet):
    """Backward-compatible name for CIFAR ResNet-18 LeJEPA model."""

    def __init__(self, feature_dim: int = 512, projection_dim: int = 256, prediction_dim: int = 512):
        super().__init__(
            model_name="resnet18",
            cifar_stem=True,
            feature_dim=feature_dim,
            projection_dim=projection_dim,
            prediction_dim=prediction_dim,
        )


class LinearProbeResNet(nn.Module):
    def __init__(self, backbone: ResNetBackbone, num_classes: int = 10):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.feature_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))


class LinearProbeResNet18Cifar(LinearProbeResNet):
    def __init__(self, backbone: ResNet18CifarBackbone, num_classes: int = 10):
        super().__init__(backbone=backbone, num_classes=num_classes)
