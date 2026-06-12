from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18

from src.networks.heads import MLPHead


class ResNet18CifarBackbone(nn.Module):
    """ResNet-18 adapted for CIFAR-10 resolution.

    The first convolution is 3x3 stride 1 and maxpool is removed.
    Forward returns the global pooled 512-dim representation.
    Layer modules remain available as backbone.layer1 ... backbone.layer4 for hooks.
    """

    def __init__(self):
        super().__init__()
        net = resnet18(weights=None)
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()
        net.fc = nn.Identity()
        self.net = net

        # Expose canonical children for simple hook paths.
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


class SupervisedResNet18Cifar(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.backbone = ResNet18CifarBackbone()
        self.classifier = nn.Linear(512, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))


class LeJEPAResNet18Cifar(nn.Module):
    """Minimal LeJEPA-style model for Stage 1.

    This is intentionally compact: a ResNet backbone plus projection/prediction heads.
    The loss in src.training.losses provides the SIGReg-style regularization.
    """

    def __init__(self, feature_dim: int = 512, projection_dim: int = 256, prediction_dim: int = 256):
        super().__init__()
        self.backbone = ResNet18CifarBackbone()
        self.projector = MLPHead(feature_dim, prediction_dim, projection_dim)
        self.predictor = MLPHead(projection_dim, prediction_dim, projection_dim)

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward_projected(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(self.forward_backbone(x))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> dict[str, torch.Tensor]:
        z1 = self.forward_projected(x1)
        z2 = self.forward_projected(x2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        return {"z1": z1, "z2": z2, "p1": p1, "p2": p2}


class LinearProbeResNet18Cifar(nn.Module):
    """Classifier over a LeJEPA backbone.

    During probe training the backbone can be frozen. During Grad-CAM evaluation the
    backbone can be temporarily set to requires_grad=True to obtain activation gradients
    without changing its weights.
    """

    def __init__(self, backbone: ResNet18CifarBackbone, num_classes: int = 10):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(512, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))
