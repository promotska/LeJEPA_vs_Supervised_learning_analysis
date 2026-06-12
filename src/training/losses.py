from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    if n != m:
        raise ValueError("off_diagonal expects a square matrix.")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class SigRegStyleLoss(nn.Module):
    """Small, practical SIGReg-style regularizer for Stage 1.

    This is not a claim of exact reproduction of the LeJEPA paper implementation.
    It encourages non-collapsed, approximately isotropic embeddings by penalizing:
    - non-zero batch mean
    - dimensions with too little standard deviation
    - off-diagonal covariance
    """

    def __init__(self, mean_weight: float = 1.0, std_weight: float = 1.0, cov_weight: float = 0.04, eps: float = 1e-4):
        super().__init__()
        self.mean_weight = mean_weight
        self.std_weight = std_weight
        self.cov_weight = cov_weight
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z.float()
        mean_loss = z.mean(dim=0).pow(2).mean()

        z_centered = z - z.mean(dim=0, keepdim=True)
        std = torch.sqrt(z_centered.var(dim=0, unbiased=False) + self.eps)
        std_loss = F.relu(1.0 - std).mean()

        if z.shape[0] <= 1:
            cov_loss = z.new_tensor(0.0)
        else:
            cov = (z_centered.T @ z_centered) / (z.shape[0] - 1)
            cov_loss = off_diagonal(cov).pow(2).mean()

        return self.mean_weight * mean_loss + self.std_weight * std_loss + self.cov_weight * cov_loss


class LeJEPALoss(nn.Module):
    """Symmetric joint-embedding prediction loss with SIGReg-style anti-collapse regularization."""

    def __init__(self, sigreg_weight: float = 0.05):
        super().__init__()
        self.sigreg = SigRegStyleLoss()
        self.sigreg_weight = sigreg_weight

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor) -> dict[str, torch.Tensor]:
        z1n = F.normalize(z1, dim=1)
        z2n = F.normalize(z2, dim=1)
        p1n = F.normalize(p1, dim=1)
        p2n = F.normalize(p2, dim=1)

        prediction_loss = 0.5 * (F.mse_loss(p1n, z2n) + F.mse_loss(p2n, z1n))
        regularization_loss = 0.5 * (self.sigreg(z1) + self.sigreg(z2))
        total = prediction_loss + self.sigreg_weight * regularization_loss
        return {
            "loss": total,
            "prediction_loss": prediction_loss.detach(),
            "sigreg_loss": regularization_loss.detach(),
        }
