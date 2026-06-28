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
    """Backward-compatible Stage 1 covariance/isotropy regularizer.

    This is intentionally kept for old Stage 1 checkpoints. It is not exact
    LeJEPA SIGReg and should not be used as the final scientific LeJEPA method.
    """

    def __init__(
        self,
        mean_weight: float = 1.0,
        std_weight: float = 1.0,
        cov_weight: float = 0.04,
        eps: float = 1e-4,
    ):
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


class OfficialSIGRegLoss(nn.Module):
    """Official LeJEPA SIGReg wrapper.

    Requires the external package:
        pip install lejepa
    """

    def __init__(self, num_slices: int = 1024, num_points: int = 17):
        super().__init__()
        try:
            import lejepa
        except ImportError as exc:
            raise ImportError(
                "OfficialSIGRegLoss requires the official LeJEPA package. "
                "Install it in the CINECA venv with: pip install lejepa"
            ) from exc

        univariate_test = lejepa.univariate.EppsPulley(num_points=num_points)
        self.loss_fn = lejepa.multivariate.SlicingUnivariateTest(
            univariate_test=univariate_test,
            num_slices=num_slices,
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2:
            raise ValueError(f"Expected embeddings [N,D], got {tuple(embeddings.shape)}")
        return self.loss_fn(embeddings.float())


class FallbackSlicedGaussianMomentLoss(nn.Module):
    """Debug fallback approximation to SIGReg.

    This is not exact SIGReg. Use it only when the official package is not
    available and mark the experiment accordingly.
    """

    def __init__(
        self,
        num_slices: int = 1024,
        mean_weight: float = 1.0,
        var_weight: float = 1.0,
        skew_weight: float = 0.1,
        kurt_weight: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_slices = int(num_slices)
        self.mean_weight = mean_weight
        self.var_weight = var_weight
        self.skew_weight = skew_weight
        self.kurt_weight = kurt_weight
        self.eps = eps

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2:
            raise ValueError(f"Expected embeddings [N,D], got {tuple(embeddings.shape)}")
        z = embeddings.float()
        _, d = z.shape
        directions = torch.randn(d, self.num_slices, device=z.device, dtype=z.dtype)
        directions = F.normalize(directions, dim=0)
        projected = z @ directions
        mean = projected.mean(dim=0)
        var = projected.var(dim=0, unbiased=False)
        centered = projected - mean
        std = torch.sqrt(var + self.eps)
        skew = (centered / std).pow(3).mean(dim=0)
        kurt = (centered / std).pow(4).mean(dim=0)
        return (
            self.mean_weight * mean.pow(2).mean()
            + self.var_weight * (var - 1.0).pow(2).mean()
            + self.skew_weight * skew.pow(2).mean()
            + self.kurt_weight * (kurt - 3.0).pow(2).mean()
        )


class LeJEPALoss(nn.Module):
    """Backward-compatible symmetric prediction loss for Stage 1."""

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
        return {"loss": total, "prediction_loss": prediction_loss.detach(), "sigreg_loss": regularization_loss.detach()}


class MultiViewLeJEPASIGRegLoss(nn.Module):
    """Multi-view JEPA prediction loss + SIGReg.

    Given projections z_i and predictions p_i for each view, each prediction is
    trained to predict the mean of the other view projections. No stop-gradient is
    applied here. SIGReg regularizes all projected embeddings.
    """

    def __init__(
        self,
        sigreg: nn.Module,
        prediction_weight: float = 1.0,
        sigreg_weight: float = 0.05,
        normalize_prediction: bool = False,
    ):
        super().__init__()
        self.sigreg = sigreg
        self.prediction_weight = prediction_weight
        self.sigreg_weight = sigreg_weight
        self.normalize_prediction = normalize_prediction

    def forward(
        self,
        projections: list[torch.Tensor],
        predictions: list[torch.Tensor],
        embeddings_for_sigreg: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if len(projections) != len(predictions):
            raise ValueError("projections and predictions must have same length")
        if len(projections) < 2:
            raise ValueError("At least two views are required")

        pred_loss = embeddings_for_sigreg.new_tensor(0.0)
        for i, pred in enumerate(predictions):
            targets = [z for j, z in enumerate(projections) if j != i]
            target = torch.stack(targets, dim=0).mean(dim=0)
            if self.normalize_prediction:
                pred = F.normalize(pred, dim=-1)
                target = F.normalize(target, dim=-1)
            pred_loss = pred_loss + F.mse_loss(pred, target)
        pred_loss = pred_loss / len(predictions)

        sigreg_loss = self.sigreg(embeddings_for_sigreg)
        total = self.prediction_weight * pred_loss + self.sigreg_weight * sigreg_loss
        return {
            "loss": total,
            "prediction_loss": pred_loss.detach(),
            "sigreg_loss": sigreg_loss.detach(),
        }


def build_sigreg_from_config(cfg: dict) -> nn.Module:
    lejepa_cfg = cfg.get("lejepa", {})
    impl = str(lejepa_cfg.get("sigreg_implementation", "official")).lower()
    num_slices = int(lejepa_cfg.get("sigreg_num_slices", 1024))
    num_points = int(lejepa_cfg.get("sigreg_num_points", 17))

    if impl in {"official", "lejepa"}:
        return OfficialSIGRegLoss(num_slices=num_slices, num_points=num_points)
    if impl in {"fallback", "moments", "debug"}:
        return FallbackSlicedGaussianMomentLoss(num_slices=num_slices)
    if impl in {"stage1", "covariance", "vicreg_style"}:
        return SigRegStyleLoss()
    raise ValueError(f"Unknown sigreg_implementation: {impl}")
