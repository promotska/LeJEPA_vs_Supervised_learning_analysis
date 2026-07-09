from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class PopulationPCABasis:
    mean: torch.Tensor        # [C]
    components: torch.Tensor  # [K, C], unit-norm rows


@torch.no_grad()
def fit_population_pca_from_activations(
    activations: list[torch.Tensor],
    num_components: int = 8,
) -> PopulationPCABasis:
    """Fit a shared channel-space PCA basis from pooled [B,C,H,W] activations.

    Every spatial position of every sampled image is one observation in R^C.
    Unlike the per-image PCA in pca_masks.py, this basis is fixed, so
    "component 0" is the same direction for every image and every model —
    which is what you need to compare supervised vs. LeJEPA on a common axis
    instead of each getting its own private one.
    """
    vectors = [
        act.detach().float().permute(0, 2, 3, 1).reshape(-1, act.shape[1]).cpu()
        for act in activations
    ]
    x = torch.cat(vectors, dim=0)  # [N, C]
    mean = x.mean(dim=0)
    xc = x - mean
    n, c = xc.shape
    k = min(num_components, n - 1, c)

    if n <= c:
        gram = xc @ xc.transpose(0, 1)
        eigvals, eigvecs = torch.linalg.eigh(gram)
        order = eigvals.argsort(descending=True)[:k]
        components = torch.stack(
            [F.normalize(xc.transpose(0, 1) @ eigvecs[:, idx], dim=0) for idx in order],
            dim=0,
        )
    else:
        cov = xc.transpose(0, 1) @ xc
        eigvals, eigvecs = torch.linalg.eigh(cov)
        order = eigvals.argsort(descending=True)[:k]
        components = F.normalize(eigvecs[:, order].transpose(0, 1), dim=1)

    return PopulationPCABasis(mean=mean, components=components)


def _normalize_map(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    return (values - values.min()) / (values.max() - values.min() + 1e-8)


@torch.no_grad()
def population_pca_heatmaps_from_activation(
    activation: torch.Tensor,
    basis: PopulationPCABasis,
    output_size: tuple[int, int],
    component: int = 0,
    use_abs: bool = True,
) -> np.ndarray:
    """Drop-in replacement for pca_heatmaps_from_activation, using a fixed basis."""
    b, c, h, w = activation.shape
    mean = basis.mean.to(activation.device, activation.dtype)
    comp = basis.components[component].to(activation.device, activation.dtype)

    maps = []
    for i in range(b):
        feat = activation[i].detach().float()
        x = feat.reshape(c, h * w).transpose(0, 1) - mean
        score = (x * comp).sum(dim=-1).reshape(h, w)
        if use_abs:
            score = score.abs()
        heatmap = F.interpolate(score[None, None], size=output_size, mode="bilinear", align_corners=False)[0, 0]
        maps.append(_normalize_map(heatmap).cpu().numpy())
    return np.stack(maps, axis=0)