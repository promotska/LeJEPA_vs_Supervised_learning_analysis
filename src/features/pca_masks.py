from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _normalize_map(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    min_v = values.min()
    max_v = values.max()
    return (values - min_v) / (max_v - min_v + 1e-8)


@torch.no_grad()
def _pca_heatmap_single(
    feature: torch.Tensor,
    component: int = 0,
    use_abs: bool = True,
) -> torch.Tensor:
    """
    feature: [C, H, W]
    returns: [H, W] normalized to [0, 1]
    """
    if feature.ndim != 3:
        raise ValueError(f"Expected feature [C,H,W], got shape {tuple(feature.shape)}")

    c, h, w = feature.shape
    x = feature.reshape(c, h * w).transpose(0, 1).float()  # [N, C]
    x = x - x.mean(dim=0, keepdim=True)

    if x.numel() == 0 or torch.allclose(x.abs().sum(), torch.tensor(0.0, device=x.device)):
        return torch.zeros((h, w), device=feature.device, dtype=torch.float32)

    n, c = x.shape
    max_component = min(n, c) - 1
    if component > max_component:
        component = max_component

    try:
        # More efficient route depending on whether spatial positions or channels are fewer.
        if n <= c:
            gram = x @ x.transpose(0, 1)  # [N, N]
            eigvals, eigvecs = torch.linalg.eigh(gram)
            idx = eigvals.argsort(descending=True)[component]
            score = eigvecs[:, idx] * torch.sqrt(torch.clamp(eigvals[idx], min=0.0))
        else:
            cov = x.transpose(0, 1) @ x  # [C, C]
            eigvals, eigvecs = torch.linalg.eigh(cov)
            idx = eigvals.argsort(descending=True)[component]
            pc = eigvecs[:, idx]
            score = x @ pc
    except RuntimeError:
        # Fallback to CPU if GPU eigendecomposition fails.
        x_cpu = x.detach().cpu()
        if n <= c:
            gram = x_cpu @ x_cpu.transpose(0, 1)
            eigvals, eigvecs = torch.linalg.eigh(gram)
            idx = eigvals.argsort(descending=True)[component]
            score_cpu = eigvecs[:, idx] * torch.sqrt(torch.clamp(eigvals[idx], min=0.0))
        else:
            cov = x_cpu.transpose(0, 1) @ x_cpu
            eigvals, eigvecs = torch.linalg.eigh(cov)
            idx = eigvals.argsort(descending=True)[component]
            pc = eigvecs[:, idx]
            score_cpu = x_cpu @ pc
        score = score_cpu.to(feature.device)

    if use_abs:
        score = score.abs()

    heatmap = score.reshape(h, w)
    return _normalize_map(heatmap)


@torch.no_grad()
def pca_heatmaps_from_activation(
    activation: torch.Tensor,
    output_size: tuple[int, int],
    component: int = 0,
    use_abs: bool = True,
) -> np.ndarray:
    """
    activation: [B, C, H, W]
    returns numpy [B, output_H, output_W]
    """
    if activation.ndim != 4:
        raise ValueError(f"Expected activation [B,C,H,W], got {tuple(activation.shape)}")

    activation = activation.detach()
    maps = []

    for i in range(activation.shape[0]):
        heatmap = _pca_heatmap_single(
            feature=activation[i],
            component=component,
            use_abs=use_abs,
        )
        heatmap = heatmap[None, None, :, :]
        heatmap = F.interpolate(
            heatmap,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        heatmap = _normalize_map(heatmap)
        maps.append(heatmap.detach().cpu().numpy())

    return np.stack(maps, axis=0)


@torch.no_grad()
def pca_heatmap_from_activation(
    activation: torch.Tensor,
    output_size: tuple[int, int],
    component: int = 0,
    use_abs: bool = True,
) -> np.ndarray:
    """
    Backward-compatible single-map function.

    Accepts:
        [B,C,H,W] where B can be 1
        [C,H,W]

    Returns:
        numpy [output_H, output_W]
    """
    if activation.ndim == 3:
        activation = activation.unsqueeze(0)

    maps = pca_heatmaps_from_activation(
        activation=activation,
        output_size=output_size,
        component=component,
        use_abs=use_abs,
    )
    return maps[0]