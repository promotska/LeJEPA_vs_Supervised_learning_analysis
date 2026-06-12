from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA


def normalize_map(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    values = values.astype(np.float32)
    min_v = float(values.min())
    max_v = float(values.max())
    return (values - min_v) / (max_v - min_v + eps)


def pca_heatmap_from_activation(
    activation: torch.Tensor,
    output_size: tuple[int, int],
    component: int = 0,
    use_abs: bool = True,
) -> np.ndarray:
    """Convert a CNN activation tensor into one PCA heatmap.

    Args:
        activation: Tensor shaped [1, C, H, W] or [C, H, W].
        output_size: Final heatmap size, usually input image H,W.
        component: PCA component index.
        use_abs: If true, use absolute component values. This avoids arbitrary PCA sign flips.
    """
    if activation.ndim == 4:
        if activation.shape[0] != 1:
            raise ValueError("pca_heatmap_from_activation expects a single image activation with batch size 1.")
        activation = activation[0]
    if activation.ndim != 3:
        raise ValueError(f"Expected activation [C,H,W], got shape {tuple(activation.shape)}")

    c, h, w = activation.shape
    spatial_features = activation.detach().float().cpu().permute(1, 2, 0).reshape(h * w, c).numpy()
    spatial_features = spatial_features - spatial_features.mean(axis=0, keepdims=True)

    pca = PCA(n_components=max(component + 1, 1), random_state=0)
    pcs = pca.fit_transform(spatial_features)
    heat = pcs[:, component].reshape(h, w)
    if use_abs:
        heat = np.abs(heat)
    heat = normalize_map(heat)

    heat_t = torch.from_numpy(heat)[None, None]
    upsampled = F.interpolate(heat_t, size=output_size, mode="bilinear", align_corners=False)[0, 0]
    return normalize_map(upsampled.numpy())


def topk_binary_mask(heatmap: np.ndarray, keep_ratio: float = 0.2) -> np.ndarray:
    if not 0.0 < keep_ratio < 1.0:
        raise ValueError("keep_ratio must be in (0,1).")
    threshold = np.quantile(heatmap, 1.0 - keep_ratio)
    return (heatmap >= threshold).astype(np.float32)
