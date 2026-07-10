from __future__ import annotations

import numpy as np


def safe_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + eps)


def pearson_corr(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a = safe_normalize(a).reshape(-1)
    b = safe_normalize(b).reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()) + eps)
    return float((a * b).sum() / denom)

def _histogram2d_probs(a: np.ndarray, b: np.ndarray, bins: int = 32) -> np.ndarray:
    a = safe_normalize(a).reshape(-1)
    b = safe_normalize(b).reshape(-1)
    hist, _, _ = np.histogram2d(a, b, bins=bins, range=[[0, 1], [0, 1]])
    return hist / (hist.sum() + 1e-12)


def mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 32, eps: float = 1e-12) -> float:
    """Histogram-based MI between two normalized spatial maps (nats)."""
    p_xy = _histogram2d_probs(a, b, bins=bins)
    p_x = p_xy.sum(axis=1, keepdims=True)
    p_y = p_xy.sum(axis=0, keepdims=True)
    p_ind = p_x @ p_y
    nz = p_xy > 0
    return float((p_xy[nz] * np.log(p_xy[nz] / (p_ind[nz] + eps) + eps)).sum())


def normalized_mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 32, eps: float = 1e-12) -> float:
    """NMI = MI / H(X,Y), bounded to [0, 1]. Comparable across layers with different entropy."""
    p_xy = _histogram2d_probs(a, b, bins=bins)
    nz = p_xy > 0
    h_xy = float(-(p_xy[nz] * np.log(p_xy[nz] + eps)).sum())
    if h_xy <= eps:
        return 0.0
    return float(np.clip(mutual_information(a, b, bins=bins, eps=eps) / h_xy, 0.0, 1.0))

def soft_iou(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a = safe_normalize(a)
    b = safe_normalize(b)
    intersection = np.minimum(a, b).sum()
    union = np.maximum(a, b).sum()
    return float(intersection / (union + eps))


def binary_iou(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a = (a > 0).astype(np.float32)
    b = (b > 0).astype(np.float32)
    intersection = (a * b).sum()
    union = ((a + b) > 0).sum()
    return float(intersection / (union + eps))


def lsas(
    pca_map: np.ndarray,
    xai_map: np.ndarray,
    corr_weight: float = 0.5,
    soft_iou_weight: float = 0.5,
    mi_weight: float = 0.0,
    mi_bins: int = 32,
    report_mi: bool = True,          # <-- NMI reported as its own column, score unchanged
) -> dict[str, float]:
    corr = pearson_corr(pca_map, xai_map)
    overlap = soft_iou(pca_map, xai_map)
    result = {"corr_pca_xai": corr, "soft_iou_pca_xai": overlap}
    score = corr_weight * corr + soft_iou_weight * overlap
    if report_mi or mi_weight > 0.0:
        nmi = normalized_mutual_information(pca_map, xai_map, bins=mi_bins)
        result["nmi_pca_xai"] = nmi
        if mi_weight > 0.0:
            score += mi_weight * nmi
    result["lsas"] = float(score)
    return result

def g_lsas(
    pca_map: np.ndarray,
    xai_map: np.ndarray,
    gt_mask: np.ndarray,
    pca_aug_map: np.ndarray | None = None,
) -> dict[str, float]:
    """Grounded LSAS for later stages with segmentation masks."""
    corr = pearson_corr(pca_map, xai_map)
    pca_gt = soft_iou(pca_map, gt_mask)
    xai_gt = soft_iou(xai_map, gt_mask)
    stability = pearson_corr(pca_map, pca_aug_map) if pca_aug_map is not None else 0.0
    score = 0.35 * corr + 0.25 * pca_gt + 0.25 * xai_gt + 0.15 * stability
    return {
        "corr_pca_xai": corr,
        "soft_iou_pca_gt": pca_gt,
        "soft_iou_xai_gt": xai_gt,
        "stability": stability,
        "g_lsas": float(score),
    }
