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
) -> dict[str, float]:
    """Stage-1 LSAS for CIFAR-10 without ground-truth masks.

    This measures PCA↔XAI agreement only. It is useful for debugging and the first
    comparison, but it is not yet the grounded G-LSAS from the full project.
    """
    corr = pearson_corr(pca_map, xai_map)
    overlap = soft_iou(pca_map, xai_map)
    score = corr_weight * corr + soft_iou_weight * overlap
    return {"corr_pca_xai": corr, "soft_iou_pca_xai": overlap, "lsas": float(score)}


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
