from __future__ import annotations

import numpy as np

from src.evaluation.sas import pearson_corr, safe_normalize, soft_iou


def _valid_bool(valid_mask: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
    if valid_mask is None:
        return np.ones(shape, dtype=bool)
    return valid_mask.astype(bool)


def masked_array(x: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    x = x.astype(np.float32)
    valid = _valid_bool(valid_mask, x.shape)
    out = np.zeros_like(x, dtype=np.float32)
    out[valid] = x[valid]
    return out


def masked_pearson_corr(a: np.ndarray, b: np.ndarray, valid_mask: np.ndarray | None = None, eps: float = 1e-8) -> float:
    valid = _valid_bool(valid_mask, a.shape)
    if valid.sum() < 2:
        return 0.0
    aa = safe_normalize(a.astype(np.float32))[valid].reshape(-1)
    bb = safe_normalize(b.astype(np.float32))[valid].reshape(-1)
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    denom = float(np.sqrt((aa * aa).sum()) * np.sqrt((bb * bb).sum()) + eps)
    return float((aa * bb).sum() / denom)


def masked_soft_iou(a: np.ndarray, b: np.ndarray, valid_mask: np.ndarray | None = None, eps: float = 1e-8) -> float:
    a = masked_array(a, valid_mask)
    b = masked_array(b, valid_mask)
    return soft_iou(a, b, eps=eps)


def topk_binary_map(x: np.ndarray, valid_mask: np.ndarray | None = None, top_frac: float = 0.20) -> np.ndarray:
    x = safe_normalize(x.astype(np.float32))
    valid = _valid_bool(valid_mask, x.shape)
    out = np.zeros_like(x, dtype=np.float32)

    values = x[valid]
    if values.size == 0:
        return out

    top_frac = float(np.clip(top_frac, 1e-4, 1.0))
    k = max(1, int(round(values.size * top_frac)))
    threshold = np.partition(values, -k)[-k]
    out[valid] = (x[valid] >= threshold).astype(np.float32)
    return out


def masked_binary_iou(a: np.ndarray, b: np.ndarray, valid_mask: np.ndarray | None = None, eps: float = 1e-8) -> float:
    valid = _valid_bool(valid_mask, a.shape)
    aa = (a > 0).astype(bool) & valid
    bb = (b > 0).astype(bool) & valid
    inter = np.logical_and(aa, bb).sum(dtype=np.float64)
    union = np.logical_or(aa, bb).sum(dtype=np.float64)
    return float(inter / (union + eps))


def object_focus_ratio(heatmap: np.ndarray, gt_mask: np.ndarray, valid_mask: np.ndarray | None = None, eps: float = 1e-8) -> float:
    valid = _valid_bool(valid_mask, heatmap.shape)
    h = safe_normalize(heatmap.astype(np.float32))
    h = h * valid.astype(np.float32)
    gt = (gt_mask > 0).astype(bool) & valid
    return float(h[gt].sum() / (h.sum() + eps))


def foreground_area(gt_mask: np.ndarray, valid_mask: np.ndarray | None = None, eps: float = 1e-8) -> float:
    valid = _valid_bool(valid_mask, gt_mask.shape)
    gt = (gt_mask > 0).astype(bool) & valid
    return float(gt.sum() / (valid.sum() + eps))


def grounded_lsas(
    pca_map: np.ndarray,
    xai_map: np.ndarray,
    gt_mask: np.ndarray,
    valid_mask: np.ndarray | None = None,
    pca_aug_map: np.ndarray | None = None,
    top_frac: float = 0.20,
    corr_weight: float = 0.35,
    pca_gt_weight: float = 0.25,
    xai_gt_weight: float = 0.25,
    stability_weight: float = 0.15,
) -> dict[str, float]:
    """Grounded Layer-wise Semantic Alignment Score.

    G-LSAS_l =
        corr_weight * Corr(PCA_l, XAI_l)
      + pca_gt_weight * IoU(top-k PCA_l, GT)
      + xai_gt_weight * IoU(top-k XAI_l, GT)
      + stability_weight * Corr(PCA_l, PCA_l_aug)

    Heatmap-to-GT IoU uses a top-fraction threshold so PCA and XAI maps are
    binarized at comparable saliency mass. The same top_frac must be used for
    all models in the final comparison.
    """
    valid = _valid_bool(valid_mask, pca_map.shape)
    gt = (gt_mask > 0).astype(np.float32)

    corr = masked_pearson_corr(pca_map, xai_map, valid)
    pca_xai_soft = masked_soft_iou(pca_map, xai_map, valid)

    pca_bin = topk_binary_map(pca_map, valid, top_frac=top_frac)
    xai_bin = topk_binary_map(xai_map, valid, top_frac=top_frac)

    pca_gt_iou = masked_binary_iou(pca_bin, gt, valid)
    xai_gt_iou = masked_binary_iou(xai_bin, gt, valid)
    pca_gt_soft_iou = masked_soft_iou(pca_map, gt, valid)
    xai_gt_soft_iou = masked_soft_iou(xai_map, gt, valid)

    pca_focus = object_focus_ratio(pca_map, gt, valid)
    xai_focus = object_focus_ratio(xai_map, gt, valid)

    if pca_aug_map is None:
        stability = 0.0
        g_lsas_no_stability = (
            0.40 * corr
            + 0.30 * pca_gt_iou
            + 0.30 * xai_gt_iou
        )
    else:
        stability = masked_pearson_corr(pca_map, pca_aug_map, valid)
        g_lsas_no_stability = (
            0.40 * corr
            + 0.30 * pca_gt_iou
            + 0.30 * xai_gt_iou
        )

    g_score = (
        corr_weight * corr
        + pca_gt_weight * pca_gt_iou
        + xai_gt_weight * xai_gt_iou
        + stability_weight * stability
    )

    return {
        "corr_pca_xai": float(corr),
        "soft_iou_pca_xai": float(pca_xai_soft),
        "iou_pca_gt": float(pca_gt_iou),
        "iou_xai_gt": float(xai_gt_iou),
        "soft_iou_pca_gt": float(pca_gt_soft_iou),
        "soft_iou_xai_gt": float(xai_gt_soft_iou),
        "object_focus_pca": float(pca_focus),
        "object_focus_xai": float(xai_focus),
        "foreground_area": foreground_area(gt, valid),
        "pca_stability": float(stability),
        "g_lsas_no_stability": float(g_lsas_no_stability),
        "g_lsas": float(g_score),
        "top_frac": float(top_frac),
    }
