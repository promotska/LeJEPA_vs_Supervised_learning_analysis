from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.lei import layer_sort_key
from src.utils import ensure_dir


def save_grounded_layer_curve(
    csv_path: str | Path,
    output_path: str | Path,
    metric: str = "g_lsas",
    title: str = "Grounded LSAS across layers",
    show_ci: bool = True,
) -> None:
    csv_path = Path(csv_path)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    df = pd.read_csv(csv_path)
    required = [metric, "corr_pca_xai", "iou_pca_gt", "iou_xai_gt", "pca_stability"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for grounded curve: {missing}")

    grouped = df.groupby("layer", as_index=False).agg(
        metric_mean=(metric, "mean"),
        metric_std=(metric, "std"),
        metric_n=(metric, "count"),
        corr_mean=("corr_pca_xai", "mean"),
        pca_gt_mean=("iou_pca_gt", "mean"),
        xai_gt_mean=("iou_xai_gt", "mean"),
        stability_mean=("pca_stability", "mean"),
    )
    grouped = grouped.set_index("layer").loc[sorted(grouped["layer"].tolist(), key=layer_sort_key)].reset_index()
    grouped["metric_sem"] = grouped["metric_std"] / np.sqrt(grouped["metric_n"].clip(lower=1))
    grouped["metric_ci95"] = 1.96 * grouped["metric_sem"]

    x = np.arange(len(grouped))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(x, grouped["metric_mean"].values, marker="o", label=metric)
    if show_ci:
        ax.fill_between(
            x,
            grouped["metric_mean"].values - grouped["metric_ci95"].values,
            grouped["metric_mean"].values + grouped["metric_ci95"].values,
            alpha=0.2,
            label=f"{metric} 95% CI",
        )
    ax.plot(x, grouped["corr_mean"].values, marker="o", label="PCA-XAI Corr")
    ax.plot(x, grouped["pca_gt_mean"].values, marker="o", label="PCA-GT IoU")
    ax.plot(x, grouped["xai_gt_mean"].values, marker="o", label="XAI-GT IoU")
    ax.plot(x, grouped["stability_mean"].values, marker="o", label="PCA Stability")

    ax.set_xticks(x)
    ax.set_xticklabels(grouped["layer"].tolist(), rotation=30, ha="right")
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
