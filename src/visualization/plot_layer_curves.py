from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import ensure_dir


def _summary_by_layer(
    df: pd.DataFrame,
    metric: str = "lsas",
) -> pd.DataFrame:
    grouped = (
        df.groupby("layer", as_index=False)
        .agg(
            mean=(metric, "mean"),
            std=(metric, "std"),
            n=(metric, "count"),
        )
    )

    grouped["sem"] = grouped["std"] / np.sqrt(grouped["n"].clip(lower=1))
    grouped["ci95"] = 1.96 * grouped["sem"]

    return grouped


def save_layer_curve(
    csv_path: str | Path,
    output_path: str | Path,
    title: str,
    show_ci: bool = True,
) -> None:
    """
    Save one model's layer-wise LSAS/PCA-XAI/Soft-IoU curves.

    If show_ci=True, LSAS is plotted with 95% CI.
    Corr and Soft IoU are kept as plain curves to avoid making the plot too crowded.
    """
    csv_path = Path(csv_path)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    df = pd.read_csv(csv_path)

    grouped = df.groupby("layer", as_index=False).agg(
        lsas_mean=("lsas", "mean"),
        lsas_std=("lsas", "std"),
        lsas_n=("lsas", "count"),
        corr_mean=("corr_pca_xai", "mean"),
        soft_iou_mean=("soft_iou_pca_xai", "mean"),
    )

    grouped["lsas_sem"] = grouped["lsas_std"] / np.sqrt(grouped["lsas_n"].clip(lower=1))
    grouped["lsas_ci95"] = 1.96 * grouped["lsas_sem"]

    fig, ax = plt.subplots(figsize=(8, 4))

    x = np.arange(len(grouped))

    ax.plot(x, grouped["lsas_mean"].values, marker="o", label="LSAS")
    if show_ci:
        ax.fill_between(
            x,
            grouped["lsas_mean"].values - grouped["lsas_ci95"].values,
            grouped["lsas_mean"].values + grouped["lsas_ci95"].values,
            alpha=0.2,
            label="LSAS 95% CI",
        )

    ax.plot(x, grouped["corr_mean"].values, marker="o", label="PCA-XAI Corr")
    ax.plot(x, grouped["soft_iou_mean"].values, marker="o", label="Soft IoU")

    ax.set_xticks(x)
    ax.set_xticklabels(grouped["layer"].tolist(), rotation=30, ha="right")
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_comparison_layer_curve(
    supervised_csv_path: str | Path,
    lejepa_csv_path: str | Path,
    output_path: str | Path,
    metric: str = "lsas",
    title: str = "Stage 1: PCA-GradCAM alignment across ResNet-18 layers",
    show_ci: bool = True,
) -> None:
    """
    Save supervised-vs-LeJEPA layer-wise comparison with optional 95% CI.

    This is the main plot you should use in the report.
    """
    supervised_csv_path = Path(supervised_csv_path)
    lejepa_csv_path = Path(lejepa_csv_path)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    supervised_df = pd.read_csv(supervised_csv_path)
    lejepa_df = pd.read_csv(lejepa_csv_path)

    supervised_summary = _summary_by_layer(supervised_df, metric=metric)
    lejepa_summary = _summary_by_layer(lejepa_df, metric=metric)

    layer_order = supervised_summary["layer"].tolist()

    supervised_summary = supervised_summary.set_index("layer").loc[layer_order].reset_index()
    lejepa_summary = lejepa_summary.set_index("layer").loc[layer_order].reset_index()

    x = np.arange(len(layer_order))

    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.plot(
        x,
        supervised_summary["mean"].values,
        marker="o",
        label="supervised",
    )
    if show_ci:
        ax.fill_between(
            x,
            supervised_summary["mean"].values - supervised_summary["ci95"].values,
            supervised_summary["mean"].values + supervised_summary["ci95"].values,
            alpha=0.2,
        )

    ax.plot(
        x,
        lejepa_summary["mean"].values,
        marker="o",
        label="LeJEPA",
    )
    if show_ci:
        ax.fill_between(
            x,
            lejepa_summary["mean"].values - lejepa_summary["ci95"].values,
            lejepa_summary["mean"].values + lejepa_summary["ci95"].values,
            alpha=0.2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(layer_order, rotation=30, ha="right")
    ax.set_ylabel(f"mean {metric}")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)