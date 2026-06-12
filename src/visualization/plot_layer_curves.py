from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils import ensure_dir


def save_layer_curve(csv_path: str | Path, output_path: str | Path, title: str) -> None:
    csv_path = Path(csv_path)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    df = pd.read_csv(csv_path)
    grouped = df.groupby("layer", as_index=False).agg(
        lsas_mean=("lsas", "mean"),
        lsas_std=("lsas", "std"),
        corr_mean=("corr_pca_xai", "mean"),
        soft_iou_mean=("soft_iou_pca_xai", "mean"),
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(grouped))
    ax.plot(x, grouped["lsas_mean"].values, marker="o", label="LSAS")
    ax.plot(x, grouped["corr_mean"].values, marker="o", label="PCA-XAI Corr")
    ax.plot(x, grouped["soft_iou_mean"].values, marker="o", label="Soft IoU")
    ax.set_xticks(list(x))
    ax.set_xticklabels(grouped["layer"].tolist(), rotation=30, ha="right")
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
