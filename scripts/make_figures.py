from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervised_csv", default="outputs/metrics/stage1_supervised_lsas.csv")
    parser.add_argument("--lejepa_csv", default="outputs/metrics/stage1_lejepa_lsas.csv")
    parser.add_argument("--output", default="outputs/figures/stage1_supervised_vs_lejepa_lsas.png")
    args = parser.parse_args()

    frames = []
    for mode, path in [("supervised", args.supervised_csv), ("lejepa", args.lejepa_csv)]:
        df = pd.read_csv(path)
        df["mode"] = mode
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    grouped = all_df.groupby(["mode", "layer"], as_index=False).agg(lsas_mean=("lsas", "mean"))

    layer_order = list(dict.fromkeys(all_df["layer"].tolist()))
    fig, ax = plt.subplots(figsize=(8, 4))
    for mode in ["supervised", "lejepa"]:
        sub = grouped[grouped["mode"] == mode].set_index("layer").loc[layer_order].reset_index()
        ax.plot(range(len(layer_order)), sub["lsas_mean"], marker="o", label=mode)
    ax.set_xticks(range(len(layer_order)))
    ax.set_xticklabels(layer_order, rotation=30, ha="right")
    ax.set_ylabel("mean LSAS")
    ax.set_title("Stage 1: PCA-GradCAM alignment across ResNet-18 layers")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output = Path(args.output)
    ensure_dir(output.parent)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Saved comparison figure to {output}")


if __name__ == "__main__":
    main()
