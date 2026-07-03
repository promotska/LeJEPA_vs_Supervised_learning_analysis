from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def infer_layer_order(layers: list[str]) -> list[str]:
    def key(layer: str):
        # Handles backbone.layer1, backbone.layer2, blocks.1, blocks.2, etc.
        parts = layer.replace("_", ".").split(".")
        nums = [int(p) for p in parts if p.isdigit()]
        return nums[-1] if nums else layer

    return sorted(layers, key=key)


def compute_lei(csv_path: str | Path, threshold: float) -> dict:
    df = pd.read_csv(csv_path)

    grouped = (
        df.groupby("layer", as_index=False)
        .agg(
            mean_lsas=("lsas", "mean"),
            std_lsas=("lsas", "std"),
            n=("lsas", "count"),
            mean_corr=("corr_pca_xai", "mean"),
            mean_soft_iou=("soft_iou_pca_xai", "mean"),
        )
    )

    layer_order = infer_layer_order(grouped["layer"].tolist())
    grouped = grouped.set_index("layer").loc[layer_order].reset_index()
    grouped["layer_index"] = np.arange(len(grouped))

    crossing = grouped[grouped["mean_lsas"] >= threshold]

    if len(crossing) > 0:
        emergence_row = crossing.iloc[0]
        emergence_layer = str(emergence_row["layer"])
        emergence_index = int(emergence_row["layer_index"])
        normalized_emergence_index = (
            emergence_index / max(len(grouped) - 1, 1)
        )
    else:
        emergence_layer = None
        emergence_index = None
        normalized_emergence_index = None

    peak_row = grouped.iloc[grouped["mean_lsas"].argmax()]
    auc_lsas = float(np.trapz(grouped["mean_lsas"].values, dx=1.0) / max(len(grouped) - 1, 1))

    return {
        "csv_path": str(csv_path),
        "threshold": float(threshold),
        "num_layers": int(len(grouped)),
        "emergence_layer": emergence_layer,
        "emergence_layer_index": emergence_index,
        "normalized_emergence_index": normalized_emergence_index,
        "peak_layer": str(peak_row["layer"]),
        "peak_lsas": float(peak_row["mean_lsas"]),
        "auc_lsas": auc_lsas,
        "layer_summary": grouped.to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = compute_lei(args.csv, args.threshold)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([{
        k: v for k, v in result.items()
        if k != "layer_summary"
    }]).to_csv(output_path, index=False)

    summary_path = output_path.with_name(output_path.stem + "_layers.csv")
    pd.DataFrame(result["layer_summary"]).to_csv(summary_path, index=False)

    print("Layer Emergence Index computed.")
    print(f"CSV: {args.csv}")
    print(f"Threshold: {args.threshold}")
    print(f"Emergence layer: {result['emergence_layer']}")
    print(f"Peak layer: {result['peak_layer']}")
    print(f"Peak LSAS: {result['peak_lsas']:.4f}")
    print(f"AUC LSAS: {result['auc_lsas']:.4f}")
    print(f"Saved: {output_path}")
    print(f"Saved layer summary: {summary_path}")


if __name__ == "__main__":
    main()