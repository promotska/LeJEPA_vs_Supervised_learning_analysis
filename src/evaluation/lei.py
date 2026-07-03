from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def layer_sort_key(layer: str):
    numbers = [int(x) for x in re.findall(r"\d+", str(layer))]
    return numbers[-1] if numbers else str(layer)


def compute_layer_emergence_index(
    csv_path: str | Path,
    metric: str = "lsas",
    threshold: float = 0.30,
    output_csv: str | Path | None = None,
    output_yaml: str | Path | None = None,
) -> dict[str, Any]:
    df = pd.read_csv(csv_path)
    if metric not in df.columns:
        raise KeyError(f"Metric {metric!r} not found in {csv_path}. Available: {list(df.columns)}")

    grouped = df.groupby("layer", as_index=False).agg(
        mean=(metric, "mean"),
        std=(metric, "std"),
        n=(metric, "count"),
    )
    grouped["sem"] = grouped["std"] / np.sqrt(grouped["n"].clip(lower=1))
    grouped["ci95"] = 1.96 * grouped["sem"]

    layer_order = sorted(grouped["layer"].tolist(), key=layer_sort_key)
    grouped = grouped.set_index("layer").loc[layer_order].reset_index()
    grouped["layer_index"] = np.arange(len(grouped))
    grouped["normalized_layer_index"] = grouped["layer_index"] / max(len(grouped) - 1, 1)

    crossing = grouped[grouped["mean"] >= float(threshold)]
    if len(crossing) > 0:
        emergence_row = crossing.iloc[0]
        emergence_layer = str(emergence_row["layer"])
        emergence_layer_index = int(emergence_row["layer_index"])
        normalized_emergence_index = float(emergence_row["normalized_layer_index"])
    else:
        emergence_layer = None
        emergence_layer_index = None
        normalized_emergence_index = None

    peak_row = grouped.iloc[int(grouped["mean"].argmax())]
    auc = float(np.trapz(grouped["mean"].values, dx=1.0) / max(len(grouped) - 1, 1))

    result: dict[str, Any] = {
        "csv_path": str(csv_path),
        "metric": metric,
        "threshold": float(threshold),
        "num_layers": int(len(grouped)),
        "emergence_layer": emergence_layer,
        "emergence_layer_index": emergence_layer_index,
        "normalized_emergence_index": normalized_emergence_index,
        "peak_layer": str(peak_row["layer"]),
        "peak_layer_index": int(peak_row["layer_index"]),
        "peak_value": float(peak_row["mean"]),
        "auc": auc,
    }

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([result]).to_csv(output_csv, index=False)
        grouped.to_csv(output_csv.with_name(output_csv.stem + "_layers.csv"), index=False)

    if output_yaml is not None:
        output_yaml = Path(output_yaml)
        output_yaml.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(result)
        payload["layer_summary"] = grouped.to_dict(orient="records")
        with output_yaml.open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False)

    return result
