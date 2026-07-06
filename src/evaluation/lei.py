from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def layer_sort_key(layer: str):
    parts = str(layer).replace("_", ".").split(".")
    nums = [int(p) for p in parts if p.isdigit()]
    if nums:
        return (0, nums[-1], str(layer))
    return (1, 0, str(layer))


def _infer_layer_order(layers: list[str]) -> list[str]:
    def key(layer: str):
        parts = layer.replace("_", ".").split(".")
        nums = [int(p) for p in parts if p.isdigit()]
        if nums:
            return (0, nums[-1], layer)
        return (1, 0, layer)

    return sorted(layers, key=key)


def _normalized_auc(values: np.ndarray) -> float:
    """
    Normalized trapezoidal area under the layer-wise metric curve.

    Avoids np.trapz / np.trapezoid compatibility issues across NumPy versions.
    For n layers, the raw trapezoidal integral is divided by n - 1, so the
    result remains on approximately the same scale as the original metric.
    """
    values = np.asarray(values, dtype=float)

    if values.size < 2:
        return 0.0

    raw_auc = ((values[:-1] + values[1:]) * 0.5).sum()
    return float(raw_auc / (values.size - 1))


def compute_layer_emergence_index(
    csv_path: str | Path,
    metric: str = "lsas",
    threshold: float = 0.30,
    relative_threshold: float = 0.90,
    output_csv: str | Path | None = None,
    output_yaml: str | Path | None = None,
) -> dict[str, Any]:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if "layer" not in df.columns:
        raise ValueError(f"{csv_path} does not contain a 'layer' column.")

    if metric not in df.columns:
        raise ValueError(f"{csv_path} does not contain metric column {metric!r}.")
    
    df = df.copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=["layer", metric])

    if df.empty:
        raise ValueError(
            f"{csv_path} contains no valid non-NaN values for metric {metric!r}."
        )

    grouped = (
        df.groupby("layer", as_index=False)
        .agg(
            mean=(metric, "mean"),
            std=(metric, "std"),
            count=(metric, "count"),
        )
    )

    layer_order = _infer_layer_order(grouped["layer"].astype(str).tolist())
    grouped = grouped.set_index("layer").loc[layer_order].reset_index()
    grouped["layer_index"] = np.arange(len(grouped), dtype=int)

    values = grouped["mean"].to_numpy(dtype=float)
    peak_idx = int(np.nanargmax(values))
    peak_layer = str(grouped.iloc[peak_idx]["layer"])
    peak_value = float(values[peak_idx])

    # Absolute LEI: first layer where metric >= fixed threshold.
    abs_crossing = grouped[grouped["mean"] >= threshold]
    if len(abs_crossing) > 0:
        abs_row = abs_crossing.iloc[0]
        emergence_layer = str(abs_row["layer"])
        emergence_layer_index = int(abs_row["layer_index"])
        normalized_emergence_index = (
            emergence_layer_index / max(len(grouped) - 1, 1)
        )
    else:
        emergence_layer = None
        emergence_layer_index = None
        normalized_emergence_index = None

    # Relative LEI: first layer where metric reaches relative_threshold * peak.
    rel_tau = float(relative_threshold * peak_value)
    rel_crossing = grouped[grouped["mean"] >= rel_tau]
    if len(rel_crossing) > 0:
        rel_row = rel_crossing.iloc[0]
        relative_emergence_layer = str(rel_row["layer"])
        relative_emergence_layer_index = int(rel_row["layer_index"])
        normalized_relative_emergence_index = (
            relative_emergence_layer_index / max(len(grouped) - 1, 1)
        )
    else:
        relative_emergence_layer = None
        relative_emergence_layer_index = None
        normalized_relative_emergence_index = None

    auc = _normalized_auc(values)

    result: dict[str, Any] = {
        "csv_path": str(csv_path),
        "metric": metric,
        "threshold": float(threshold),
        "relative_threshold": float(relative_threshold),
        "num_layers": int(len(grouped)),
        "peak_layer": peak_layer,
        "peak_value": peak_value,
        "auc": auc,
        "emergence_layer": emergence_layer,
        "emergence_layer_index": emergence_layer_index,
        "normalized_emergence_index": normalized_emergence_index,
        "relative_emergence_threshold_value": rel_tau,
        "relative_emergence_layer": relative_emergence_layer,
        "relative_emergence_layer_index": relative_emergence_layer_index,
        "normalized_relative_emergence_index": normalized_relative_emergence_index,
    }

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([result]).to_csv(output_csv, index=False)

        layer_csv = output_csv.with_name(output_csv.stem + "_layers.csv")
        grouped.to_csv(layer_csv, index=False)

    if output_yaml is not None:
        output_yaml = Path(output_yaml)
        output_yaml.parent.mkdir(parents=True, exist_ok=True)
        with output_yaml.open("w", encoding="utf-8") as f:
            yaml.safe_dump(result, f, sort_keys=False)

    return result
