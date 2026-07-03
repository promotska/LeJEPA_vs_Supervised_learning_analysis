from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.lei import compute_layer_emergence_index


def _iter_candidate_csvs(experiment_dir: Path, patterns: list[str]) -> Iterable[Path]:
    metrics_dir = experiment_dir / "metrics"
    if not metrics_dir.exists():
        raise FileNotFoundError(f"metrics directory does not exist: {metrics_dir}")
    seen: set[Path] = set()
    for pattern in patterns:
        for path in metrics_dir.rglob(pattern):
            if path in seen:
                continue
            seen.add(path)
            yield path


def _has_metric_column(path: Path, metric: str) -> bool:
    try:
        cols = pd.read_csv(path, nrows=1).columns
    except Exception:
        return False
    return "layer" in cols and metric in cols


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute LSAS/G-LSAS Layer Emergence Index for all matching CSVs in an experiment."
    )
    parser.add_argument("--experiment-dir", required=True, help="Path such as experiments/experiment-c10-r18-sup")
    parser.add_argument("--metric", default="lsas", choices=["lsas", "g_lsas", "g_lsas_no_stability"])
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument(
        "--patterns",
        nargs="*",
        default=["*.csv"],
        help="CSV glob patterns under metrics/, default: *.csv",
    )
    parser.add_argument(
        "--suffix",
        default=None,
        help="Output suffix. Default: _<metric>_lei",
    )
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    suffix = args.suffix or f"_{args.metric}_lei"

    written = []
    for csv_path in _iter_candidate_csvs(experiment_dir, args.patterns):
        # Avoid recursively reading already-created LEI summaries.
        if "lei" in csv_path.stem.lower() or csv_path.name.endswith("_layers.csv"):
            continue
        if not _has_metric_column(csv_path, args.metric):
            continue

        out_csv = csv_path.with_name(csv_path.stem + suffix + ".csv")
        out_yaml = csv_path.with_name(csv_path.stem + suffix + ".yaml")
        result = compute_layer_emergence_index(
            csv_path=csv_path,
            metric=args.metric,
            threshold=args.threshold,
            output_csv=out_csv,
            output_yaml=out_yaml,
        )
        written.append((csv_path, out_csv, out_yaml, result))

    if not written:
        raise RuntimeError(
            f"No CSVs under {experiment_dir / 'metrics'} contained columns: layer and {args.metric!r}."
        )

    print("Computed Layer Emergence Index files:")
    for src, out_csv, out_yaml, result in written:
        print(f"- source: {src}")
        print(f"  csv:    {out_csv}")
        print(f"  yaml:   {out_yaml}")
        print(f"  peak:   {result['peak_layer']}={result['peak_value']:.4f}")
        print(f"  LEI:    {result['emergence_layer']}")


if __name__ == "__main__":
    main()
