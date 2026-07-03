from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.lei import compute_layer_emergence_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Layer Emergence Index from LSAS/G-LSAS CSV.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--metric", default="lsas")
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-yaml", default=None)
    args = parser.parse_args()

    output_yaml = args.output_yaml or str(Path(args.output_csv).with_suffix(".yaml"))
    result = compute_layer_emergence_index(
        csv_path=args.csv,
        metric=args.metric,
        threshold=args.threshold,
        output_csv=args.output_csv,
        output_yaml=output_yaml,
    )
    print("Layer Emergence Index computed")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
