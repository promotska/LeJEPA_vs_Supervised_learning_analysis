from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.visualization.plot_layer_curves import save_comparison_layer_curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--supervised_csv",
        default="outputs/metrics/stage1_supervised_lsas.csv",
    )
    parser.add_argument(
        "--lejepa_csv",
        default="outputs/metrics/stage1_lejepa_lsas.csv",
    )
    parser.add_argument(
        "--output",
        default="outputs/figures/stage1_supervised_vs_lejepa_lsas_ci.png",
    )
    parser.add_argument(
        "--metric",
        default="lsas",
        choices=["lsas", "corr_pca_xai", "soft_iou_pca_xai"],
    )
    args = parser.parse_args()

    save_comparison_layer_curve(
        supervised_csv_path=args.supervised_csv,
        lejepa_csv_path=args.lejepa_csv,
        output_path=args.output,
        metric=args.metric,
        title=f"Stage 1: supervised vs LeJEPA layer-wise {args.metric}",
        show_ci=True,
    )

    print(f"Saved comparison figure to {args.output}")


if __name__ == "__main__":
    main()