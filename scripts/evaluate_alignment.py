from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.layer_analysis import evaluate_layer_alignment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["supervised", "lejepa"], required=True)
    args = parser.parse_args()
    evaluate_layer_alignment(args.config, mode=args.mode)


if __name__ == "__main__":
    main()
