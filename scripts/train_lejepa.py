from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.train_lejepa import train_lejepa


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cifar10_resnet18_lejepa.yaml")
    args = parser.parse_args()
    train_lejepa(args.config)


if __name__ == "__main__":
    main()
