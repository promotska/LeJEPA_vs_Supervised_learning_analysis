from __future__ import annotations

import argparse
import traceback
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.train_lejepa import train_lejepa
from src.experiment import prepare_experiment, update_manifest
from src.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/cifar10_resnet18_lejepa.yaml",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    exp = prepare_experiment(
        cfg=cfg,
        config_path=args.config,
        run_kind="train_lejepa",
        mode="lejepa",
    )

    try:
        # IMPORTANT:
        # prepare_experiment() mutates cfg paths so checkpoints go into:
        # experiments/<experiment-name>/checkpoints/
        result = train_lejepa(args.config)

        update_manifest(
            exp=exp,
            cfg=cfg,
            status="completed",
            extra={
                "note": "lejepa training completed",
                "result": result,
            },
        )

    except Exception as exc:
        update_manifest(
            exp=exp,
            cfg=cfg,
            status="failed",
            extra={
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()