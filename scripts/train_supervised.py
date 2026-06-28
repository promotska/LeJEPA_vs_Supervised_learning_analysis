from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment import prepare_experiment, update_manifest
from src.training.train_supervised import train_supervised
from src.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/cifar10_resnet18_supervised.yaml",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    exp = prepare_experiment(
        cfg=cfg,
        config_path=args.config,
        run_kind="train_supervised",
        mode="supervised",
    )

    try:
        # prepare_experiment() rewrites checkpoint paths into experiments/<name>/checkpoints.
        # Therefore we must train from the resolved config, not the original config.
        result = train_supervised(str(exp.resolved_config_path))

        update_manifest(
            exp=exp,
            cfg=cfg,
            status="completed",
            extra={
                "note": "supervised training completed",
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