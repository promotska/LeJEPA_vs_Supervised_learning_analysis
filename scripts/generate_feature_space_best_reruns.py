from __future__ import annotations

import argparse
import re
from pathlib import Path


CONFIG_MAP = {
    "experiment-c10-r18-lejepa": "configs/base/cifar10_resnet18_lejepa.yaml",
    "experiment-c10-r18-lejepa-light-config": "configs/base/cifar10_resnet18_lejepa.yaml",
    "experiment-c10-r18-lejepa-light-v3": "configs/base/cifar10_resnet18_lejepa.yaml",
    "experiment-c10-r18-lejepa-light-v3-sig-005": "configs/base/cifar10_resnet18_lejepa.yaml",
    "experiment-c10-r18-lejepa-v3-sigreg-003": "configs/cifar10_resnet18_lejepa_v3_sigreg_003.yaml",
    "experiment-c10-r18-lejepa-v4-local-crops": "configs/cifar10_resnet18_lejepa_v4_local_crops.yaml",
    "experiment-c10-r18-lejepa-v4-proj-512": "configs/cifar10_resnet18_lejepa_v4_proj_512.yaml",
    "experiment-c10-r18-lejepa-v5-sigreg-002": "configs/cifar10_resnet18_lejepa_v5_sigreg_002.yaml",

    "experiment-c10-vit-lejepa": "configs/base/cifar10_vit_lejepa.yaml",
    "experiment-c10-vit-lejepa-light-config": "configs/base/cifar10_vit_lejepa.yaml",
    "experiment-c10-vit-lejepa-light-v3": "configs/base/cifar10_vit_lejepa.yaml",
    "experiment-c10-vit-lejepa-v4-short-lr1e4-sig005": "configs/cifar10_vit_lejepa_v4_short_lr1e4_sig005.yaml",
    "experiment-c10-vit-lejepa-v4-short-lr2e4": "configs/cifar10_vit_lejepa_v4_short_lr2e4.yaml",
    "experiment-c10-vit-lejepa-v4-short-lr2e4-sig005": "configs/cifar10_vit_lejepa_v4_short_lr2e4_sig005.yaml",
    "experiment-c10-vit-lejepa-v4-short-lr2e4-sig005-proj512": "configs/cifar10_vit_lejepa_v4_short_lr2e4_sig005_proj512.yaml",
    "experiment-c10-vit-lejepa-v5-short-lr15e4-sig005": "configs/cifar10_vit_lejepa_v5_short_lr15e4_sig005.yaml",

    "experiment-imagenet100-r18-lejepa": "configs/imagenet100_resnet18_lejepa_glsas.yaml",
    "experiment-imagenet100-vit-lejepa": "configs/imagenet100_vit_lejepa_glsas.yaml",
    "experiment-imagenets50-r18-lejepa": "configs/imagenets50_resnet18_lejepa_glsas.yaml",
    "experiment-imagenets50-vit-lejepa": "configs/imagenets50_vit_lejepa_glsas.yaml",
}


def infer_target(exp_name: str) -> str:
    return "vit_lejepa" if "vit" in exp_name.lower() else "lejepa"


def checkpoint_exists(exp_dir: Path, target: str) -> bool:
    if target == "vit_lejepa":
        return (exp_dir / "checkpoints" / "vit_lejepa_backbone_best.pt").exists()
    return (exp_dir / "checkpoints" / "resnet18_lejepa_backbone_best.pt").exists()


def has_last_feature_space_log(exp_dir: Path) -> bool:
    logs = exp_dir / "logs"
    if not logs.exists():
        return False

    pattern = re.compile(r"backbone_last\.pt")
    for p in list(logs.glob("*feature_spaces*.out")) + list(logs.glob("*feature_spaces*.err")):
        try:
            if pattern.search(p.read_text(errors="ignore")):
                return True
        except Exception:
            pass
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-root", default="experiments")
    parser.add_argument("--output", default="rerun_feature_spaces_best.sh")
    parser.add_argument(
        "--only-if-last-log",
        action="store_true",
        help="Only include experiments whose feature-space logs mention backbone_last.pt.",
    )
    args = parser.parse_args()

    root = Path(args.experiments_root)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Auto-generated feature-space reruns with FEATURE_CHECKPOINT=best.",
        "# Review before running.",
        "",
    ]

    skipped = []

    for exp_name, config in sorted(CONFIG_MAP.items()):
        exp_dir = root / exp_name
        target = infer_target(exp_name)
        config_path = Path(config)

        if not exp_dir.exists():
            skipped.append((exp_name, "missing experiment dir"))
            continue

        if not config_path.exists():
            skipped.append((exp_name, f"missing config: {config}"))
            continue

        if not checkpoint_exists(exp_dir, target):
            skipped.append((exp_name, "missing backbone_best checkpoint"))
            continue

        if args.only_if_last_log and not has_last_feature_space_log(exp_dir):
            skipped.append((exp_name, "no feature-space log with backbone_last.pt"))
            continue

        lines.append(
            f'FEATURE_CHECKPOINT=best bash jobs/submit_experiment.sh '
            f'"{exp_name}" feature_spaces "{target}" "{config}"'
        )

    out = Path(args.output)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out.chmod(0o755)

    print(f"Wrote: {out}")
    print()
    print("Skipped:")
    for exp, reason in skipped:
        print(f"- {exp}: {reason}")


if __name__ == "__main__":
    main()
