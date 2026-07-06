from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


CHECKPOINT_PATTERNS = [
    re.compile(r"(?:checkpoint|Checkpoint|Using backbone checkpoint|Loaded checkpoint|Loading checkpoint)[^:\n]*:\s*(?P<path>\S+\.pt)"),
    re.compile(r"(?P<path>\S+(?:best|last|epoch_\d+)\S*\.pt)"),
]


def classify_checkpoint(path: str) -> str:
    name = Path(path).name.lower()

    if "best" in name:
        return "BEST"
    if "last" in name:
        return "LAST"
    if "epoch_" in name:
        return "EPOCH"
    return "UNKNOWN"


def infer_kind(log_path: Path, line: str) -> str:
    name = log_path.name.lower()
    line_l = line.lower()

    if "feature_spaces" in name or "feature_spaces" in line_l or "using backbone checkpoint" in line_l:
        return "feature_spaces"
    if "grounded" in name:
        return "grounded"
    if "eval" in name:
        return "lsas_eval"
    if "repr" in name:
        return "representation"
    if "train" in name:
        return "train"
    return "unknown"


def infer_target(experiment_name: str) -> str:
    e = experiment_name.lower()

    if "vit" in e and "lejepa" in e:
        return "vit_lejepa"
    if "vit" in e and ("sup" in e or "supervised" in e):
        return "vit_supervised"
    if "lejepa" in e:
        return "lejepa"
    return "supervised"


def audit_experiment(exp_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    logs_dir = exp_dir / "logs"

    if not logs_dir.exists():
        return rows

    for log_path in sorted(list(logs_dir.glob("*.out")) + list(logs_dir.glob("*.err"))):
        try:
            text = log_path.read_text(errors="ignore")
        except Exception:
            continue

        for line_no, line in enumerate(text.splitlines(), start=1):
            if ".pt" not in line:
                continue

            for pat in CHECKPOINT_PATTERNS:
                m = pat.search(line)
                if not m:
                    continue

                ckpt = m.group("path").strip()
                status = classify_checkpoint(ckpt)
                kind = infer_kind(log_path, line)

                if kind == "feature_spaces" and status == "LAST":
                    action = "RERUN_FEATURE_SPACES_WITH_BEST"
                elif status == "LAST":
                    action = "CHECK_AND_PROBABLY_RERUN"
                elif status in {"BEST", "EPOCH"}:
                    action = "OK"
                else:
                    action = "CHECK"

                rows.append(
                    {
                        "experiment": exp_dir.name,
                        "log_file": str(log_path),
                        "line": str(line_no),
                        "kind": kind,
                        "checkpoint_status": status,
                        "checkpoint": ckpt,
                        "action": action,
                    }
                )
                break

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-root", default="experiments")
    parser.add_argument("--output-csv", default="experiments/checkpoint_usage_audit.csv")
    parser.add_argument("--write-rerun-script", default=None)
    args = parser.parse_args()

    experiments_root = Path(args.experiments_root)
    all_rows: list[dict[str, str]] = []

    for exp_dir in sorted(experiments_root.iterdir()):
        if exp_dir.is_dir():
            all_rows.extend(audit_experiment(exp_dir))

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "experiment",
        "kind",
        "checkpoint_status",
        "action",
        "checkpoint",
        "log_file",
        "line",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"Wrote audit CSV: {output_csv}")

    bad = [r for r in all_rows if r["action"] != "OK"]
    print(f"Total checkpoint mentions: {len(all_rows)}")
    print(f"Needs attention: {len(bad)}")

    for row in bad[:50]:
        print(
            f"[{row['action']}] {row['experiment']} | {row['kind']} | "
            f"{row['checkpoint_status']} | {row['checkpoint']}"
        )

    if args.write_rerun_script:
        rerun_path = Path(args.write_rerun_script)
        seen: set[tuple[str, str]] = set()
        lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            "# Auto-generated rerun commands for LeJEPA feature-space jobs that used backbone_last.pt.",
            "# Review before running.",
            "",
        ]

        for row in all_rows:
            if row["action"] != "RERUN_FEATURE_SPACES_WITH_BEST":
                continue

            exp = row["experiment"]
            target = infer_target(exp)

            # Only feature-space reruns are safe to infer automatically.
            key = (exp, target)
            if key in seen:
                continue
            seen.add(key)

            lines.append(
                f'FEATURE_CHECKPOINT=best bash jobs/submit_experiment.sh "{exp}" feature_spaces "{target}"'
            )

        rerun_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        rerun_path.chmod(0o755)
        print(f"Wrote rerun script: {rerun_path}")


if __name__ == "__main__":
    main()
