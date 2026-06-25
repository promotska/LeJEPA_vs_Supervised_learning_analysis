from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentContext:
    name: str
    root: Path
    dir: Path
    run_id: str
    run_kind: str
    mode: str | None
    manifest_path: Path
    config_snapshot_path: Path
    resolved_config_path: Path
    checkpoints_dir: Path
    metrics_dir: Path
    figures_dir: Path
    logs_dir: Path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_name(value: str) -> str:
    safe = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_", "."}:
            safe.append(ch)
        else:
            safe.append("-")
    return "".join(safe).strip("-")


def _run_command(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _git_info() -> dict[str, Any]:
    return {
        "branch": _run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": _run_command(["git", "rev-parse", "HEAD"]),
        "commit_short": _run_command(["git", "rev-parse", "--short", "HEAD"]),
        "is_dirty": _run_command(["git", "status", "--porcelain"]) not in {None, ""},
        "remote": _run_command(["git", "remote", "get-url", "origin"]),
    }


def _environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cwd": str(Path.cwd()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        "slurm_submit_dir": os.environ.get("SLURM_SUBMIT_DIR"),
    }

    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as exc:
        info["torch_error"] = repr(exc)

    return info


def _ensure_dirs(exp_dir: Path) -> dict[str, Path]:
    dirs = {
        "configs": exp_dir / "configs",
        "checkpoints": exp_dir / "checkpoints",
        "metrics": exp_dir / "metrics",
        "figures": exp_dir / "figures",
        "logs": exp_dir / "logs",
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    return dirs


def _relocate_file_path(path_value: str | Path, target_dir: Path) -> str:
    """
    Move an output file path into target_dir while preserving only the filename.

    Example:
      outputs/checkpoints/model_best.pt
      -> experiments/exp-1/checkpoints/model_best.pt
    """
    old = Path(path_value)
    return str(target_dir / old.name)


def _relocate_dir_path(path_value: str | Path, target_dir: Path, default_name: str) -> str:
    """
    Move an output directory path into target_dir while preserving the last folder name.

    Example:
      outputs/figures/supervised
      -> experiments/exp-1/figures/supervised
    """
    old = Path(path_value)
    name = old.name if old.name else default_name
    return str(target_dir / name)


def _relocate_config_paths(
    cfg: dict[str, Any],
    exp: ExperimentContext,
    relocate_checkpoint_path: bool,
) -> None:
    """
    Mutates cfg so all generated artifacts go inside experiments/<name>/.
    """

    cfg.setdefault("project", {})
    cfg["project"]["output_dir"] = str(exp.dir)

    # Training checkpoint outputs.
    if "checkpoints" in cfg and isinstance(cfg["checkpoints"], dict):
        for key, value in list(cfg["checkpoints"].items()):
            if value is not None:
                cfg["checkpoints"][key] = _relocate_file_path(value, exp.checkpoints_dir)

    # Evaluation outputs.
    if "evaluation" in cfg and isinstance(cfg["evaluation"], dict):
        evaluation = cfg["evaluation"]

        if "output_csv" in evaluation and evaluation["output_csv"] is not None:
            evaluation["output_csv"] = _relocate_file_path(evaluation["output_csv"], exp.metrics_dir)

        if "figure_dir" in evaluation and evaluation["figure_dir"] is not None:
            evaluation["figure_dir"] = _relocate_dir_path(
                evaluation["figure_dir"],
                exp.figures_dir,
                default_name=exp.mode or exp.run_kind,
            )

        # This should point to the checkpoint produced inside the same experiment.
        # Keep this true for normal experiment runs.
        if relocate_checkpoint_path and "checkpoint_path" in evaluation and evaluation["checkpoint_path"] is not None:
            evaluation["checkpoint_path"] = _relocate_file_path(
                evaluation["checkpoint_path"],
                exp.checkpoints_dir,
            )


def _scan_artifacts(exp_dir: Path) -> dict[str, list[dict[str, Any]]]:
    artifacts: dict[str, list[dict[str, Any]]] = {
        "configs": [],
        "checkpoints": [],
        "metrics": [],
        "figures": [],
        "logs": [],
    }

    groups = {
        "configs": exp_dir / "configs",
        "checkpoints": exp_dir / "checkpoints",
        "metrics": exp_dir / "metrics",
        "figures": exp_dir / "figures",
        "logs": exp_dir / "logs",
    }

    for group, root in groups.items():
        if not root.exists():
            continue

        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue

            artifacts[group].append(
                {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "modified_utc": datetime.fromtimestamp(
                        path.stat().st_mtime,
                        tz=timezone.utc,
                    ).replace(microsecond=0).isoformat(),
                }
            )

    return artifacts


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data if isinstance(data, dict) else {}


def _dump_manifest(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )


def prepare_experiment(
    cfg: dict[str, Any],
    config_path: str | Path,
    run_kind: str,
    mode: str | None = None,
) -> ExperimentContext:
    """
    Prepare one experiment directory and mutate cfg paths so outputs go there.

    Experiment name priority:
      1. EXPERIMENT_NAME environment variable
      2. cfg["experiment"]["name"]
      3. generated name from project.name + timestamp

    Add this to config if you want stable output:
      experiment:
        root: experiments
        name: experiment-3
        relocate_checkpoint_path: true
    """

    project_name = str(cfg.get("project", {}).get("name", "experiment"))
    exp_cfg = cfg.get("experiment", {}) or {}

    root = Path(os.environ.get("EXPERIMENT_ROOT", exp_cfg.get("root", "experiments")))

    env_name = os.environ.get("EXPERIMENT_NAME")
    cfg_name = exp_cfg.get("name")

    if env_name:
        name = env_name
    elif cfg_name:
        name = str(cfg_name)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{project_name}_{timestamp}"

    name = _safe_name(name)
    exp_dir = root / name

    dirs = _ensure_dirs(exp_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = _safe_name(f"{timestamp}_{run_kind}_{mode or 'default'}_{os.getpid()}")

    config_path = Path(config_path)
    config_snapshot_path = dirs["configs"] / f"original_{config_path.name}"
    resolved_config_path = dirs["configs"] / f"resolved_{run_id}.yaml"

    # Copy original config only once per experiment/config name.
    if config_path.exists() and not config_snapshot_path.exists():
        shutil.copy2(config_path, config_snapshot_path)

    exp = ExperimentContext(
        name=name,
        root=root,
        dir=exp_dir,
        run_id=run_id,
        run_kind=run_kind,
        mode=mode,
        manifest_path=exp_dir / "manifest.yaml",
        config_snapshot_path=config_snapshot_path,
        resolved_config_path=resolved_config_path,
        checkpoints_dir=dirs["checkpoints"],
        metrics_dir=dirs["metrics"],
        figures_dir=dirs["figures"],
        logs_dir=dirs["logs"],
    )

    relocate_checkpoint_path = bool(exp_cfg.get("relocate_checkpoint_path", True))
    _relocate_config_paths(cfg, exp, relocate_checkpoint_path=relocate_checkpoint_path)

    cfg["_experiment"] = {
        "name": exp.name,
        "dir": str(exp.dir),
        "run_id": exp.run_id,
        "run_kind": exp.run_kind,
        "mode": exp.mode,
        "manifest_path": str(exp.manifest_path),
        "config_snapshot_path": str(exp.config_snapshot_path),
        "resolved_config_path": str(exp.resolved_config_path),
        "checkpoints_dir": str(exp.checkpoints_dir),
        "metrics_dir": str(exp.metrics_dir),
        "figures_dir": str(exp.figures_dir),
        "logs_dir": str(exp.logs_dir),
    }

    with exp.resolved_config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    update_manifest(
        exp=exp,
        cfg=cfg,
        status="started",
        extra=None,
    )

    return exp


def update_manifest(
    exp: ExperimentContext,
    cfg: dict[str, Any],
    status: str,
    extra: dict[str, Any] | None = None,
) -> None:
    manifest = _load_manifest(exp.manifest_path)

    if not manifest:
        manifest = {
            "experiment": {
                "name": exp.name,
                "root": str(exp.root),
                "dir": str(exp.dir),
                "created_at_utc": _utc_now_iso(),
            },
            "git": _git_info(),
            "runs": [],
            "latest_status": None,
            "artifacts": {},
        }

    manifest["experiment"]["updated_at_utc"] = _utc_now_iso()
    manifest["latest_status"] = status
    manifest["git"] = _git_info()

    run_record = {
        "run_id": exp.run_id,
        "run_kind": exp.run_kind,
        "mode": exp.mode,
        "status": status,
        "updated_at_utc": _utc_now_iso(),
        "command": " ".join(sys.argv),
        "config_snapshot_path": str(exp.config_snapshot_path),
        "resolved_config_path": str(exp.resolved_config_path),
        "environment": _environment_info(),
        "project": cfg.get("project"),
        "data": cfg.get("data"),
        "model": cfg.get("model"),
        "training": cfg.get("training"),
        "checkpoints": cfg.get("checkpoints"),
        "evaluation": cfg.get("evaluation"),
    }

    if extra:
        run_record["extra"] = extra

    runs = manifest.setdefault("runs", [])
    replaced = False

    for idx, existing in enumerate(runs):
        if existing.get("run_id") == exp.run_id:
            runs[idx] = run_record
            replaced = True
            break

    if not replaced:
        runs.append(run_record)

    manifest["artifacts"] = _scan_artifacts(exp.dir)

    _dump_manifest(exp.manifest_path, manifest)