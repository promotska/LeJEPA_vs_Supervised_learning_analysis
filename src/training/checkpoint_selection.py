from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from src.data.dataloaders import build_loaders
from src.training.checkpointing import load_checkpoint, save_checkpoint
from src.utils import ensure_dir


def _as_list_of_ints(value: Any, default: list[int]) -> list[int]:
    if value is None:
        return default
    if isinstance(value, list):
        return [int(v) for v in value]
    if isinstance(value, str):
        return [int(v.strip()) for v in value.split(",") if v.strip()]
    return [int(value)]


def _unique_existing(paths: list[str | Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()

    for path in paths:
        p = Path(path)
        key = str(p)

        if key in seen or not p.exists():
            continue

        seen.add(key)
        out.append(p)

    return out


@torch.no_grad()
def _extract_features(
    backbone: torch.nn.Module,
    loader,
    device: torch.device,
    max_samples: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    backbone.eval()

    features = []
    labels_all = []
    total = 0

    for images, labels in tqdm(loader, desc="selection extract", leave=False):
        if max_samples is not None and total >= max_samples:
            break

        if max_samples is not None and total + images.shape[0] > max_samples:
            remaining = max_samples - total
            images = images[:remaining]
            labels = labels[:remaining]

        images = images.to(device, non_blocking=True)
        feats = backbone(images).float()
        feats = F.normalize(feats, dim=1)

        features.append(feats.cpu())
        labels_all.append(labels.cpu())

        total += int(labels.numel())

    return torch.cat(features, dim=0), torch.cat(labels_all, dim=0).long()


@torch.no_grad()
def _knn_accuracy(
    bank_features: torch.Tensor,
    bank_labels: torch.Tensor,
    query_features: torch.Tensor,
    query_labels: torch.Tensor,
    num_classes: int,
    k: int,
    temperature: float,
    chunk_size: int,
) -> float:
    k = min(int(k), int(bank_features.shape[0]))

    correct = 0
    total = 0

    for start in tqdm(range(0, query_features.shape[0], chunk_size), desc=f"selection kNN k={k}", leave=False):
        end = min(start + chunk_size, query_features.shape[0])
        q = query_features[start:end]

        sim = q @ bank_features.T
        values, indices = sim.topk(k=k, dim=1)

        topk_labels = bank_labels[indices]
        weights = torch.exp(values / temperature)

        votes = torch.zeros((q.shape[0], num_classes), dtype=torch.float32)

        for i in range(q.shape[0]):
            votes[i].index_add_(0, topk_labels[i], weights[i])

        preds = votes.argmax(dim=1)

        correct += int((preds == query_labels[start:end]).sum().item())
        total += int(end - start)

    return correct / max(total, 1)


def select_best_backbone_checkpoint_by_knn(
    cfg: dict[str, Any],
    model: torch.nn.Module,
    candidate_paths: list[str | Path],
    device: torch.device,
) -> dict[str, Any]:
    selection_cfg = cfg.get("checkpoint_selection", {}) or {}

    candidates = _unique_existing(candidate_paths)

    if not candidates:
        raise FileNotFoundError("No checkpoint candidates exist for kNN selection.")

    old_batch_size = cfg.get("data", {}).get("batch_size")
    cfg.setdefault("data", {})
    cfg["data"]["batch_size"] = int(
        selection_cfg.get(
            "batch_size",
            cfg.get("evaluation", {}).get("representation_batch_size", 256),
        )
    )

    loaders = build_loaders(cfg, self_supervised=False)

    if old_batch_size is not None:
        cfg["data"]["batch_size"] = old_batch_size

    max_bank_samples = selection_cfg.get("max_bank_samples", 10000)
    max_test_samples = selection_cfg.get("max_test_samples", 2000)

    max_bank_samples = None if max_bank_samples in {None, "none", "None", "null"} else int(max_bank_samples)
    max_test_samples = None if max_test_samples in {None, "none", "None", "null"} else int(max_test_samples)

    knn_ks = _as_list_of_ints(selection_cfg.get("knn_ks", selection_cfg.get("knn_k", 20)), [20])
    temperature = float(selection_cfg.get("temperature", cfg.get("evaluation", {}).get("knn_temperature", 0.07)))
    chunk_size = int(selection_cfg.get("query_chunk_size", cfg.get("evaluation", {}).get("knn_query_chunk_size", 256)))
    num_classes = int(cfg["model"]["num_classes"])

    rows: list[dict[str, Any]] = []

    print("Checkpoint selection by frozen-backbone kNN")
    print(f"  candidates: {len(candidates)}")
    print(f"  k values: {knn_ks}")
    print(f"  max_bank_samples: {max_bank_samples}")
    print(f"  max_test_samples: {max_test_samples}")

    for path in candidates:
        print(f"\nEvaluating candidate: {path}")

        checkpoint = load_checkpoint(path, map_location=device)
        model.load_state_dict(checkpoint["model_state"], strict=True)

        bank_features, bank_labels = _extract_features(
            backbone=model.backbone,
            loader=loaders.train_eval,
            device=device,
            max_samples=max_bank_samples,
        )

        query_features, query_labels = _extract_features(
            backbone=model.backbone,
            loader=loaders.test,
            device=device,
            max_samples=max_test_samples,
        )

        for k in knn_ks:
            acc = _knn_accuracy(
                bank_features=bank_features,
                bank_labels=bank_labels,
                query_features=query_features,
                query_labels=query_labels,
                num_classes=num_classes,
                k=k,
                temperature=temperature,
                chunk_size=chunk_size,
            )

            rows.append(
                {
                    "checkpoint_path": str(path),
                    "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
                    "pretrain_loss": float(checkpoint.get("pretrain_loss", float("nan"))),
                    "prediction_loss": float(checkpoint.get("prediction_loss", float("nan"))),
                    "sigreg_loss": float(checkpoint.get("sigreg_loss", float("nan"))),
                    "knn_k": int(k),
                    "knn_accuracy": float(acc),
                    "bank_samples": int(bank_features.shape[0]),
                    "query_samples": int(query_features.shape[0]),
                    "feature_dim": int(bank_features.shape[1]),
                }
            )

    best = max(rows, key=lambda r: float(r["knn_accuracy"]))

    metrics_dir = Path(cfg.get("_experiment", {}).get("metrics_dir", "outputs/metrics"))
    output_dir = ensure_dir(metrics_dir / "checkpoint_selection")

    csv_path = output_dir / "backbone_checkpoint_selection_knn.csv"
    yaml_path = output_dir / "backbone_checkpoint_selection_knn.yaml"

    pd.DataFrame(rows).to_csv(csv_path, index=False)

    result = {
        "selected_checkpoint_path": best["checkpoint_path"],
        "selected_epoch": best["checkpoint_epoch"],
        "best_knn_accuracy": best["knn_accuracy"],
        "knn_k": best["knn_k"],
        "num_candidates": len(candidates),
        "selection_csv": str(csv_path),
        "selection_yaml": str(yaml_path),
        "best_row": best,
    }

    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(result, f, sort_keys=False)

    print("\nCheckpoint selection completed.")
    print(f"  selected: {result['selected_checkpoint_path']}")
    print(f"  best kNN: {result['best_knn_accuracy']:.4f} at k={result['knn_k']}")
    print(f"  saved CSV: {csv_path}")

    return result


def maybe_select_checkpoint_by_knn(
    cfg: dict[str, Any],
    model: torch.nn.Module,
    candidate_paths: list[str | Path],
    best_path: str | Path,
    device: torch.device,
) -> None:
    selection_cfg = cfg.get("checkpoint_selection", {}) or {}

    if not bool(selection_cfg.get("enabled", False)):
        return

    if str(selection_cfg.get("select_backbone_by", "pretrain_loss")).lower() != "knn":
        return

    result = select_best_backbone_checkpoint_by_knn(
        cfg=cfg,
        model=model,
        candidate_paths=candidate_paths,
        device=device,
    )

    selected_checkpoint = load_checkpoint(result["selected_checkpoint_path"], map_location=device)
    selected_checkpoint["checkpoint_selection"] = result

    # From now on, backbone_best_path means best representation checkpoint.
    save_checkpoint(best_path, selected_checkpoint)