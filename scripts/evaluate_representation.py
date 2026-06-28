from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataloaders import build_loaders
from src.experiment import prepare_experiment, update_manifest
from src.networks.resnet import (
    LinearProbeResNet18Cifar,
    ResNet18CifarBackbone,
    SupervisedResNet18Cifar,
)
from src.networks.vit import SupervisedViTCifar
from src.networks.factory import load_classifier_from_checkpoint
from src.utils import ensure_dir, get_device, load_yaml, set_seed


def _parse_ks(value: Any) -> list[int]:
    if value is None:
        return [1, 5, 10, 20, 50]
    if isinstance(value, list):
        return [int(v) for v in value]
    if isinstance(value, str):
        return [int(v.strip()) for v in value.split(",") if v.strip()]
    return [int(value)]


def load_model(config: dict[str, Any], mode: str, device: torch.device) -> torch.nn.Module:
    return load_classifier_from_checkpoint(
        cfg=config,
        mode=mode,
        device=device,
        requires_grad=False,
    )


@torch.no_grad()
def classifier_accuracy(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    max_samples: int | None = None,
) -> dict[str, float | int]:
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="classifier/probe accuracy", leave=False):
        if max_samples is not None and total >= max_samples:
            break

        if max_samples is not None and total + images.shape[0] > max_samples:
            remaining = max_samples - total
            images = images[:remaining]
            labels = labels[:remaining]

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        preds = logits.argmax(dim=1)

        correct += int((preds == labels).sum().item())
        total += int(labels.numel())

    return {"accuracy": correct / max(total, 1), "correct": correct, "total": total}


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    max_samples: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = []
    labels_all = []
    total = 0

    for images, labels in tqdm(loader, desc="extract features", leave=False):
        if max_samples is not None and total >= max_samples:
            break

        if max_samples is not None and total + images.shape[0] > max_samples:
            remaining = max_samples - total
            images = images[:remaining]
            labels = labels[:remaining]

        images = images.to(device, non_blocking=True)
        feats = model.forward_features(images)
        feats = F.normalize(feats.float(), dim=1)

        features.append(feats.cpu())
        labels_all.append(labels.cpu())
        total += int(labels.numel())

    return torch.cat(features, dim=0), torch.cat(labels_all, dim=0)


@torch.no_grad()
def knn_accuracy_for_k(
    bank_features: torch.Tensor,
    bank_labels: torch.Tensor,
    query_features: torch.Tensor,
    query_labels: torch.Tensor,
    num_classes: int,
    k: int,
    temperature: float,
    query_chunk_size: int,
) -> dict[str, float | int]:
    bank_features = bank_features.float()
    query_features = query_features.float()
    k = min(k, bank_features.shape[0])

    correct = 0
    total = 0

    for start in tqdm(range(0, query_features.shape[0], query_chunk_size), desc=f"kNN k={k}", leave=False):
        end = min(start + query_chunk_size, query_features.shape[0])
        q = query_features[start:end]

        sim = q @ bank_features.T
        values, indices = sim.topk(k=k, dim=1)
        topk_labels = bank_labels[indices]
        weights = torch.exp(values / temperature)

        votes = torch.zeros((q.shape[0], num_classes), dtype=torch.float32)
        for i in range(q.shape[0]):
            votes[i].index_add_(0, topk_labels[i], weights[i])

        preds = votes.argmax(dim=1)
        labels = query_labels[start:end]
        correct += int((preds == labels).sum().item())
        total += int(labels.numel())

    return {"k": int(k), "accuracy": correct / max(total, 1), "correct": correct, "total": total}


def evaluate_representation(config_path: str, mode: str) -> None:
    cfg = load_yaml(config_path)

    exp = prepare_experiment(
        cfg=cfg,
        config_path=config_path,
        run_kind=f"evaluate_representation_{mode}",
        mode=mode,
    )

    try:
        set_seed(int(cfg["project"]["seed"]))
        device = get_device()
        print(f"Using device: {device}")

        batch_size = int(cfg["evaluation"].get("representation_batch_size", cfg["evaluation"].get("batch_size", 256)))
        max_test_samples = cfg["evaluation"].get("representation_max_test_samples", None)
        max_bank_samples = cfg["evaluation"].get("representation_max_bank_samples", None)
        knn_ks = _parse_ks(cfg["evaluation"].get("knn_ks", [1, 5, 10, 20, 50]))
        temperature = float(cfg["evaluation"].get("knn_temperature", 0.07))
        query_chunk_size = int(cfg["evaluation"].get("knn_query_chunk_size", 256))

        if max_test_samples is not None:
            max_test_samples = int(max_test_samples)
        if max_bank_samples is not None:
            max_bank_samples = int(max_bank_samples)

        num_classes = int(cfg["model"]["num_classes"])

        print("Representation evaluation settings:")
        print(f"  mode: {mode}")
        print(f"  architecture: {cfg['model'].get('architecture')}")
        print(f"  checkpoint: {cfg['evaluation']['checkpoint_path']}")
        print(f"  batch_size: {batch_size}")
        print(f"  max_test_samples: {max_test_samples}")
        print(f"  max_bank_samples: {max_bank_samples}")
        print(f"  kNN ks: {knn_ks}")
        print("  feature bank: full train set with eval transforms")
        print("  query set: test set")

        cfg.setdefault("data", {})
        cfg["data"]["batch_size"] = batch_size
        loaders = build_loaders(cfg, self_supervised=False)

        model = load_model(cfg, mode=mode, device=device)
        started = time.time()

        cls_metrics = classifier_accuracy(model, loaders.test, device, max_samples=max_test_samples)
        bank_features, bank_labels = extract_features(model, loaders.train_eval, device, max_samples=max_bank_samples)
        query_features, query_labels = extract_features(model, loaders.test, device, max_samples=max_test_samples)

        knn_rows = []
        for k in knn_ks:
            knn_rows.append(
                knn_accuracy_for_k(
                    bank_features=bank_features,
                    bank_labels=bank_labels,
                    query_features=query_features,
                    query_labels=query_labels,
                    num_classes=num_classes,
                    k=k,
                    temperature=temperature,
                    query_chunk_size=query_chunk_size,
                )
            )

        best_knn = max(knn_rows, key=lambda r: float(r["accuracy"]))
        elapsed = time.time() - started

        result = {
            "mode": mode,
            "architecture": cfg["model"].get("architecture"),
            "checkpoint_path": cfg["evaluation"]["checkpoint_path"],
            "classifier_or_probe_accuracy": cls_metrics,
            "knn": {
                "feature_bank": "full_train_eval_transform",
                "query": "test",
                "bank_samples": int(bank_features.shape[0]),
                "query_samples": int(query_features.shape[0]),
                "feature_dim": int(bank_features.shape[1]),
                "temperature": temperature,
                "results": knn_rows,
                "best": best_knn,
            },
            "elapsed_minutes": elapsed / 60,
        }

        metrics_dir = ensure_dir(exp.metrics_dir)
        output_yaml = metrics_dir / f"representation_{mode}.yaml"
        output_csv = metrics_dir / f"representation_{mode}.csv"

        with output_yaml.open("w", encoding="utf-8") as f:
            yaml.safe_dump(result, f, sort_keys=False)

        flat_rows = []
        for row in knn_rows:
            flat_rows.append(
                {
                    "mode": mode,
                    "architecture": cfg["model"].get("architecture"),
                    "checkpoint_path": cfg["evaluation"]["checkpoint_path"],
                    "classifier_or_probe_accuracy": cls_metrics["accuracy"],
                    "classifier_or_probe_correct": cls_metrics["correct"],
                    "classifier_or_probe_total": cls_metrics["total"],
                    "knn_k": row["k"],
                    "knn_accuracy": row["accuracy"],
                    "knn_correct": row["correct"],
                    "knn_total": row["total"],
                    "bank_samples": int(bank_features.shape[0]),
                    "query_samples": int(query_features.shape[0]),
                    "feature_dim": int(bank_features.shape[1]),
                    "temperature": temperature,
                    "elapsed_minutes": elapsed / 60,
                }
            )

        pd.DataFrame(flat_rows).to_csv(output_csv, index=False)

        print("\nRepresentation evaluation completed.")
        print(f"Classifier/probe accuracy: {cls_metrics['accuracy']:.4f}")
        print(f"Best kNN accuracy: {best_knn['accuracy']:.4f} at k={best_knn['k']}")
        print(f"Saved YAML: {output_yaml}")
        print(f"Saved CSV: {output_csv}")

        update_manifest(
            exp=exp,
            cfg=cfg,
            status="completed",
            extra={"representation_metrics": result, "output_yaml": str(output_yaml), "output_csv": str(output_csv)},
        )
    except Exception as exc:
        update_manifest(
            exp=exp,
            cfg=cfg,
            status="failed",
            extra={"error": repr(exc), "traceback": traceback.format_exc()},
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True, choices=["supervised", "lejepa", "vit_supervised", "vit_lejepa"])
    args = parser.parse_args()

    evaluate_representation(config_path=args.config, mode=args.mode)


if __name__ == "__main__":
    main()
