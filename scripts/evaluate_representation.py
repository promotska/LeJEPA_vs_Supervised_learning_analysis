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

from src.data.dataloaders import build_cifar10_loaders
from src.experiment import prepare_experiment, update_manifest
from src.networks.resnet import (
    LinearProbeResNet18Cifar,
    ResNet18CifarBackbone,
    SupervisedResNet18Cifar,
)
from src.training.checkpointing import load_checkpoint
from src.utils import ensure_dir, get_device, load_yaml, set_seed


def load_model(config: dict[str, Any], mode: str, device: torch.device) -> torch.nn.Module:
    checkpoint_path = config["evaluation"]["checkpoint_path"]

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path, map_location=device)

    if mode == "supervised":
        model = SupervisedResNet18Cifar(num_classes=int(config["model"]["num_classes"]))
        model.load_state_dict(checkpoint["model_state"])
    elif mode == "lejepa":
        backbone = ResNet18CifarBackbone()
        model = LinearProbeResNet18Cifar(
            backbone,
            num_classes=int(config["model"]["num_classes"]),
        )
        model.load_state_dict(checkpoint["model_state"])
    else:
        raise ValueError("mode must be 'supervised' or 'lejepa'.")

    return model.to(device).eval()


@torch.no_grad()
def classifier_accuracy(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    max_samples: int | None = None,
) -> dict[str, float]:
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

    return {
        "accuracy": correct / max(total, 1),
        "correct": correct,
        "total": total,
    }


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
def knn_accuracy(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    query_features: torch.Tensor,
    query_labels: torch.Tensor,
    num_classes: int,
    k: int = 20,
    temperature: float = 0.07,
    query_chunk_size: int = 256,
) -> dict[str, float]:
    """
    Cosine kNN with soft voting.

    train_features: [N, D], normalized
    query_features: [M, D], normalized
    """
    train_features = train_features.float()
    query_features = query_features.float()

    correct = 0
    total = 0

    for start in tqdm(range(0, query_features.shape[0], query_chunk_size), desc="kNN", leave=False):
        end = min(start + query_chunk_size, query_features.shape[0])
        q = query_features[start:end]

        sim = q @ train_features.T
        values, indices = sim.topk(k=min(k, train_features.shape[0]), dim=1)

        topk_labels = train_labels[indices]
        weights = torch.exp(values / temperature)

        votes = torch.zeros((q.shape[0], num_classes), dtype=torch.float32)

        for i in range(q.shape[0]):
            votes[i].index_add_(0, topk_labels[i], weights[i])

        preds = votes.argmax(dim=1)
        labels = query_labels[start:end]

        correct += int((preds == labels).sum().item())
        total += int(labels.numel())

    return {
        "knn_accuracy": correct / max(total, 1),
        "knn_correct": correct,
        "knn_total": total,
        "k": k,
        "temperature": temperature,
    }


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

        batch_size = int(cfg["evaluation"].get("representation_batch_size", cfg["evaluation"].get("batch_size", 128)))
        max_test_samples = cfg["evaluation"].get("representation_max_test_samples", None)
        max_bank_samples = cfg["evaluation"].get("representation_max_bank_samples", None)

        if max_test_samples is not None:
            max_test_samples = int(max_test_samples)
        if max_bank_samples is not None:
            max_bank_samples = int(max_bank_samples)

        k = int(cfg["evaluation"].get("knn_k", 20))
        num_classes = int(cfg["model"]["num_classes"])

        print("Representation evaluation settings:")
        print(f"  mode: {mode}")
        print(f"  checkpoint: {cfg['evaluation']['checkpoint_path']}")
        print(f"  batch_size: {batch_size}")
        print(f"  max_test_samples: {max_test_samples}")
        print(f"  max_bank_samples: {max_bank_samples}")
        print(f"  kNN k: {k}")

        loaders = build_cifar10_loaders(
            root=cfg["data"]["root"],
            batch_size=batch_size,
            num_workers=int(cfg["data"]["num_workers"]),
            val_fraction=float(cfg["data"]["val_fraction"]),
            seed=int(cfg["project"]["seed"]),
            self_supervised=False,
        )

        model = load_model(cfg, mode=mode, device=device)

        started = time.time()

        cls_metrics = classifier_accuracy(
            model=model,
            loader=loaders.test,
            device=device,
            max_samples=max_test_samples,
        )

        # We use validation split as the kNN feature bank because it has eval transforms.
        # This avoids using random-crop training transforms for the feature bank.
        bank_features, bank_labels = extract_features(
            model=model,
            loader=loaders.val,
            device=device,
            max_samples=max_bank_samples,
        )

        query_features, query_labels = extract_features(
            model=model,
            loader=loaders.test,
            device=device,
            max_samples=max_test_samples,
        )

        knn_metrics = knn_accuracy(
            train_features=bank_features,
            train_labels=bank_labels,
            query_features=query_features,
            query_labels=query_labels,
            num_classes=num_classes,
            k=k,
        )

        elapsed = time.time() - started

        result = {
            "mode": mode,
            "checkpoint_path": cfg["evaluation"]["checkpoint_path"],
            "classifier_or_probe_accuracy": cls_metrics,
            "knn": knn_metrics,
            "feature_bank": {
                "source": "validation split",
                "num_samples": int(bank_features.shape[0]),
                "feature_dim": int(bank_features.shape[1]),
            },
            "query": {
                "source": "test split",
                "num_samples": int(query_features.shape[0]),
            },
            "elapsed_minutes": elapsed / 60,
        }

        metrics_dir = ensure_dir(exp.metrics_dir)
        output_yaml = metrics_dir / f"representation_{mode}.yaml"
        output_csv = metrics_dir / f"representation_{mode}.csv"

        with output_yaml.open("w", encoding="utf-8") as f:
            yaml.safe_dump(result, f, sort_keys=False)

        flat = {
            "mode": mode,
            "checkpoint_path": cfg["evaluation"]["checkpoint_path"],
            "accuracy": cls_metrics["accuracy"],
            "correct": cls_metrics["correct"],
            "total": cls_metrics["total"],
            "knn_accuracy": knn_metrics["knn_accuracy"],
            "knn_correct": knn_metrics["knn_correct"],
            "knn_total": knn_metrics["knn_total"],
            "knn_k": knn_metrics["k"],
            "bank_samples": int(bank_features.shape[0]),
            "query_samples": int(query_features.shape[0]),
            "feature_dim": int(bank_features.shape[1]),
            "elapsed_minutes": elapsed / 60,
        }
        pd.DataFrame([flat]).to_csv(output_csv, index=False)

        print()
        print("Representation evaluation completed.")
        print(f"Classifier/probe accuracy: {cls_metrics['accuracy']:.4f}")
        print(f"kNN accuracy: {knn_metrics['knn_accuracy']:.4f}")
        print(f"Saved YAML: {output_yaml}")
        print(f"Saved CSV: {output_csv}")

        update_manifest(
            exp=exp,
            cfg=cfg,
            status="completed",
            extra={
                "representation_metrics": result,
                "output_yaml": str(output_yaml),
                "output_csv": str(output_csv),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True, choices=["supervised", "lejepa"])
    args = parser.parse_args()

    evaluate_representation(config_path=args.config, mode=args.mode)


if __name__ == "__main__":
    main()