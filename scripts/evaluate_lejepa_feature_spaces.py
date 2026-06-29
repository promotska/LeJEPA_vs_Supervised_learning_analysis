from __future__ import annotations

import argparse
import sys
import time
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
from src.networks.factory import build_lejepa_model
from src.training.checkpointing import load_checkpoint
from src.utils import ensure_dir, get_device, load_yaml, set_seed


def _parse_ks(value: Any) -> list[int]:
    if value is None:
        return [1, 5, 10, 20, 50, 100]
    if isinstance(value, list):
        return [int(v) for v in value]
    if isinstance(value, str):
        return [int(v.strip()) for v in value.split(",") if v.strip()]
    return [int(value)]


def _load_lejepa_model(cfg: dict[str, Any], checkpoint_path: str | Path, device: torch.device):
    checkpoint_path = Path(checkpoint_path)

    if "probe" in checkpoint_path.name:
        raise ValueError(
            f"You passed a probe checkpoint: {checkpoint_path}\n"
            "For projector-feature evaluation, use backbone_best_path or backbone_last_path."
        )

    checkpoint = load_checkpoint(checkpoint_path, map_location=device)

    if "model_state" not in checkpoint:
        raise KeyError(
            f"Checkpoint {checkpoint_path} does not contain 'model_state'. "
            "Use a LeJEPA backbone checkpoint saved during pretraining."
        )

    model = build_lejepa_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    return model


def _forward_backbone(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "forward_backbone"):
        return model.forward_backbone(images)

    if hasattr(model, "forward_features"):
        return model.forward_features(images)

    if hasattr(model, "backbone") and hasattr(model.backbone, "forward_features"):
        return model.backbone.forward_features(images)

    if hasattr(model, "backbone"):
        return model.backbone(images)

    raise TypeError(f"Cannot extract backbone features from model type: {type(model)}")


def _extract_one_feature_source(
    model: torch.nn.Module,
    images: torch.Tensor,
    feature_source: str,
) -> torch.Tensor:
    h = _forward_backbone(model, images)

    if feature_source == "backbone":
        return h

    if feature_source == "projector":
        if not hasattr(model, "projector"):
            raise AttributeError("Model has no projector.")
        z = model.projector(h)
        return z

    if feature_source == "predictor":
        if not hasattr(model, "projector") or not hasattr(model, "predictor"):
            raise AttributeError("Model has no projector/predictor.")
        z = model.projector(h)
        p = model.predictor(z)
        return p

    raise ValueError(f"Unknown feature_source={feature_source!r}")


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    feature_source: str,
    max_samples: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    features = []
    labels_all = []
    raw_std_means = []
    raw_std_mins = []
    raw_norm_means = []

    total = 0

    for images, labels in tqdm(loader, desc=f"extract {feature_source}", leave=False):
        if max_samples is not None and total >= max_samples:
            break

        if max_samples is not None and total + images.shape[0] > max_samples:
            remaining = max_samples - total
            images = images[:remaining]
            labels = labels[:remaining]

        images = images.to(device, non_blocking=True)
        raw_feats = _extract_one_feature_source(model, images, feature_source=feature_source).float()

        # Collapse diagnostics before normalization.
        batch_std = raw_feats.std(dim=0, unbiased=False)
        raw_std_means.append(batch_std.mean().detach().cpu())
        raw_std_mins.append(batch_std.min().detach().cpu())
        raw_norm_means.append(raw_feats.norm(dim=1).mean().detach().cpu())

        feats = F.normalize(raw_feats, dim=1)

        features.append(feats.detach().cpu())
        labels_all.append(labels.detach().cpu())

        total += int(labels.numel())

    features_t = torch.cat(features, dim=0)
    labels_t = torch.cat(labels_all, dim=0)

    diagnostics = {
        "raw_feature_std_mean": float(torch.stack(raw_std_means).mean().item()),
        "raw_feature_std_min": float(torch.stack(raw_std_mins).mean().item()),
        "raw_feature_norm_mean": float(torch.stack(raw_norm_means).mean().item()),
        "feature_dim": int(features_t.shape[1]),
        "num_samples": int(features_t.shape[0]),
    }

    return features_t, labels_t, diagnostics


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
    bank_labels = bank_labels.long()
    query_labels = query_labels.long()

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

    return {
        "k": int(k),
        "accuracy": correct / max(total, 1),
        "correct": correct,
        "total": total,
    }


def evaluate_feature_source(
    cfg: dict[str, Any],
    model: torch.nn.Module,
    loaders,
    device: torch.device,
    feature_source: str,
    max_bank_samples: int | None,
    max_test_samples: int | None,
    knn_ks: list[int],
    temperature: float,
    query_chunk_size: int,
) -> dict[str, Any]:
    bank_features, bank_labels, bank_diag = extract_features(
        model=model,
        loader=loaders.train_eval,
        device=device,
        feature_source=feature_source,
        max_samples=max_bank_samples,
    )

    query_features, query_labels, query_diag = extract_features(
        model=model,
        loader=loaders.test,
        device=device,
        feature_source=feature_source,
        max_samples=max_test_samples,
    )

    num_classes = int(cfg["model"]["num_classes"])

    knn_rows = []
    for k in knn_ks:
        row = knn_accuracy_for_k(
            bank_features=bank_features,
            bank_labels=bank_labels,
            query_features=query_features,
            query_labels=query_labels,
            num_classes=num_classes,
            k=k,
            temperature=temperature,
            query_chunk_size=query_chunk_size,
        )
        knn_rows.append(row)

    best = max(knn_rows, key=lambda r: float(r["accuracy"]))

    return {
        "feature_source": feature_source,
        "bank_diagnostics": bank_diag,
        "query_diagnostics": query_diag,
        "knn": {
            "results": knn_rows,
            "best": best,
            "bank_samples": int(bank_features.shape[0]),
            "query_samples": int(query_features.shape[0]),
            "feature_dim": int(bank_features.shape[1]),
            "temperature": temperature,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Use backbone_best/backbone_last checkpoint. If omitted, uses cfg['checkpoints']['backbone_last_path'].",
    )
    parser.add_argument(
        "--feature-source",
        default="all",
        choices=["all", "backbone", "projector", "predictor"],
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg["project"]["seed"]))
    device = get_device()

    checkpoint_path = args.checkpoint or cfg["checkpoints"]["backbone_last_path"]

    batch_size = int(cfg["evaluation"].get("representation_batch_size", 256))
    max_test_samples = cfg["evaluation"].get("representation_max_test_samples", None)
    max_bank_samples = cfg["evaluation"].get("representation_max_bank_samples", None)
    knn_ks = _parse_ks(cfg["evaluation"].get("knn_ks", [1, 5, 10, 20, 50, 100]))
    temperature = float(cfg["evaluation"].get("knn_temperature", 0.07))
    query_chunk_size = int(cfg["evaluation"].get("knn_query_chunk_size", 256))

    if max_test_samples is not None:
        max_test_samples = int(max_test_samples)
    if max_bank_samples is not None:
        max_bank_samples = int(max_bank_samples)

    cfg.setdefault("data", {})
    cfg["data"]["batch_size"] = batch_size

    print("Evaluating LeJEPA feature spaces")
    print(f"  config: {args.config}")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  device: {device}")
    print(f"  batch_size: {batch_size}")
    print(f"  max_bank_samples: {max_bank_samples}")
    print(f"  max_test_samples: {max_test_samples}")

    loaders = build_loaders(cfg, self_supervised=False)
    model = _load_lejepa_model(cfg, checkpoint_path=checkpoint_path, device=device)

    if args.feature_source == "all":
        sources = ["backbone", "projector", "predictor"]
    else:
        sources = [args.feature_source]

    started = time.time()
    results = []

    for source in sources:
        print(f"\n=== Feature source: {source} ===")
        result = evaluate_feature_source(
            cfg=cfg,
            model=model,
            loaders=loaders,
            device=device,
            feature_source=source,
            max_bank_samples=max_bank_samples,
            max_test_samples=max_test_samples,
            knn_ks=knn_ks,
            temperature=temperature,
            query_chunk_size=query_chunk_size,
        )
        results.append(result)

        best = result["knn"]["best"]
        diag = result["query_diagnostics"]
        print(f"Best kNN: {best['accuracy']:.4f} at k={best['k']}")
        print(
            "Diagnostics: "
            f"std_mean={diag['raw_feature_std_mean']:.6f}, "
            f"std_min={diag['raw_feature_std_min']:.6f}, "
            f"norm_mean={diag['raw_feature_norm_mean']:.6f}, "
            f"dim={diag['feature_dim']}"
        )

    elapsed = time.time() - started

    ckpt_stem = Path(checkpoint_path).stem
    arch = str(cfg["model"].get("architecture", "model"))

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        # If the resolved checkpoint is:
        # experiments/<experiment-name>/checkpoints/model.pt
        # then write to:
        # experiments/<experiment-name>/metrics/feature_spaces/
        checkpoint_path_obj = Path(checkpoint_path)

        if "checkpoints" in checkpoint_path_obj.parts:
            checkpoints_idx = checkpoint_path_obj.parts.index("checkpoints")
            experiment_dir = Path(*checkpoint_path_obj.parts[:checkpoints_idx])
            output_dir = experiment_dir / "metrics" / "feature_spaces"
        else:
            # Fallback for direct non-relocated runs.
            output_dir = Path("outputs/metrics/feature_spaces")

    output_dir = ensure_dir(output_dir)

    output_yaml = output_dir / f"{arch}_{ckpt_stem}_feature_spaces.yaml"
    output_csv = output_dir / f"{arch}_{ckpt_stem}_feature_spaces.csv"

    payload = {
        "config": args.config,
        "checkpoint": str(checkpoint_path),
        "architecture": arch,
        "elapsed_minutes": elapsed / 60,
        "results": results,
    }

    with output_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)

    flat_rows = []
    for result in results:
        source = result["feature_source"]
        bank_diag = result["bank_diagnostics"]
        query_diag = result["query_diagnostics"]
        for row in result["knn"]["results"]:
            flat_rows.append(
                {
                    "architecture": arch,
                    "checkpoint": str(checkpoint_path),
                    "feature_source": source,
                    "knn_k": row["k"],
                    "knn_accuracy": row["accuracy"],
                    "knn_correct": row["correct"],
                    "knn_total": row["total"],
                    "feature_dim": result["knn"]["feature_dim"],
                    "bank_samples": result["knn"]["bank_samples"],
                    "query_samples": result["knn"]["query_samples"],
                    "temperature": result["knn"]["temperature"],
                    "bank_std_mean": bank_diag["raw_feature_std_mean"],
                    "bank_std_min": bank_diag["raw_feature_std_min"],
                    "bank_norm_mean": bank_diag["raw_feature_norm_mean"],
                    "query_std_mean": query_diag["raw_feature_std_mean"],
                    "query_std_min": query_diag["raw_feature_std_min"],
                    "query_norm_mean": query_diag["raw_feature_norm_mean"],
                    "elapsed_minutes": elapsed / 60,
                }
            )

    pd.DataFrame(flat_rows).to_csv(output_csv, index=False)

    print("\nSaved:")
    print(f"  {output_yaml}")
    print(f"  {output_csv}")


if __name__ == "__main__":
    main()