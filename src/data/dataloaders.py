from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder

from src.data.transforms import (
    build_eval_transform,
    build_lejepa_train_transform,
    build_supervised_train_transform,
)
from src.utils import seed_worker


@dataclass(frozen=True)
class DataLoaders:
    train: DataLoader
    train_eval: DataLoader
    val: DataLoader
    test: DataLoader


# Backward-compatible alias.
CifarLoaders = DataLoaders


class HFCifarDataset(Dataset):
    """Wrapper for CIFAR-like datasets saved with Hugging Face datasets.save_to_disk()."""

    def __init__(self, dataset_dir: str | Path, split: str, transform=None):
        try:
            from datasets import load_from_disk
        except ImportError as exc:
            raise ImportError("Install Hugging Face datasets with: pip install datasets") from exc

        dataset_dir = Path(dataset_dir)
        dataset_dict = load_from_disk(str(dataset_dir))

        if split not in dataset_dict:
            raise KeyError(
                f"Split {split!r} not found in {dataset_dir}. "
                f"Available splits: {list(dataset_dict.keys())}"
            )

        self.dataset = dataset_dict[split]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def _get_image(self, row: dict[str, Any]):
        for key in ("img", "image"):
            if key in row:
                image = row[key]
                return image.convert("RGB") if hasattr(image, "convert") else image
        raise KeyError(f"No image column found. Available keys: {list(row.keys())}")

    def _get_label(self, row: dict[str, Any]) -> int:
        for key in ("label", "fine_label", "coarse_label"):
            if key in row:
                return int(row[key])
        raise KeyError(f"No label column found. Available keys: {list(row.keys())}")

    def __getitem__(self, index: int):
        row = self.dataset[index]
        image = self._get_image(row)
        label = self._get_label(row)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


# Backward-compatible name.
HFCifar10Dataset = HFCifarDataset


def _dataset_name(cfg_or_name: dict[str, Any] | str) -> str:
    if isinstance(cfg_or_name, str):
        return cfg_or_name.lower()
    return str(cfg_or_name.get("data", {}).get("dataset", "CIFAR10")).lower()


def _hf_dir_for_dataset(root: Path, dataset_name: str) -> Path:
    if dataset_name in {"cifar10", "cifar-10"}:
        return root / "hf_cifar10"
    if dataset_name in {"cifar100", "cifar-100"}:
        return root / "hf_cifar100"
    return root / f"hf_{dataset_name}"


def _build_classification_dataset(
    dataset_name: str,
    root: Path,
    train: bool,
    transform,
) -> Dataset:
    dataset_name_l = dataset_name.lower()

    if dataset_name_l in {"cifar10", "cifar-10"}:
        hf_dir = _hf_dir_for_dataset(root, dataset_name_l)
        if hf_dir.exists():
            split = "train" if train else "test"
            return HFCifarDataset(hf_dir, split=split, transform=transform)
        return CIFAR10(root=str(root), train=train, download=False, transform=transform)

    if dataset_name_l in {"cifar100", "cifar-100"}:
        hf_dir = _hf_dir_for_dataset(root, dataset_name_l)
        if hf_dir.exists():
            split = "train" if train else "test"
            return HFCifarDataset(hf_dir, split=split, transform=transform)
        return CIFAR100(root=str(root), train=train, download=False, transform=transform)

    if dataset_name_l in {"imagefolder", "imagenet100", "imagenet-100"}:
        split_dir = root / ("train" if train else "val")
        if not split_dir.exists() and not train:
            split_dir = root / "test"
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Could not find ImageFolder split directory for dataset={dataset_name}: {split_dir}"
            )
        return ImageFolder(root=str(split_dir), transform=transform)

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _make_train_val_subsets(
    train_dataset: Dataset,
    val_dataset: Dataset,
    val_fraction: float,
    seed: int,
):
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    total_size = len(train_dataset)
    val_size = int(total_size * val_fraction)
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_size, generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    return Subset(train_dataset, train_indices), Subset(val_dataset, val_indices)


def _loader(dataset: Dataset, batch_size: int, shuffle: bool, drop_last: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        persistent_workers=num_workers > 0,
    )


def build_loaders(cfg: dict[str, Any], self_supervised: bool = False) -> DataLoaders:
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    lejepa_views_cfg = cfg.get("lejepa_views", {}) or cfg.get("lejepa", {}).get("views", {}) or {}

    dataset_name = str(data_cfg.get("dataset", "CIFAR10"))
    root = Path(data_cfg.get("root", "datasets"))
    batch_size = int(data_cfg.get("batch_size", 128))
    num_workers = int(data_cfg.get("num_workers", 4))
    val_fraction = float(data_cfg.get("val_fraction", 0.1))
    seed = int(cfg.get("project", {}).get("seed", 42))
    image_size = int(cfg.get("model", {}).get("image_size", data_cfg.get("image_size", 32)))

    if self_supervised:
        train_transform = build_lejepa_train_transform(
            image_size=image_size,
            num_global_views=int(lejepa_views_cfg.get("num_global_views", train_cfg.get("num_global_views", 2))),
            num_local_views=int(lejepa_views_cfg.get("num_local_views", train_cfg.get("num_local_views", 0))),
            global_scale=lejepa_views_cfg.get("global_scale", train_cfg.get("global_scale", (0.6, 1.0))),
            local_scale=lejepa_views_cfg.get("local_scale", train_cfg.get("local_scale", (0.2, 0.6))),
            color_jitter_strength=float(lejepa_views_cfg.get("color_jitter_strength", 0.25)),
            grayscale_p=float(lejepa_views_cfg.get("grayscale_p", 0.1)),
        )
    else:
        train_transform = build_supervised_train_transform(image_size=image_size)

    eval_transform = build_eval_transform(image_size=image_size)

    train_full = _build_classification_dataset(dataset_name, root, train=True, transform=train_transform)
    val_full = _build_classification_dataset(dataset_name, root, train=True, transform=eval_transform)
    train_eval_full = _build_classification_dataset(dataset_name, root, train=True, transform=eval_transform)
    test = _build_classification_dataset(dataset_name, root, train=False, transform=eval_transform)

    train_subset, val_subset = _make_train_val_subsets(
        train_dataset=train_full,
        val_dataset=val_full,
        val_fraction=val_fraction,
        seed=seed,
    )

    return DataLoaders(
        train=_loader(train_subset, batch_size=batch_size, shuffle=True, drop_last=self_supervised, num_workers=num_workers),
        train_eval=_loader(train_eval_full, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers),
        val=_loader(val_subset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers),
        test=_loader(test, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers),
    )


def build_cifar10_loaders(
    root: str | Path,
    batch_size: int,
    num_workers: int,
    val_fraction: float,
    seed: int,
    self_supervised: bool = False,
) -> DataLoaders:
    """Backward-compatible wrapper for older Stage 1 code."""
    cfg = {
        "project": {"seed": seed},
        "data": {
            "dataset": "CIFAR10",
            "root": str(root),
            "batch_size": batch_size,
            "num_workers": num_workers,
            "val_fraction": val_fraction,
        },
        "model": {"image_size": 32},
    }
    return build_loaders(cfg, self_supervised=self_supervised)
