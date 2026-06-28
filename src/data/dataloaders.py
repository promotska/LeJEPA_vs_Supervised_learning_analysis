from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10

from src.data.transforms import (
    build_eval_transform,
    build_lejepa_train_transform,
    build_supervised_train_transform,
)
from src.utils import seed_worker


@dataclass(frozen=True)
class CifarLoaders:
    train: DataLoader
    train_eval: DataLoader
    val: DataLoader
    test: DataLoader


class HFCifar10Dataset(Dataset):
    """
    CIFAR-10 wrapper for a dataset saved by:

        ds = load_dataset("uoft-cs/cifar10")
        ds.save_to_disk("datasets/hf_cifar10")
    """

    def __init__(self, dataset_dir: str | Path, split: str, transform=None):
        try:
            from datasets import load_from_disk
        except ImportError as exc:
            raise ImportError(
                "Hugging Face datasets is required. Install with: pip install datasets"
            ) from exc

        dataset_dir = Path(dataset_dir)
        dataset_dict = load_from_disk(str(dataset_dir))

        if split not in dataset_dict:
            raise KeyError(
                f"Split '{split}' not found in {dataset_dir}. "
                f"Available splits: {list(dataset_dict.keys())}"
            )

        self.dataset = dataset_dict[split]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def _get_image(self, row: dict[str, Any]):
        if "img" in row:
            image = row["img"]
        elif "image" in row:
            image = row["image"]
        else:
            raise KeyError(f"No image column found. Available keys: {list(row.keys())}")

        if hasattr(image, "convert"):
            image = image.convert("RGB")

        return image

    def _get_label(self, row: dict[str, Any]) -> int:
        if "label" in row:
            return int(row["label"])
        if "fine_label" in row:
            return int(row["fine_label"])
        raise KeyError(f"No label column found. Available keys: {list(row.keys())}")

    def __getitem__(self, index: int):
        row = self.dataset[index]
        image = self._get_image(row)
        label = self._get_label(row)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def _build_dataset(root: Path, train: bool, transform):
    """
    Priority:
    1. Hugging Face saved dataset: root/hf_cifar10
    2. Torchvision CIFAR-10 folder: root/cifar-10-batches-py

    No downloading is performed here.
    """
    hf_dir = root / "hf_cifar10"

    if hf_dir.exists():
        split = "train" if train else "test"
        return HFCifar10Dataset(hf_dir, split=split, transform=transform)

    return CIFAR10(
        root=str(root),
        train=train,
        download=False,
        transform=transform,
    )


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


def build_cifar10_loaders(
    root: str | Path,
    batch_size: int,
    num_workers: int,
    val_fraction: float,
    seed: int,
    self_supervised: bool = False,
) -> CifarLoaders:
    root = Path(root)

    train_transform = (
        build_lejepa_train_transform()
        if self_supervised
        else build_supervised_train_transform()
    )
    eval_transform = build_eval_transform()

    train_full = _build_dataset(root=root, train=True, transform=train_transform)
    val_full = _build_dataset(root=root, train=True, transform=eval_transform)

    # Full train set with eval transforms.
    # This is what we use as the kNN feature bank.
    train_eval_full = _build_dataset(root=root, train=True, transform=eval_transform)

    test = _build_dataset(root=root, train=False, transform=eval_transform)

    train_subset, val_subset = _make_train_val_subsets(
        train_dataset=train_full,
        val_dataset=val_full,
        val_fraction=val_fraction,
        seed=seed,
    )

    common = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "persistent_workers": num_workers > 0,
    }

    return CifarLoaders(
        train=DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=self_supervised,
            **common,
        ),
        train_eval=DataLoader(
            train_eval_full,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            **common,
        ),
        val=DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            **common,
        ),
        test=DataLoader(
            test,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            **common,
        ),
    )