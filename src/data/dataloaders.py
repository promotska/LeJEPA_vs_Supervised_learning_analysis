from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
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
    val: DataLoader
    test: DataLoader


def build_cifar10_loaders(
    root: str | Path,
    batch_size: int,
    num_workers: int,
    val_fraction: float,
    seed: int,
    self_supervised: bool = False,
) -> CifarLoaders:
    root = Path(root)
    train_transform = build_lejepa_train_transform() if self_supervised else build_supervised_train_transform()
    eval_transform = build_eval_transform()

    train_full = CIFAR10(root=str(root), train=True, download=True, transform=train_transform)
    val_full = CIFAR10(root=str(root), train=True, download=True, transform=eval_transform)
    test = CIFAR10(root=str(root), train=False, download=True, transform=eval_transform)

    val_size = int(len(train_full) * val_fraction)
    train_size = len(train_full) - val_size
    generator = torch.Generator().manual_seed(seed)

    train_subset, _ = random_split(train_full, [train_size, val_size], generator=generator)
    _, val_subset = random_split(val_full, [train_size, val_size], generator=generator)

    common = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "persistent_workers": num_workers > 0,
    }

    return CifarLoaders(
        train=DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=self_supervised, **common),
        val=DataLoader(val_subset, batch_size=batch_size, shuffle=False, drop_last=False, **common),
        test=DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=False, **common),
    )
