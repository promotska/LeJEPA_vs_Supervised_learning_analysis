from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.globals import CIFAR10_MEAN, CIFAR10_STD
from src.utils import ensure_dir


def denormalize_cifar(image: torch.Tensor) -> np.ndarray:
    """Convert normalized CIFAR tensor [3,H,W] to numpy image [H,W,3]."""
    mean = torch.tensor(CIFAR10_MEAN, dtype=image.dtype, device=image.device).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD, dtype=image.dtype, device=image.device).view(3, 1, 1)
    img = image * std + mean
    img = img.clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy()
    return img


def save_map_grid(
    image: torch.Tensor,
    pca_map: np.ndarray,
    xai_map: np.ndarray,
    output_path: str | Path,
    title: str,
) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    img = denormalize_cifar(image)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    axes[0].imshow(img)
    axes[0].set_title("image")
    axes[1].imshow(pca_map, cmap="magma")
    axes[1].set_title("PCA mask")
    axes[2].imshow(xai_map, cmap="magma")
    axes[2].set_title("Grad-CAM")
    axes[3].imshow(img)
    axes[3].imshow(xai_map, cmap="magma", alpha=0.45)
    axes[3].set_title("overlay")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
