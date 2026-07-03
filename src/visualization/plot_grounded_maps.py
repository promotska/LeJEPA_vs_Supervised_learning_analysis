from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils import ensure_dir


def denormalize_image(
    image: torch.Tensor,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    mean_t = torch.tensor(mean, dtype=image.dtype, device=image.device).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=image.dtype, device=image.device).view(3, 1, 1)
    img = (image * std_t + mean_t).clamp(0, 1)
    return img.detach().cpu().permute(1, 2, 0).numpy()


def save_grounded_map_grid(
    image: torch.Tensor,
    gt_mask: np.ndarray,
    pca_map: np.ndarray,
    xai_map: np.ndarray,
    output_path: str | Path,
    title: str,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    img = denormalize_image(image, mean=mean, std=std)

    fig, axes = plt.subplots(1, 6, figsize=(18, 3.2))
    axes[0].imshow(img)
    axes[0].set_title("image")
    axes[1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("GT mask")
    axes[2].imshow(pca_map, cmap="magma", vmin=0, vmax=1)
    axes[2].set_title("PCA")
    axes[3].imshow(xai_map, cmap="magma", vmin=0, vmax=1)
    axes[3].set_title("XAI")
    axes[4].imshow(img)
    axes[4].imshow(gt_mask, cmap="Greens", alpha=0.35, vmin=0, vmax=1)
    axes[4].set_title("GT overlay")
    axes[5].imshow(img)
    axes[5].imshow(xai_map, cmap="magma", alpha=0.45, vmin=0, vmax=1)
    axes[5].set_title("XAI overlay")

    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
