from __future__ import annotations

"""
Population-PCA individual image export (PCA projections only).

Saves each image's population-PCA projection as an individual file.
No original images, no side-by-side comparisons.

Usage:
  python scripts/make_population_pca_individual.py \
    --config configs/cifar10_vit_lejepa_v4_short_lr1e4_sig005.yaml \
    --mode vit_lejepa --layer blocks.3 --num-images 12 \
    --out_dir experiments/experiment-c10-vit-lejepa-v4-short-lr1e4-sig005/pca_images/
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataloaders import build_loaders
from src.networks.factory import load_classifier_from_checkpoint
from src.utils import ensure_dir, get_device, get_module_by_name, load_yaml, set_seed
from src.xai.gradcam import _LayerHook

try:
    from src.globals import CIFAR10_MEAN, CIFAR10_STD
except Exception:
    CIFAR10_MEAN, CIFAR10_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


@torch.no_grad()
def extract_patches(model, images, layer, device):
    """Return (patch_matrix [B,N,D], grid_hw) for a ViT block or a ResNet conv layer."""
    backbone = model.backbone
    if hasattr(backbone, "grid_size"):  # ViT
        num_prefix = 1 + int(getattr(backbone, "num_registers", 0))
        _logits, captured = model.forward_with_intermediates(images, capture_layers=[layer])
        tokens = captured[layer][:, num_prefix:, :]  # drop CLS (+ registers)
        g = int(backbone.grid_size)
        return tokens, (g, g)
    # ResNet / conv: hook the layer, reshape [B,C,H,W] -> [B, H*W, C]
    hook = _LayerHook(get_module_by_name(model, layer))
    model(images)
    act = hook.data.activation
    hook.close()
    b, c, h, w = act.shape
    return act.permute(0, 2, 3, 1).reshape(b, h * w, c), (h, w)


@torch.no_grad()
def fit_population_pca(patch_list: list[torch.Tensor]):
    """Fit a shared 3-component basis over pooled patches. Returns (mean[D], comps[D,3])."""
    x = torch.cat([p.reshape(-1, p.shape[-1]) for p in patch_list], dim=0).float()
    mean = x.mean(dim=0)
    xc = x - mean
    cov = xc.transpose(0, 1) @ xc
    eigvals, eigvecs = torch.linalg.eigh(cov)
    comps = eigvecs[:, eigvals.argsort(descending=True)[:3]]  # [D,3]
    return mean, comps


def project_rgb(patches, grid_hw, mean, comps, lo, hi, out_size):
    """[N,D] patches -> upsampled [out_H,out_W,3] RGB image using global percentile scaling."""
    proj = (patches.float() - mean) @ comps           # [N,3]
    gh, gw = grid_hw
    grid = proj.reshape(1, gh, gw, 3).permute(0, 3, 1, 2)  # [1,3,gh,gw]
    grid = F.interpolate(grid, size=out_size, mode="bilinear", align_corners=False)
    rgb = grid[0].permute(1, 2, 0).cpu().numpy()
    rgb = (rgb - lo) / (hi - lo + 1e-8)
    return np.clip(rgb, 0.0, 1.0)


def save_individual_pca_images(
    patches_list: list[torch.Tensor],
    grid_hw: tuple,
    mean: torch.Tensor,
    comps: torch.Tensor,
    lo: np.ndarray,
    hi: np.ndarray,
    out_size: tuple,
    out_dir: Path,
    num_to_save: int
) -> None:
    """
    Save each image's PCA projection as an individual file.
    
    Args:
        patches_list: List of corresponding patch features
        grid_hw: Grid height and width (h, w)
        mean: PCA mean vector
        comps: PCA components
        lo, hi: Percentile bounds for scaling
        out_size: Output image size (H, W)
        out_dir: Output directory
        num_to_save: Number of images to save
    """
    ensure_dir(out_dir)
    
    print(f"Saving {num_to_save} PCA projection image(s) to {out_dir}")
    
    for idx in range(num_to_save):
        # Get PCA projection
        rgb_pca = project_rgb(
            patches_list[idx], grid_hw, mean, comps, lo, hi, out_size
        )
        
        # Save just the PCA projection
        fig, ax = plt.subplots(1, 1, figsize=(3.2, 3.2))
        ax.imshow(rgb_pca)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"PCA projection {idx+1}", fontsize=9)
        fig.tight_layout()
        
        save_path = out_dir / f"pca_projection_{idx+1:04d}.png"
        fig.savefig(save_path, dpi=170, bbox_inches='tight')
        plt.close(fig)
    
    print(f"Saved {num_to_save} PCA projection images to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Save individual population-PCA projections as separate images"
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", required=True,
                    choices=["supervised", "lejepa", "vit_supervised", "vit_lejepa"])
    ap.add_argument("--layer", required=True, help="e.g. blocks.3 (ViT) or backbone.layer3 (ResNet)")
    ap.add_argument("--num-images", type=int, default=12,
                    help="Number of images to extract and save")
    ap.add_argument("--fit-batches", type=int, default=8,
                    help="Batches pooled to fit the PCA basis")
    ap.add_argument("--checkpoint", default=None,
                    help="Path to the probe checkpoint (overrides the config's checkpoint_path)")
    ap.add_argument("--out-dir", required=True,
                    help="Directory to save individual PCA projection images")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg["project"]["seed"]))
    device = get_device()

    model = load_classifier_from_checkpoint(
        cfg, mode=args.mode, device=device, requires_grad=False, checkpoint_path=args.checkpoint
    )
    loaders = build_loaders(cfg, self_supervised=False)
    out_size = (int(cfg["model"].get("image_size", 32)),) * 2

    # Number of images to save equals num-images
    num_to_save = args.num_images
    print(f"Will extract and save {num_to_save} PCA projection image(s)")

    # 1) Pool patches across several batches to fit the shared basis
    pooled, show_patches = [], []
    for bi, (images, _labels) in enumerate(loaders.test):
        if bi >= args.fit_batches:
            break
        images = images.to(device)
        patches, grid_hw = extract_patches(model, images, args.layer, device)
        pooled.append(patches)
        # Collect patches for as many images as we need
        if len(show_patches) < num_to_save:
            for i in range(images.shape[0]):
                if len(show_patches) >= num_to_save:
                    break
                show_patches.append(patches[i].detach())

    # Ensure we have enough patches
    if len(show_patches) < num_to_save:
        print(f"Warning: Only {len(show_patches)} images available, saving {len(show_patches)} instead of {num_to_save}")
        num_to_save = len(show_patches)

    mean, comps = fit_population_pca(pooled)

    # 2) Global per-channel percentile range for consistent coloring across all images.
    all_proj = torch.cat([((p.float() - mean) @ comps).reshape(-1, 3) for p in pooled], dim=0).cpu().numpy()
    lo = np.percentile(all_proj, 2, axis=0)
    hi = np.percentile(all_proj, 98, axis=0)

    # 3) Save individual PCA projections (no originals)
    out_dir = Path(args.out_dir)
    save_individual_pca_images(
        patches_list=show_patches,
        grid_hw=grid_hw,
        mean=mean,
        comps=comps,
        lo=lo,
        hi=hi,
        out_size=out_size,
        out_dir=out_dir,
        num_to_save=num_to_save
    )


if __name__ == "__main__":
    main()