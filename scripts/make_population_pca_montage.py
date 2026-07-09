from __future__ import annotations

"""
Paper-style population-PCA RGB montage (cf. LeJEPA Fig. 1, bottom-left).

Unlike the per-image PCA used for the LSAS metric, this fits ONE PCA basis over
patch features pooled across many images, then maps each image's top-3 PCA
projections to R/G/B. Because the basis (and its sign) is shared, the same color
means the same semantic direction across every image — that's what produces the
"different photos, consistent part colors" montage.

Works for ViT (patch tokens at a block) and ResNet (a conv feature map).

Example:
  python scripts/make_population_pca_montage.py \
    --config configs/cifar10_vit_lejepa_v4_short_lr1e4_sig005.yaml \
    --mode vit_lejepa --layer blocks.3 --num-images 12 \
    --out experiments/experiment-c10-vit-lejepa-v4-short-lr1e4-sig005/figures/population_pca_blocks3.png
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


def denorm(img: torch.Tensor, dataset: str) -> np.ndarray:
    """Best-effort un-normalize a [3,H,W] tensor to a displayable [H,W,3]."""
    if str(dataset).upper().startswith("CIFAR"):
        mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
        std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
        out = (img.cpu() * std + mean).clamp(0, 1)
    else:  # unknown normalization -> just min-max for display
        out = img.cpu()
        out = (out - out.amin()) / (out.amax() - out.amin() + 1e-8)
    return out.permute(1, 2, 0).numpy()


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", required=True,
                    choices=["supervised", "lejepa", "vit_supervised", "vit_lejepa"])
    ap.add_argument("--layer", required=True, help="e.g. blocks.3 (ViT) or backbone.layer3 (ResNet)")
    ap.add_argument("--num-images", type=int, default=12, help="images shown in the montage")
    ap.add_argument("--fit-batches", type=int, default=8, help="batches pooled to fit the PCA basis")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg["project"]["seed"]))
    device = get_device()
    dataset = cfg.get("data", {}).get("dataset", "CIFAR10")

    model = load_classifier_from_checkpoint(cfg, mode=args.mode, device=device, requires_grad=False)
    loaders = build_loaders(cfg, self_supervised=False)
    out_size = (int(cfg["model"].get("image_size", 32)),) * 2

    # 1) pool patches across several batches to fit the shared basis
    pooled, show_images, show_patches = [], [], []
    for bi, (images, _labels) in enumerate(loaders.test):
        if bi >= args.fit_batches:
            break
        images = images.to(device)
        patches, grid_hw = extract_patches(model, images, args.layer, device)
        pooled.append(patches)
        if len(show_images) < args.num_images:
            for i in range(images.shape[0]):
                if len(show_images) >= args.num_images:
                    break
                show_images.append(images[i].detach())
                show_patches.append(patches[i].detach())

    mean, comps = fit_population_pca(pooled)

    # 2) global per-channel percentile range for consistent coloring across all images
    all_proj = torch.cat([(p.float() - mean) @ comps for p in pooled], dim=0).cpu().numpy()
    lo = np.percentile(all_proj, 2, axis=0)
    hi = np.percentile(all_proj, 98, axis=0)

    # 3) render montage: row 0 = original, row 1 = population-PCA RGB
    n = len(show_images)
    fig, axes = plt.subplots(2, n, figsize=(1.5 * n, 3.2), squeeze=False)
    for j in range(n):
        axes[0][j].imshow(denorm(show_images[j], dataset))
        rgb = project_rgb(show_patches[j], grid_hw, mean, comps, lo, hi, out_size)
        axes[1][j].imshow(rgb)
        for r in (0, 1):
            axes[r][j].set_xticks([]); axes[r][j].set_yticks([])
    axes[0][0].set_ylabel("image", fontsize=10)
    axes[1][0].set_ylabel("population PCA\n(top-3 → RGB)", fontsize=9)
    fig.suptitle(f"{args.mode} · {args.layer} · shared PCA basis (same color = same direction across images)",
                 fontsize=10)
    fig.tight_layout()
    ensure_dir(Path(args.out).parent)
    fig.savefig(args.out, dpi=170)
    plt.close(fig)
    print(f"Saved population-PCA montage to {args.out}  ({n} images, layer={args.layer})")


if __name__ == "__main__":
    main()
