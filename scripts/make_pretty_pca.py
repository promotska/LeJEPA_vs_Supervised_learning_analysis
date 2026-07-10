from __future__ import annotations

"""
Pretty single-image PCA through a trained model: RGB (top-3 components) and
grayscale/heatmap (top component), rendered at the photo's native resolution.

Note on resolution: the PCA's real spatial detail is the model's feature grid
(e.g. ResNet layer3 @224 -> 14x14). We render it on the full-res photo canvas
via bicubic upsampling so it looks smooth, but detail is grid-limited. To get a
genuinely finer PCA grid on a ResNet (fully convolutional), feed a bigger input
with --input-size (e.g. 512 -> layer3 ~32x32). ViTs have a fixed grid, so
--input-size is ignored for them.

Outputs per image (in --out-dir/<name>/):
  <name>_image.png        your photo at native resolution
  <name>_pca_rgb.png      top-3 PCA components -> RGB
  <name>_pca_gray.png     top component -> grayscale
  <name>_pca_heat.png     top component -> colormap (--cmap)
  <name>_overlay_rgb.png  RGB PCA over the photo
  <name>_overlay_heat.png heatmap over the photo

Example (ResNet, finer grid via larger input):
  python scripts/make_pretty_pca.py \
    --config configs/imagenet100_resnet18_lejepa_glsas.yaml --mode lejepa \
    --layer backbone.layer3 --input-size 512 --cmap magma \
    --checkpoint /path/imagenet100_resnet18_lejepa_probe_best.pt \
    --image my_photo.jpg --out-dir figures/pretty
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
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.networks.factory import load_classifier_from_checkpoint
from src.utils import ensure_dir, get_device, get_module_by_name, load_yaml, set_seed
from src.xai.gradcam import _LayerHook

try:
    from src.globals import CIFAR10_MEAN, CIFAR10_STD
except Exception:
    CIFAR10_MEAN, CIFAR10_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


def norm_stats(dataset):
    if str(dataset).upper().startswith("CIFAR"):
        return np.array(CIFAR10_MEAN), np.array(CIFAR10_STD)
    return np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def preprocess(path, input_size, dataset, device):
    mean, std = norm_stats(dataset)
    pil = Image.open(path).convert("RGB")
    native = np.asarray(pil, dtype=np.float32) / 255.0            # full-res original for display
    small = pil.resize((input_size, input_size), Image.BICUBIC)
    arr = (np.asarray(small, dtype=np.float32) / 255.0 - mean) / std
    t = torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    return t, native


@torch.no_grad()
def extract_patches(model, images, layer):
    bb = model.backbone
    if hasattr(bb, "grid_size"):  # ViT: fixed grid
        num_prefix = 1 + int(getattr(bb, "num_registers", 0))
        _logits, cap = model.forward_with_intermediates(images, capture_layers=[layer])
        tok = cap[layer][:, num_prefix:, :]
        g = int(bb.grid_size)
        return tok[0], (g, g), True
    hook = _LayerHook(get_module_by_name(model, layer))          # ResNet/conv: hook the layer
    model(images)
    act = hook.data.activation
    hook.close()
    b, c, h, w = act.shape
    return act.permute(0, 2, 3, 1).reshape(b, h * w, c)[0], (h, w), False


@torch.no_grad()
def pca_components(patches, k=3):
    x = patches.float()
    x = x - x.mean(0)
    cov = x.transpose(0, 1) @ x
    ev, evec = torch.linalg.eigh(cov)
    comps = evec[:, ev.argsort(descending=True)[:k]]
    return (x @ comps).cpu().numpy()                              # [N, k]


def pct_norm(a, lo=2, hi=98):
    a = np.asarray(a, dtype=np.float32)
    l, h = np.percentile(a, lo), np.percentile(a, hi)
    return np.clip((a - l) / (h - l + 1e-8), 0.0, 1.0)


def upsample(arr, size):
    t = torch.from_numpy(np.ascontiguousarray(arr)).float()
    single = t.ndim == 2
    t = (t[None, None] if single else t.permute(2, 0, 1)[None])
    up = F.interpolate(t, size=size, mode="bicubic", align_corners=False)[0]
    return up[0].numpy() if single else up.permute(1, 2, 0).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", required=True, choices=["supervised", "lejepa", "vit_supervised", "vit_lejepa"])
    ap.add_argument("--layer", required=True, help="e.g. backbone.layer3 (ResNet) or blocks.3 (ViT)")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--image", nargs="+", required=True)
    ap.add_argument("--input-size", type=int, default=None,
                    help="model input size; larger => finer PCA grid (ResNet only; ignored for ViT)")
    ap.add_argument("--cmap", default="magma", help="colormap for the heatmap PCA (e.g. magma, viridis, inferno)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg["project"]["seed"]))
    device = get_device()
    dataset = cfg.get("data", {}).get("dataset", "CIFAR10")
    cfg_size = int(cfg["model"].get("image_size", 224))
    input_size = args.input_size or cfg_size

    model = load_classifier_from_checkpoint(
        cfg, mode=args.mode, device=device, requires_grad=False, checkpoint_path=args.checkpoint)
    out_dir = ensure_dir(args.out_dir)
    cmap = plt.get_cmap(args.cmap)

    for path in args.image:
        stem = Path(path).stem
        t, native = preprocess(path, input_size, dataset, device)
        patches, (gh, gw), is_vit = extract_patches(model, t, args.layer)
        if is_vit and input_size != cfg_size:
            print(f"  note: ViT has a fixed grid; --input-size ignored (used {cfg_size}).")
        proj = pca_components(patches, k=3)                       # [N,3]
        H, W = native.shape[:2]

        gray = pct_norm(upsample(pct_norm(proj[:, 0].reshape(gh, gw)), (H, W)))
        rgb = np.clip(upsample(np.stack([pct_norm(proj[:, c].reshape(gh, gw)) for c in range(3)], -1), (H, W)), 0, 1)
        heat = cmap(gray)[..., :3]

        d = ensure_dir(Path(out_dir) / stem)
        plt.imsave(d / f"{stem}_image.png", np.clip(native, 0, 1))
        plt.imsave(d / f"{stem}_pca_gray.png", gray, cmap="gray")
        plt.imsave(d / f"{stem}_pca_heat.png", gray, cmap=args.cmap)
        plt.imsave(d / f"{stem}_pca_rgb.png", rgb)
        plt.imsave(d / f"{stem}_overlay_heat.png", np.clip(0.55 * native + 0.45 * heat, 0, 1))
        plt.imsave(d / f"{stem}_overlay_rgb.png", np.clip(0.55 * native + 0.45 * rgb, 0, 1))
        print(f"{stem}: feature grid {gh}x{gw} -> rendered {W}x{H}px  (saved to {d})")


if __name__ == "__main__":
    main()
