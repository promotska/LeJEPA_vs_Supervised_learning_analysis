from __future__ import annotations

import argparse
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataloaders import build_loaders
from src.data.transforms import default_normalization
from src.evaluation.sas import lsas
from src.experiment import prepare_experiment, update_manifest
from src.features.pca_masks import pca_heatmaps_from_tokens
from src.utils import ensure_dir, get_device, load_yaml, set_seed
from src.visualization.plot_layer_curves import save_layer_curve


def _as_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"true", "1", "yes", "y"}:
            return True
        if value in {"false", "0", "no", "n"}:
            return False
    return bool(value)


def _append_stem_suffix(path: str | Path, suffix: str) -> str:
    path = Path(path)
    if path.stem.endswith(f"_{suffix}"):
        return str(path)
    return str(path.with_name(f"{path.stem}_{suffix}{path.suffix}"))


def _normalize_maps(maps: torch.Tensor) -> torch.Tensor:
    maps = maps.detach().float()
    b = maps.shape[0]
    flat = maps.reshape(b, -1)
    min_v = flat.min(dim=1, keepdim=True).values
    max_v = flat.max(dim=1, keepdim=True).values
    flat = (flat - min_v) / (max_v - min_v + 1e-8)
    return flat.reshape_as(maps)


def _strip_state_dict_prefixes(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        key = str(k)
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        out[key] = v
    return out


def _find_checkpoint_file(local_dir: str | Path) -> Path:
    local_dir = Path(local_dir)
    candidates = [
        local_dir / "model.safetensors",
        local_dir / "pytorch_model.bin",
        local_dir / "open_clip_pytorch_model.bin",
    ]
    for path in candidates:
        if path.exists():
            return path
    safetensors = sorted(local_dir.glob("*.safetensors"))
    if safetensors:
        return safetensors[0]
    bins = sorted(local_dir.glob("*.bin"))
    if bins:
        return bins[0]
    raise FileNotFoundError(
        f"No model.safetensors / pytorch_model.bin / *.bin found in local_dir={local_dir}"
    )


def _load_local_state_dict(local_dir: str | Path) -> dict[str, torch.Tensor]:
    ckpt_path = _find_checkpoint_file(local_dir)
    print(f"Loading local supervised ViT checkpoint: {ckpt_path}")
    if ckpt_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "safetensors is required to load model.safetensors. Install it with: pip install safetensors"
            ) from exc
        state = load_file(str(ckpt_path), device="cpu")
    else:
        loaded = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(loaded, dict) and "state_dict" in loaded:
            state = loaded["state_dict"]
        elif isinstance(loaded, dict) and "model" in loaded and isinstance(loaded["model"], dict):
            state = loaded["model"]
        elif isinstance(loaded, dict):
            state = loaded
        else:
            raise TypeError(f"Unsupported checkpoint object type: {type(loaded)}")
    return _strip_state_dict_prefixes(state)


def build_timm_supervised_vit(cfg: dict[str, Any], device: torch.device):
    try:
        import timm
    except ImportError as exc:
        raise ImportError("timm is required. Install it with: pip install timm") from exc

    model_cfg = cfg["model"]
    timm_model_name = str(model_cfg.get("timm_model_name", "vit_base_patch16_224.augreg_in1k"))
    local_dir = model_cfg.get("local_dir") or model_cfg.get("hf_model_id")
    num_classes = int(model_cfg.get("imagenet_num_classes", 1000))

    print(f"Building timm model: {timm_model_name}")
    try:
        model = timm.create_model(timm_model_name, pretrained=False, num_classes=num_classes)
    except Exception as exc:
        # Fallback for older timm versions that do not recognize the HF tag suffix.
        fallback = timm_model_name.split(".")[0]
        print(f"WARNING: timm.create_model({timm_model_name!r}) failed: {exc}")
        print(f"Trying fallback architecture name: {fallback!r}")
        model = timm.create_model(fallback, pretrained=False, num_classes=num_classes)

    if local_dir is not None and Path(str(local_dir)).is_dir():
        state = _load_local_state_dict(local_dir)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded local state dict with strict=False")
        print(f"  missing keys: {len(missing)}")
        if missing[:20]:
            print("  first missing:", missing[:20])
        print(f"  unexpected keys: {len(unexpected)}")
        if unexpected[:20]:
            print("  first unexpected:", unexpected[:20])
    elif _as_bool(model_cfg.get("allow_hf_hub_download", False), default=False):
        # This branch needs internet. Do not use it on CINECA offline nodes.
        print("Loading from HF Hub via timm. This requires internet.")
        model = timm.create_model(f"hf_hub:timm/{timm_model_name}", pretrained=True)
    else:
        raise FileNotFoundError(
            "For offline CINECA inference, set model.local_dir or model.hf_model_id "
            "to a local folder containing model.safetensors / pytorch_model.bin."
        )

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _qkv_to_attention(qkv: torch.Tensor, num_heads: int, scale: float | None = None) -> torch.Tensor:
    """Convert timm qkv projection output [B,T,3D] into attention [B,heads,T,T]."""
    if qkv.ndim != 3:
        raise ValueError(f"Expected qkv [B,T,3D], got {tuple(qkv.shape)}")
    b, t, three_d = qkv.shape
    if three_d % 3 != 0:
        raise ValueError(f"qkv last dim should be 3D, got {three_d}")
    d = three_d // 3
    if d % num_heads != 0:
        raise ValueError(f"hidden dim {d} not divisible by num_heads={num_heads}")
    head_dim = d // num_heads
    qkv = qkv.reshape(b, t, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k = qkv[0], qkv[1]
    scale = float(scale) if scale is not None else head_dim ** -0.5
    attn = (q @ k.transpose(-2, -1)) * scale
    return attn.softmax(dim=-1)


def _forward_tokens_attention_logits(model: torch.nn.Module, images: torch.Tensor):
    if not hasattr(model, "blocks") or len(model.blocks) == 0:
        raise RuntimeError("Expected a timm VisionTransformer with model.blocks.")

    last_attn_module = model.blocks[-1].attn
    if not hasattr(last_attn_module, "qkv"):
        raise RuntimeError("Expected last attention module to have qkv projection.")

    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        captured["qkv"] = output.detach()

    handle = last_attn_module.qkv.register_forward_hook(hook)
    try:
        tokens = model.forward_features(images)
    finally:
        handle.remove()

    if tokens.ndim != 3:
        raise RuntimeError(
            f"Expected timm ViT forward_features to return [B,T,D], got {tuple(tokens.shape)}. "
            "If your timm version pools in forward_features, use another model variant or update timm."
        )

    if "qkv" not in captured:
        raise RuntimeError("Failed to capture qkv activation from final ViT block.")

    num_heads = int(getattr(last_attn_module, "num_heads"))
    scale = getattr(last_attn_module, "scale", None)
    last_self_attention = _qkv_to_attention(captured["qkv"].float(), num_heads=num_heads, scale=scale)

    logits = None
    try:
        logits = model.forward_head(tokens)
    except Exception:
        pass

    return tokens.float(), last_self_attention.float(), logits


def _extract_patch_tokens(tokens: torch.Tensor, model: torch.nn.Module, expected_patches: int | None = None) -> torch.Tensor:
    prefix_tokens = int(getattr(model, "num_prefix_tokens", 1))
    patch_tokens = tokens[:, prefix_tokens:, :]
    if expected_patches is not None and patch_tokens.shape[1] >= expected_patches:
        patch_tokens = patch_tokens[:, :expected_patches, :]
    return patch_tokens


def _cls_attention_to_maps(
    last_self_attention: torch.Tensor,
    model: torch.nn.Module,
    grid_size: int,
    output_size: tuple[int, int],
    head_fusion: str = "mean",
) -> torch.Tensor:
    attn = last_self_attention.detach().float()
    if attn.ndim == 4:
        if head_fusion == "mean":
            attn = attn.mean(dim=1)
        elif head_fusion == "max":
            attn = attn.max(dim=1).values
        elif head_fusion == "min":
            attn = attn.min(dim=1).values
        else:
            raise ValueError(f"Unknown attention_head_fusion={head_fusion!r}")
    elif attn.ndim != 3:
        raise ValueError(f"Expected attention [B,T,T] or [B,heads,T,T], got {tuple(attn.shape)}")

    prefix_tokens = int(getattr(model, "num_prefix_tokens", 1))
    expected = grid_size * grid_size
    cls_to_patch = attn[:, 0, prefix_tokens:prefix_tokens + expected]
    if cls_to_patch.shape[1] != expected:
        raise ValueError(
            f"CLS attention patch count={cls_to_patch.shape[1]} does not match grid_size^2={expected}. "
            f"attention shape={tuple(attn.shape)}, prefix_tokens={prefix_tokens}"
        )

    maps = cls_to_patch.reshape(attn.shape[0], 1, grid_size, grid_size)
    maps = F.interpolate(maps, size=output_size, mode="bilinear", align_corners=False)
    return _normalize_maps(maps)[:, 0]


def _denormalize(image: torch.Tensor, mean: tuple[float, float, float], std: tuple[float, float, float]) -> np.ndarray:
    mean_t = torch.tensor(mean, dtype=image.dtype, device=image.device).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=image.dtype, device=image.device).view(3, 1, 1)
    img = image * std_t + mean_t
    return img.clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy()


def _save_map_grid(
    image: torch.Tensor,
    pca_map: np.ndarray,
    xai_map: np.ndarray,
    output_path: str | Path,
    title: str,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> None:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    img = _denormalize(image, mean=mean, std=std)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    axes[0].imshow(img)
    axes[0].set_title("image")
    axes[1].imshow(pca_map, cmap="magma")
    axes[1].set_title("PCA mask")
    axes[2].imshow(xai_map, cmap="magma")
    axes[2].set_title("CLS attention")
    axes[3].imshow(img)
    axes[3].imshow(xai_map, cmap="magma", alpha=0.45)
    axes[3].set_title("overlay")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def evaluate_timm_vit_inference_alignment(config_path: str) -> None:
    cfg = load_yaml(config_path)
    architecture = str(cfg.get("model", {}).get("architecture", "")).lower()
    if "timm" not in architecture and "supervised" not in architecture and "vit" not in architecture:
        raise ValueError(
            "This script is intended for supervised timm ViT checkpoints. "
            f"Got model.architecture={architecture!r}."
        )

    cfg.setdefault("evaluation", {})
    cfg["evaluation"]["vit_xai_method"] = "last_attention"
    cfg["evaluation"].pop("checkpoint_path", None)
    cfg["evaluation"]["output_csv"] = _append_stem_suffix(
        cfg["evaluation"].get("output_csv", "outputs/metrics/timm_vit_inference_lsas.csv"),
        "inference_last_attention",
    )
    cfg["evaluation"]["figure_dir"] = str(
        Path(cfg["evaluation"].get("figure_dir", "outputs/figures/timm_vit_inference"))
        / "inference_last_attention"
    )

    exp = prepare_experiment(
        cfg=cfg,
        config_path=config_path,
        run_kind="evaluate_timm_vit_inference_alignment",
        mode="public_supervised_timm_vit_inference",
    )

    try:
        set_seed(int(cfg.get("project", {}).get("seed", 42)))
        torch.backends.cudnn.benchmark = True
        device = get_device()
        print(f"Using device: {device}")

        eval_cfg = cfg["evaluation"]
        data_cfg = cfg["data"]
        model_cfg = cfg["model"]

        eval_batch_size = int(eval_cfg.get("batch_size", 32))
        max_samples = int(eval_cfg.get("max_samples", 1000))
        num_visualizations = int(eval_cfg.get("num_visualizations", 4))
        save_visualizations = _as_bool(eval_cfg.get("save_visualizations", True), default=True)
        head_fusion = str(eval_cfg.get("attention_head_fusion", "mean"))
        image_size = int(model_cfg.get("image_size", data_cfg.get("image_size", 224)))
        patch_size = int(model_cfg.get("patch_size", 16))
        expected_grid = image_size // patch_size
        expected_patches = expected_grid * expected_grid

        dataset_name = str(data_cfg.get("dataset", "ImageFolder"))
        default_mean, default_std = default_normalization(dataset_name)
        mean = tuple(float(v) for v in data_cfg.get("mean", default_mean))
        std = tuple(float(v) for v in data_cfg.get("std", default_std))

        print("Public supervised timm ViT inference-only LSAS settings:")
        print(f"  config: {config_path}")
        print(f"  timm_model_name: {model_cfg.get('timm_model_name')}")
        print(f"  local_dir/hf_model_id: {model_cfg.get('local_dir') or model_cfg.get('hf_model_id')}")
        print("  classifier training: none; using public supervised pretrained model directly")
        print("  xai_method: final CLS-to-patch attention reconstructed from QKV")
        print(f"  max_samples: {max_samples}")
        print(f"  batch_size: {eval_batch_size}")
        print(f"  output_csv: {eval_cfg['output_csv']}")

        cfg.setdefault("data", {})
        cfg["data"]["batch_size"] = eval_batch_size
        loaders = build_loaders(cfg, self_supervised=False)
        model = build_timm_supervised_vit(cfg, device=device)

        output_csv = Path(eval_cfg["output_csv"])
        figure_dir = ensure_dir(eval_cfg["figure_dir"])

        rows: list[dict[str, Any]] = []
        processed = 0
        started_at = time.time()

        progress = tqdm(loaders.test, desc="supervised timm vit inference LSAS", dynamic_ncols=True)
        for images, labels in progress:
            if processed >= max_samples:
                break
            remaining = max_samples - processed
            if images.shape[0] > remaining:
                images = images[:remaining]
                labels = labels[:remaining]

            images = images.to(device, non_blocking=True)
            labels_cpu = labels.detach().cpu()
            current_batch_size = images.shape[0]

            with torch.no_grad():
                tokens, last_self_attention, logits = _forward_tokens_attention_logits(model, images)
                patch_tokens = _extract_patch_tokens(tokens, model, expected_patches=expected_patches)
                n_patches = patch_tokens.shape[1]
                grid_size = int(math.sqrt(n_patches))
                if grid_size * grid_size != n_patches:
                    raise RuntimeError(f"Patch token count is not square: {n_patches}")
                xai_maps_t = _cls_attention_to_maps(
                    last_self_attention=last_self_attention,
                    model=model,
                    grid_size=grid_size,
                    output_size=tuple(images.shape[-2:]),
                    head_fusion=head_fusion,
                )
                pred_imagenet1k = logits.argmax(dim=1).detach().cpu() if logits is not None else None

            pca_maps = pca_heatmaps_from_tokens(
                tokens=patch_tokens,
                grid_size=grid_size,
                output_size=tuple(images.shape[-2:]),
                component=0,
                use_abs=True,
            )
            xai_maps = xai_maps_t.cpu().numpy()

            for i in range(current_batch_size):
                global_idx = processed + i
                metrics = lsas(pca_maps[i], xai_maps[i])
                rows.append(
                    {
                        "sample_idx": global_idx,
                        "mode": "public_supervised_timm_vit_inference",
                        "architecture": model_cfg.get("architecture"),
                        "model_id": model_cfg.get("hf_model_id"),
                        "timm_model_name": model_cfg.get("timm_model_name"),
                        "xai_method": "last_attention_qkv_reconstructed",
                        "layer": "patch_tokens_final",
                        "true_label": int(labels_cpu[i].item()),
                        "pred_imagenet1k_label": int(pred_imagenet1k[i].item()) if pred_imagenet1k is not None else -1,
                        "has_classifier": True,
                        "note": "pred_imagenet1k_label is the original ImageNet-1K head index; it is not remapped to ImageNet-100 folders.",
                        **metrics,
                    }
                )

                if save_visualizations and global_idx < num_visualizations:
                    fig_path = figure_dir / f"sample_{global_idx:03d}_patch_tokens_final.png"
                    _save_map_grid(
                        image=images[i].detach().cpu(),
                        pca_map=pca_maps[i],
                        xai_map=xai_maps[i],
                        output_path=fig_path,
                        title="Supervised timm ViT-B/16 | final patch tokens | inference-only",
                        mean=mean,
                        std=std,
                    )

            processed += current_batch_size
            elapsed = time.time() - started_at
            progress.set_postfix({"images": processed, "rows": len(rows), "img/s": f"{processed / max(elapsed, 1e-8):.2f}"})

        ensure_dir(output_csv.parent)
        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False)

        curve_path = output_csv.parent / f"{output_csv.stem}_curve.png"
        save_layer_curve(
            output_csv,
            curve_path,
            title="Supervised timm ViT-B/16 inference LSAS: PCA vs final attention",
        )

        elapsed = time.time() - started_at
        print()
        print("Supervised timm ViT inference-only LSAS completed.")
        print(f"Saved CSV: {output_csv}")
        print(f"Saved curve: {curve_path}")
        print(f"Processed images: {processed}")
        print(f"Rows: {len(rows)}")
        print(f"Elapsed: {elapsed / 60:.2f} min")
        print()
        print("Mean metrics per layer:")
        print(df.groupby("layer")[["lsas", "corr_pca_xai", "soft_iou_pca_xai"]].mean())

        update_manifest(
            exp=exp,
            cfg=cfg,
            status="completed",
            extra={
                "processed_images": processed,
                "rows": len(rows),
                "has_classifier": True,
                "xai_method": "last_attention_qkv_reconstructed",
                "output_csv": str(output_csv),
                "curve_path": str(curve_path),
                "elapsed_minutes": elapsed / 60,
            },
        )
    except Exception as exc:
        update_manifest(
            exp=exp,
            cfg=cfg,
            status="failed",
            extra={"error": repr(exc), "traceback": traceback.format_exc()},
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    evaluate_timm_vit_inference_alignment(args.config)


if __name__ == "__main__":
    main()
