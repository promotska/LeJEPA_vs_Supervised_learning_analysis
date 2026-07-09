from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataloaders import build_loaders
from src.evaluation.sas import lsas
from src.experiment import prepare_experiment, update_manifest
from src.features.pca_masks import pca_heatmaps_from_tokens
from src.networks.hf_vit import HFViTBackbone
from src.utils import ensure_dir, get_device, load_yaml, set_seed
from src.visualization.plot_layer_curves import save_layer_curve
from src.visualization.plot_maps import save_map_grid


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
    flat = maps.view(b, -1)
    min_v = flat.min(dim=1, keepdim=True).values
    max_v = flat.max(dim=1, keepdim=True).values
    flat = (flat - min_v) / (max_v - min_v + 1e-8)
    return flat.view_as(maps)


def _cls_attention_to_maps(
    last_self_attention: torch.Tensor,
    grid_size: int,
    output_size: tuple[int, int],
    head_fusion: str = "mean",
) -> torch.Tensor:
    """Return normalized final CLS-to-patch attention maps as tensor [B,H,W]."""
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
        raise ValueError(
            f"Expected attention [B,T,T] or [B,heads,T,T], got {tuple(attn.shape)}"
        )

    cls_to_patch = attn[:, 0, 1:]
    b, n = cls_to_patch.shape
    expected = grid_size * grid_size
    if n != expected:
        raise ValueError(f"Attention patch count N={n} does not match grid_size^2={expected}")

    maps = cls_to_patch.reshape(b, 1, grid_size, grid_size)
    maps = F.interpolate(maps, size=output_size, mode="bilinear", align_corners=False)
    maps = _normalize_maps(maps)
    return maps[:, 0]


def build_public_hf_backbone(cfg: dict[str, Any], device: torch.device) -> HFViTBackbone:
    model_cfg = cfg["model"]
    backbone = HFViTBackbone(
        model_id=str(model_cfg["hf_model_id"]),
        revision=model_cfg.get("hf_revision", None),
        image_size=int(model_cfg.get("image_size", 224)),
        patch_size=int(model_cfg.get("patch_size", 16)),
        embed_dim=model_cfg.get("embed_dim", None),
        local_files_only=bool(model_cfg.get("hf_local_files_only", True)),
        trust_remote_code=bool(model_cfg.get("hf_trust_remote_code", True)),
        torch_dtype=model_cfg.get("hf_torch_dtype", None),
        freeze=True,
    )
    return backbone.to(device).eval()


def evaluate_public_vit_inference_alignment(config_path: str) -> None:
    cfg = load_yaml(config_path)

    architecture = str(cfg.get("model", {}).get("architecture", "")).lower()
    if not (architecture.startswith("hf_") or "okai" in architecture or "public" in architecture):
        raise ValueError(
            "This script is intended for public Hugging Face ViT checkpoints only. "
            f"Got model.architecture={architecture!r}."
        )

    cfg.setdefault("evaluation", {})
    cfg["evaluation"]["vit_xai_method"] = "last_attention"
    cfg["evaluation"].pop("checkpoint_path", None)
    cfg["evaluation"]["output_csv"] = _append_stem_suffix(
        cfg["evaluation"].get("output_csv", "outputs/metrics/public_vit_inference_lsas.csv"),
        "inference_last_attention",
    )
    cfg["evaluation"]["figure_dir"] = str(
        Path(cfg["evaluation"].get("figure_dir", "outputs/figures/public_vit_inference"))
        / "inference_last_attention"
    )

    exp = prepare_experiment(
        cfg=cfg,
        config_path=config_path,
        run_kind="evaluate_public_vit_inference_alignment",
        mode="public_vit_lejepa_inference",
    )

    try:
        set_seed(int(cfg["project"].get("seed", 42)))
        torch.backends.cudnn.benchmark = True
        device = get_device()
        print(f"Using device: {device}")

        eval_cfg = cfg["evaluation"]
        eval_batch_size = int(eval_cfg.get("batch_size", 32))
        max_samples = int(eval_cfg.get("max_samples", 1000))
        layers = list(eval_cfg.get("layers", ["patch_latent"]))
        num_visualizations = int(eval_cfg.get("num_visualizations", 4))
        save_visualizations = _as_bool(eval_cfg.get("save_visualizations", True), default=True)
        head_fusion = str(eval_cfg.get("attention_head_fusion", "mean"))

        invalid_layers = [x for x in layers if str(x).lower() not in {"patch_latent", "last", "final", "last_self_attention"}]
        if invalid_layers:
            raise ValueError(
                "Public OK-AI HF checkpoints expose final patch_latent only. "
                f"Use evaluation.layers: ['patch_latent']. Invalid layers: {invalid_layers}"
            )

        print("Public ViT inference-only LSAS settings:")
        print(f"  config: {config_path}")
        print(f"  model: {cfg['model']['hf_model_id']}")
        print("  checkpoint: none; using frozen public pretrained backbone directly")
        print("  xai_method: last_attention")
        print(f"  layers: {layers}")
        print(f"  max_samples: {max_samples}")
        print(f"  batch_size: {eval_batch_size}")
        print(f"  output_csv: {eval_cfg['output_csv']}")

        # Use evaluation batch size without changing the config file permanently.
        cfg.setdefault("data", {})
        cfg["data"]["batch_size"] = eval_batch_size
        loaders = build_loaders(cfg, self_supervised=False)
        backbone = build_public_hf_backbone(cfg, device=device)
        grid_size = int(backbone.grid_size)

        output_csv = Path(eval_cfg["output_csv"])
        figure_dir = ensure_dir(eval_cfg["figure_dir"])

        rows: list[dict[str, Any]] = []
        processed = 0
        started_at = time.time()

        progress = tqdm(loaders.test, desc="public vit inference LSAS", dynamic_ncols=True)
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
                _, patch_tokens, last_self_attention = backbone.forward_patch_tokens_and_attention(images)
                xai_maps_t = _cls_attention_to_maps(
                    last_self_attention=last_self_attention,
                    grid_size=grid_size,
                    output_size=tuple(images.shape[-2:]),
                    head_fusion=head_fusion,
                )

            pca_maps = pca_heatmaps_from_tokens(
                tokens=patch_tokens,
                grid_size=grid_size,
                output_size=tuple(images.shape[-2:]),
                component=0,
                use_abs=True,
            )
            xai_maps = xai_maps_t.cpu().numpy()

            for layer_name in layers:
                for i in range(current_batch_size):
                    global_idx = processed + i
                    metrics = lsas(pca_maps[i], xai_maps[i])
                    rows.append(
                        {
                            "sample_idx": global_idx,
                            "mode": "public_vit_lejepa_inference",
                            "architecture": cfg["model"].get("architecture"),
                            "model_id": cfg["model"].get("hf_model_id"),
                            "xai_method": "last_attention",
                            "layer": layer_name,
                            "true_label": int(labels_cpu[i].item()),
                            "pred_label": -1,
                            "target_label": -1,
                            "has_classifier": False,
                            **metrics,
                        }
                    )

                    if save_visualizations and global_idx < num_visualizations:
                        safe_layer = str(layer_name).replace(".", "_")
                        fig_path = figure_dir / f"sample_{global_idx:03d}_{safe_layer}.png"
                        save_map_grid(
                            image=images[i].detach().cpu(),
                            pca_map=pca_maps[i],
                            xai_map=xai_maps[i],
                            output_path=fig_path,
                            title=(
                                "Public OK-AI LeJEPA ViT-B/16 | "
                                f"{layer_name} | inference-only | no classifier"
                            ),
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
            title="Public OK-AI LeJEPA ViT-B/16 inference LSAS: PCA vs final attention",
        )

        elapsed = time.time() - started_at
        print()
        print("Public ViT inference-only LSAS completed.")
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
                "has_classifier": False,
                "xai_method": "last_attention",
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
    evaluate_public_vit_inference_alignment(args.config)


if __name__ == "__main__":
    main()
