from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataloaders import build_loaders
from src.evaluation.sas import lsas
from src.experiment import prepare_experiment, update_manifest
from src.features.pca_masks import pca_heatmaps_from_tokens
from src.networks.factory import load_classifier_from_checkpoint
from src.utils import ensure_dir, get_device, load_yaml, set_seed
from src.visualization.plot_layer_curves import save_layer_curve
from src.visualization.plot_maps import save_map_grid
from src.xai.attention_rollout import generate_vit_attention_rollout
from src.xai.factory import get_vit_xai_method, is_vit_xai_method_supported
from src.xai.vit_saliency import generate_vit_token_saliency


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


def _resolve_target_classes(gradcam_target: str, labels: torch.Tensor) -> torch.Tensor | None:
    gradcam_target = gradcam_target.lower().strip()
    if gradcam_target == "predicted":
        return None
    if gradcam_target in {"true", "gt", "label", "ground_truth", "ground-truth"}:
        return labels
    raise ValueError(f"Unknown gradcam_target={gradcam_target!r}. Use 'predicted' or 'true'.")


def _run_vit_xai(
    model: torch.nn.Module,
    images: torch.Tensor,
    layers: list[str],
    grid_size: int,
    xai_method: str,
    gradcam_target: str,
    target_classes: torch.Tensor | None,
    cfg: dict[str, Any],
):
    if xai_method == "token_gradient":
        with torch.enable_grad():
            return generate_vit_token_saliency(
                model=model,
                images=images,
                layers=layers,
                grid_size=grid_size,
                target_classes=target_classes,
                use_abs=True,
            )

    if xai_method == "attention_rollout":
        eval_cfg = cfg.get("evaluation", {})
        with torch.no_grad():
            return generate_vit_attention_rollout(
                model=model,
                images=images,
                layers=layers,
                grid_size=grid_size,
                discard_ratio=float(eval_cfg.get("attention_discard_ratio", 0.0)),
                head_fusion=str(eval_cfg.get("attention_head_fusion", "mean")),
            )

    raise ValueError(f"Unsupported ViT XAI method: {xai_method}")


def evaluate_vit_alignment(
    config_path: str,
    mode: str,
    gradcam_target_override: str | None = None,
    vit_xai_method_override: str | None = None,
) -> None:
    cfg = load_yaml(config_path)

    if gradcam_target_override is not None:
        cfg.setdefault("evaluation", {})
        cfg["evaluation"]["gradcam_target"] = gradcam_target_override
    if vit_xai_method_override is not None:
        cfg.setdefault("evaluation", {})
        cfg["evaluation"]["vit_xai_method"] = vit_xai_method_override

    gradcam_target = str(cfg["evaluation"].get("gradcam_target", "predicted")).lower().strip()
    xai_method = get_vit_xai_method(cfg)
    if not is_vit_xai_method_supported(xai_method):
        raise ValueError(f"Unsupported evaluation.vit_xai_method={xai_method!r}")

    cfg["evaluation"]["output_csv"] = _append_stem_suffix(
        cfg["evaluation"]["output_csv"],
        f"{xai_method}_{gradcam_target}",
    )
    cfg["evaluation"]["figure_dir"] = str(
        Path(cfg["evaluation"]["figure_dir"]) / f"{xai_method}_target_{gradcam_target}"
    )

    exp = prepare_experiment(
        cfg=cfg,
        config_path=config_path,
        run_kind=f"evaluate_vit_alignment_{mode}_{xai_method}_{gradcam_target}",
        mode=mode,
    )

    try:
        set_seed(int(cfg["project"]["seed"]))
        torch.backends.cudnn.benchmark = True
        device = get_device()
        print(f"Using device: {device}")

        eval_batch_size = int(cfg["evaluation"].get("batch_size", 32))
        max_samples = int(cfg["evaluation"].get("max_samples", 256))
        num_visualizations = int(cfg["evaluation"].get("num_visualizations", 4))
        save_visualizations = _as_bool(cfg["evaluation"].get("save_visualizations", True), default=True)
        layers = list(cfg["evaluation"]["layers"])

        print("ViT alignment evaluation settings:")
        print(f"  mode: {mode}")
        print(f"  xai_method: {xai_method}")
        print(f"  gradcam_target: {gradcam_target}")
        print(f"  checkpoint: {cfg['evaluation']['checkpoint_path']}")
        print(f"  layers: {layers}")
        print(f"  max_samples: {max_samples}")
        print(f"  output_csv: {cfg['evaluation']['output_csv']}")

        loaders = build_loaders(cfg, self_supervised=False)
        model = load_classifier_from_checkpoint(cfg, mode=mode, device=device, requires_grad=True)
        grid_size = int(model.backbone.grid_size)

        output_csv = Path(cfg["evaluation"]["output_csv"])
        figure_dir = ensure_dir(cfg["evaluation"]["figure_dir"])

        rows: list[dict] = []
        processed = 0
        correct = 0
        started_at = time.time()

        progress = tqdm(loaders.test, desc=f"evaluate {mode}/{xai_method}/{gradcam_target}", dynamic_ncols=True)
        if xai_method == "attention_rollout" and gradcam_target != "predicted":
            print("WARNING: Attention Rollout is not class-target-specific; true/predicted target affects labels only, not the rollout map.")

        for images, labels in progress:
            if processed >= max_samples:
                break

            remaining = max_samples - processed
            if images.shape[0] > remaining:
                images = images[:remaining]
                labels = labels[:remaining]

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            current_batch_size = images.shape[0]
            target_classes = _resolve_target_classes(gradcam_target=gradcam_target, labels=labels)

            xai_maps_by_layer, logits, tokens_by_layer = _run_vit_xai(
                model=model,
                images=images,
                layers=layers,
                grid_size=grid_size,
                xai_method=xai_method,
                gradcam_target=gradcam_target,
                target_classes=target_classes,
                cfg=cfg,
            )

            preds = logits.argmax(dim=1).detach().cpu()
            true_labels = labels.detach().cpu()
            target_labels = preds if target_classes is None else target_classes.detach().cpu()
            correct += int((preds == true_labels).sum().item())

            for layer_name in layers:
                tokens = tokens_by_layer[layer_name]
                pca_maps = pca_heatmaps_from_tokens(
                    tokens=tokens,
                    grid_size=grid_size,
                    output_size=tuple(images.shape[-2:]),
                    component=0,
                    use_abs=True,
                )
                xai_maps = xai_maps_by_layer[layer_name]

                for i in range(current_batch_size):
                    global_idx = processed + i
                    metrics = lsas(pca_maps[i], xai_maps[i])
                    rows.append(
                        {
                            "sample_idx": global_idx,
                            "mode": mode,
                            "architecture": cfg["model"].get("architecture"),
                            "xai_method": xai_method,
                            "gradcam_target": gradcam_target,
                            "layer": layer_name,
                            "true_label": int(true_labels[i].item()),
                            "pred_label": int(preds[i].item()),
                            "target_label": int(target_labels[i].item()),
                            **metrics,
                        }
                    )

                    if save_visualizations and global_idx < num_visualizations:
                        safe_layer = layer_name.replace(".", "_")
                        fig_path = figure_dir / f"sample_{global_idx:03d}_{safe_layer}.png"
                        save_map_grid(
                            image=images[i].detach().cpu(),
                            pca_map=pca_maps[i],
                            xai_map=xai_maps[i],
                            output_path=fig_path,
                            title=(
                                f"{mode} | {xai_method} | {layer_name} | target={gradcam_target} | "
                                f"true={int(true_labels[i].item())} pred={int(preds[i].item())} "
                                f"cam={int(target_labels[i].item())}"
                            ),
                        )

            processed += current_batch_size
            elapsed = time.time() - started_at
            progress.set_postfix({"images": processed, "rows": len(rows), "img/s": f"{processed / max(elapsed, 1e-8):.2f}"})

        ensure_dir(output_csv.parent)
        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False)
        curve_path = output_csv.parent / f"{output_csv.stem}_curve.png"
        save_layer_curve(output_csv, curve_path, title=f"ViT LSAS | {mode} | {xai_method} | target={gradcam_target}")

        elapsed = time.time() - started_at
        accuracy = correct / max(processed, 1)
        print("\nViT alignment evaluation completed.")
        print(f"Saved CSV: {output_csv}")
        print(f"Saved curve: {curve_path}")
        print(f"Processed images: {processed}")
        print(f"Rows: {len(rows)}")
        print(f"Expected rows: {processed * len(layers)}")
        print(f"Accuracy on evaluated subset: {accuracy:.4f}")
        print(f"Elapsed: {elapsed / 60:.2f} min")
        print("\nMean metrics per layer:")
        print(df.groupby("layer")[["lsas", "corr_pca_xai", "soft_iou_pca_xai"]].mean())

        if len(rows) != processed * len(layers):
            raise RuntimeError(f"Wrong number of rows. Got {len(rows)}, expected {processed * len(layers)}.")

        update_manifest(
            exp=exp,
            cfg=cfg,
            status="completed",
            extra={
                "processed_images": processed,
                "rows": len(rows),
                "accuracy_on_evaluated_subset": accuracy,
                "xai_method": xai_method,
                "gradcam_target": gradcam_target,
                "output_csv": str(output_csv),
                "curve_path": str(curve_path),
                "elapsed_minutes": elapsed / 60,
            },
        )

    except Exception as exc:
        update_manifest(exp=exp, cfg=cfg, status="failed", extra={"error": repr(exc), "traceback": traceback.format_exc()})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cifar10_vit_supervised.yaml")
    parser.add_argument("--mode", default="vit_supervised", choices=["vit_supervised", "vit_lejepa"])
    parser.add_argument("--gradcam-target", choices=["predicted", "true"], default=None)
    parser.add_argument("--vit-xai-method", choices=["token_gradient", "attention_rollout"], default=None)
    args = parser.parse_args()
    evaluate_vit_alignment(
        config_path=args.config,
        mode=args.mode,
        gradcam_target_override=args.gradcam_target,
        vit_xai_method_override=args.vit_xai_method,
    )


if __name__ == "__main__":
    main()
