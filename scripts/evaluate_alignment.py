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
from src.features.pca_masks import pca_heatmaps_from_activation
from src.networks.resnet import (
    LinearProbeResNet18Cifar,
    ResNet18CifarBackbone,
    SupervisedResNet18Cifar,
)
from src.networks.factory import load_classifier_from_checkpoint
from src.utils import ensure_dir, get_device, get_module_by_name, load_yaml, set_seed
from src.visualization.plot_layer_curves import save_layer_curve
from src.visualization.plot_maps import save_map_grid
from src.xai.gradcam import MultiLayerGradCAM


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


def load_stage1_model(config: dict[str, Any], mode: str, device: torch.device) -> torch.nn.Module:
    if mode not in {"supervised", "lejepa"}:
        raise ValueError("ResNet alignment mode must be 'supervised' or 'lejepa'.")
    return load_classifier_from_checkpoint(
        cfg=config,
        mode=mode,
        device=device,
        requires_grad=True,
    )


def _resolve_target_classes(
    gradcam_target: str,
    labels: torch.Tensor,
) -> torch.Tensor | None:
    gradcam_target = gradcam_target.strip().lower()

    if gradcam_target == "predicted":
        return None

    if gradcam_target in {"true", "gt", "label", "ground_truth", "ground-truth"}:
        return labels

    raise ValueError(f"Unknown gradcam_target={gradcam_target!r}. Use 'predicted' or 'true'.")


def evaluate_layer_alignment(
    config_path: str,
    mode: str,
    gradcam_target_override: str | None = None,
) -> None:
    cfg = load_yaml(config_path)

    architecture = str(cfg.get("model", {}).get("architecture", "")).lower()

    if architecture.startswith("vit"):
        raise ValueError(
            "You are using the ResNet/CNN evaluator on a ViT config. "
            "Use scripts/evaluate_vit_alignment.py instead, or submit with target 'vit_lejepa' / 'vit_supervised'. "
            f"Got architecture={architecture!r}, mode={mode!r}."
        )

    if gradcam_target_override is not None:
        cfg.setdefault("evaluation", {})
        cfg["evaluation"]["gradcam_target"] = gradcam_target_override

    gradcam_target = str(cfg["evaluation"].get("gradcam_target", "predicted")).strip().lower()

    cfg["evaluation"]["output_csv"] = _append_stem_suffix(
        cfg["evaluation"]["output_csv"],
        gradcam_target,
    )
    cfg["evaluation"]["figure_dir"] = str(
        Path(cfg["evaluation"]["figure_dir"]) / f"target_{gradcam_target}"
    )

    exp = prepare_experiment(
        cfg=cfg,
        config_path=config_path,
        run_kind=f"evaluate_alignment_{mode}_{gradcam_target}",
        mode=mode,
    )

    try:
        set_seed(int(cfg["project"]["seed"]))
        torch.backends.cudnn.benchmark = True

        device = get_device()
        print(f"Using device: {device}")
        if device.type != "cuda":
            print("WARNING: CUDA is not available. Evaluation may be very slow on CPU.")

        eval_batch_size = int(cfg["evaluation"].get("batch_size", 32))
        max_samples = int(cfg["evaluation"].get("max_samples", 256))
        num_visualizations = int(cfg["evaluation"].get("num_visualizations", 4))
        save_visualizations = _as_bool(cfg["evaluation"].get("save_visualizations", True), default=True)
        layers = list(cfg["evaluation"]["layers"])

        print("ResNet alignment evaluation settings:")
        print(f"  mode: {mode}")
        print(f"  gradcam_target: {gradcam_target}")
        print(f"  checkpoint: {cfg['evaluation']['checkpoint_path']}")
        print(f"  output_csv: {cfg['evaluation']['output_csv']}")
        print(f"  figure_dir: {cfg['evaluation']['figure_dir']}")
        print(f"  max_samples: {max_samples}")
        print(f"  batch_size: {eval_batch_size}")
        print(f"  layers: {layers}")

        cfg.setdefault("data", {})
        cfg["data"]["batch_size"] = eval_batch_size
        loaders = build_loaders(cfg, self_supervised=False)

        model = load_stage1_model(cfg, mode=mode, device=device)
        output_csv = Path(cfg["evaluation"]["output_csv"])
        figure_dir = ensure_dir(cfg["evaluation"]["figure_dir"])

        target_layers = {layer_name: get_module_by_name(model, layer_name) for layer_name in layers}

        rows: list[dict[str, Any]] = []
        processed = 0
        correct = 0
        started_at = time.time()

        progress = tqdm(loaders.test, desc=f"evaluate {mode}/{gradcam_target}", dynamic_ncols=True)

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

            target_classes = _resolve_target_classes(
                gradcam_target=gradcam_target,
                labels=labels,
            )

            with torch.enable_grad():
                with MultiLayerGradCAM(model, target_layers) as gradcam:
                    cam_maps_by_layer, logits, activations_by_layer = gradcam.generate(
                        images,
                        target_classes=target_classes,
                    )

            preds = logits.argmax(dim=1).detach().cpu()
            true_labels = labels.detach().cpu()
            target_labels = preds if target_classes is None else target_classes.detach().cpu()

            correct += int((preds == true_labels).sum().item())

            for layer_name in layers:
                activation = activations_by_layer[layer_name]
                pca_maps = pca_heatmaps_from_activation(
                    activation=activation,
                    output_size=tuple(images.shape[-2:]),
                    component=0,
                    use_abs=True,
                )
                xai_maps = cam_maps_by_layer[layer_name]

                for i in range(current_batch_size):
                    global_idx = processed + i
                    metrics = lsas(
                        pca_maps[i], xai_maps[i],
                        mi_weight=float(cfg["evaluation"].get("mi_weight", 0.0)),
                        mi_bins=int(cfg["evaluation"].get("mi_bins", 32)),
                    )
                    rows.append(
                        {
                            "sample_idx": global_idx,
                            "mode": mode,
                            "xai_method": "gradcam",
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
                                f"{mode} | {layer_name} | target={gradcam_target} | "
                                f"true={int(true_labels[i].item())} "
                                f"pred={int(preds[i].item())} "
                                f"cam={int(target_labels[i].item())}"
                            ),
                        )

            processed += current_batch_size
            elapsed = time.time() - started_at
            progress.set_postfix(
                {
                    "images": processed,
                    "rows": len(rows),
                    "img/s": f"{processed / max(elapsed, 1e-8):.2f}",
                }
            )

        ensure_dir(output_csv.parent)
        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False)

        curve_path = output_csv.parent / f"{output_csv.stem}_curve.png"
        save_layer_curve(
            output_csv,
            curve_path,
            title=f"Stage 1 ResNet LSAS: {mode} | target={gradcam_target}",
        )

        elapsed = time.time() - started_at
        accuracy = correct / max(processed, 1)

        print("\nEvaluation completed.")
        print(f"Saved CSV: {output_csv}")
        print(f"Saved curve: {curve_path}")
        print(f"Processed images: {processed}")
        print(f"Rows: {len(rows)}")
        print(f"Expected rows: {processed * len(layers)}")
        print(f"Accuracy on evaluated subset: {accuracy:.4f}")
        print(f"Elapsed: {elapsed / 60:.2f} min")
        print("\nMean metrics per layer:")
        metric_cols = [c for c in ["lsas", "corr_pca_xai", "soft_iou_pca_xai"] if c in df.columns]
        print(df.groupby("layer")[metric_cols].mean())

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
                "xai_method": "gradcam",
                "gradcam_target": gradcam_target,
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
    parser.add_argument("--mode", required=True, choices=["supervised", "lejepa"])
    parser.add_argument("--gradcam-target", choices=["predicted", "true"], default=None)
    args = parser.parse_args()

    evaluate_layer_alignment(
        config_path=args.config,
        mode=args.mode,
        gradcam_target_override=args.gradcam_target,
    )


if __name__ == "__main__":
    main()
