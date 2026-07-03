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

from src.data.mask_datasets import build_mask_loader
from src.evaluation.grounded_metrics import grounded_lsas
from src.evaluation.lei import compute_layer_emergence_index
from src.experiment import prepare_experiment, update_manifest
from src.features.pca_masks import pca_heatmaps_from_activation, pca_heatmaps_from_tokens
from src.networks.factory import architecture_name, load_classifier_from_checkpoint
from src.utils import ensure_dir, get_device, get_module_by_name, load_yaml, set_seed
from src.visualization.plot_grounded_curves import save_grounded_layer_curve
from src.visualization.plot_grounded_maps import save_grounded_map_grid
from src.xai.attention_rollout import generate_vit_attention_rollout
from src.xai.gradcam import MultiLayerGradCAM
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


def _resolve_target_classes(target: str, labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor | None:
    target = target.strip().lower()
    if target == "predicted":
        return None
    if target in {"true", "label", "gt", "ground_truth", "ground-truth"}:
        if torch.any(labels < 0):
            raise ValueError(
                "grounded_evaluation target=true requires valid class labels. "
                "For VOC/ImageNet-S mask-only validation, use target=predicted."
            )
        return labels
    raise ValueError(f"Unknown grounded target={target!r}. Use predicted or true.")


class ForwardActivationCapture:
    def __init__(self, model: torch.nn.Module, target_layers: dict[str, torch.nn.Module]):
        self.model = model
        self.activations: dict[str, torch.Tensor] = {}
        self.handles = [
            module.register_forward_hook(self._make_hook(name))
            for name, module in target_layers.items()
        ]

    def _make_hook(self, name: str):
        def hook(module, inputs, output):
            self.activations[name] = output.detach()
        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @torch.no_grad()
    def capture(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        self.activations = {}
        self.model.eval()
        _ = self.model(images)
        return dict(self.activations)


def _target_name(mode: str) -> str:
    mode = mode.lower()
    if mode in {"supervised", "vit_supervised", "lejepa", "vit_lejepa"}:
        return mode
    raise ValueError(f"Unsupported mode={mode!r}")


def _resolve_output_paths(cfg: dict[str, Any], exp, mode: str, target: str) -> tuple[Path, Path]:
    gcfg = cfg.get("grounded_evaluation", {})
    filename_default = f"grounded_{mode}_{target}_g_lsas.csv"
    figure_default = f"grounded_{mode}_{target}"

    out_value = gcfg.get("output_csv", filename_default)
    out_path = Path(out_value)
    if not out_path.is_absolute():
        if str(out_path).startswith("outputs/"):
            out_path = exp.metrics_dir / out_path.name
        else:
            out_path = exp.metrics_dir / out_path

    fig_value = gcfg.get("figure_dir", figure_default)
    fig_path = Path(fig_value)
    if not fig_path.is_absolute():
        if str(fig_path).startswith("outputs/"):
            fig_path = exp.figures_dir / fig_path.name
        else:
            fig_path = exp.figures_dir / fig_path

    return out_path, fig_path


def _mean_std_from_config(cfg: dict[str, Any]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    from src.data.transforms import default_normalization

    data_cfg = cfg.get("data", {})
    gcfg = cfg.get("grounded_evaluation", {})
    norm_name = str(gcfg.get("normalization", data_cfg.get("dataset", "imagenet100")))
    default_mean, default_std = default_normalization(norm_name)
    mean = tuple(float(v) for v in gcfg.get("mean", data_cfg.get("mean", default_mean)))
    std = tuple(float(v) for v in gcfg.get("std", data_cfg.get("std", default_std)))
    return mean, std


def evaluate_resnet_grounded(
    cfg: dict[str, Any],
    mode: str,
    target: str,
    loader,
    model: torch.nn.Module,
    device: torch.device,
    output_csv: Path,
    figure_dir: Path,
) -> pd.DataFrame:
    eval_cfg = cfg.get("evaluation", {})
    gcfg = cfg.get("grounded_evaluation", {})
    layers = list(gcfg.get("layers", eval_cfg.get("layers", [])))
    if not layers:
        raise ValueError("No layers specified. Set evaluation.layers or grounded_evaluation.layers.")

    target_layers = {layer_name: get_module_by_name(model, layer_name) for layer_name in layers}
    max_samples = gcfg.get("max_samples", eval_cfg.get("max_samples", 500))
    max_samples = None if max_samples in {None, "none", "None", "null"} else int(max_samples)
    top_frac = float(gcfg.get("top_frac", 0.20))
    stability = _as_bool(gcfg.get("stability", True), default=True)
    num_visualizations = int(gcfg.get("num_visualizations", eval_cfg.get("num_visualizations", 8)))
    save_visualizations = _as_bool(gcfg.get("save_visualizations", True), default=True)
    mean, std = _mean_std_from_config(cfg)

    rows: list[dict[str, Any]] = []
    processed = 0
    started = time.time()

    for images, images_aug, labels, gt_masks, valid_masks, sample_ids in tqdm(loader, desc=f"G-LSAS ResNet {mode}/{target}"):
        if max_samples is not None and processed >= max_samples:
            break
        if max_samples is not None and images.shape[0] > max_samples - processed:
            keep = max_samples - processed
            images = images[:keep]
            images_aug = images_aug[:keep]
            labels = labels[:keep]
            gt_masks = gt_masks[:keep]
            valid_masks = valid_masks[:keep]
            sample_ids = sample_ids[:keep]

        images = images.to(device, non_blocking=True)
        images_aug = images_aug.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = int(images.shape[0])

        with torch.enable_grad():
            with MultiLayerGradCAM(model, target_layers) as gradcam:
                # logits are needed before target resolution, but predicted target is handled by GradCAM.
                target_classes = None
                if target != "predicted":
                    with torch.no_grad():
                        logits_for_target = model(images)
                    target_classes = _resolve_target_classes(target, labels, logits_for_target)
                xai_maps_by_layer, logits, activations_by_layer = gradcam.generate(images, target_classes=target_classes)

        preds = logits.argmax(dim=1).detach().cpu()
        labels_cpu = labels.detach().cpu()
        target_labels = preds if target == "predicted" else labels_cpu

        if stability:
            with ForwardActivationCapture(model, target_layers) as capture:
                aug_activations_by_layer = capture.capture(images_aug)
        else:
            aug_activations_by_layer = {}

        gt_np = gt_masks.detach().cpu().numpy()
        valid_np = valid_masks.detach().cpu().numpy()

        for layer_name in layers:
            pca_maps = pca_heatmaps_from_activation(
                activation=activations_by_layer[layer_name],
                output_size=tuple(images.shape[-2:]),
                component=int(gcfg.get("pca_component", 0)),
                use_abs=_as_bool(gcfg.get("pca_use_abs", True), default=True),
            )
            xai_maps = xai_maps_by_layer[layer_name]

            if stability and layer_name in aug_activations_by_layer:
                pca_aug_maps = pca_heatmaps_from_activation(
                    activation=aug_activations_by_layer[layer_name],
                    output_size=tuple(images.shape[-2:]),
                    component=int(gcfg.get("pca_component", 0)),
                    use_abs=_as_bool(gcfg.get("pca_use_abs", True), default=True),
                )
            else:
                pca_aug_maps = [None] * batch_size

            for i in range(batch_size):
                global_idx = processed + i
                metrics = grounded_lsas(
                    pca_map=pca_maps[i],
                    xai_map=xai_maps[i],
                    gt_mask=gt_np[i],
                    valid_mask=valid_np[i],
                    pca_aug_map=None if pca_aug_maps[i] is None else pca_aug_maps[i],
                    top_frac=top_frac,
                )
                rows.append(
                    {
                        "sample_idx": global_idx,
                        "sample_id": str(sample_ids[i]),
                        "mode": mode,
                        "architecture": architecture_name(cfg),
                        "xai_method": "gradcam",
                        "target": target,
                        "layer": layer_name,
                        "true_label": int(labels_cpu[i].item()),
                        "pred_label": int(preds[i].item()),
                        "target_label": int(target_labels[i].item()),
                        **metrics,
                    }
                )

                if save_visualizations and global_idx < num_visualizations:
                    safe_layer = layer_name.replace(".", "_")
                    save_grounded_map_grid(
                        image=images[i].detach().cpu(),
                        gt_mask=gt_np[i],
                        pca_map=pca_maps[i],
                        xai_map=xai_maps[i],
                        output_path=figure_dir / f"sample_{global_idx:04d}_{safe_layer}.png",
                        title=f"{mode} | {layer_name} | {target} | pred={int(preds[i])}",
                        mean=mean,
                        std=std,
                    )

        processed += batch_size

    df = pd.DataFrame(rows)
    ensure_dir(output_csv.parent)
    df.to_csv(output_csv, index=False)
    print(f"Saved G-LSAS rows: {output_csv} ({len(df)} rows, {processed} images, {time.time() - started:.1f}s)")
    return df


def evaluate_vit_grounded(
    cfg: dict[str, Any],
    mode: str,
    target: str,
    loader,
    model: torch.nn.Module,
    device: torch.device,
    output_csv: Path,
    figure_dir: Path,
) -> pd.DataFrame:
    eval_cfg = cfg.get("evaluation", {})
    gcfg = cfg.get("grounded_evaluation", {})
    layers = list(gcfg.get("layers", eval_cfg.get("layers", [])))
    if not layers:
        raise ValueError("No layers specified. Set evaluation.layers or grounded_evaluation.layers.")

    max_samples = gcfg.get("max_samples", eval_cfg.get("max_samples", 500))
    max_samples = None if max_samples in {None, "none", "None", "null"} else int(max_samples)
    top_frac = float(gcfg.get("top_frac", 0.20))
    stability = _as_bool(gcfg.get("stability", True), default=True)
    vit_xai_method = str(gcfg.get("vit_xai_method", eval_cfg.get("vit_xai_method", "token_gradient"))).lower()
    num_visualizations = int(gcfg.get("num_visualizations", eval_cfg.get("num_visualizations", 8)))
    save_visualizations = _as_bool(gcfg.get("save_visualizations", True), default=True)
    mean, std = _mean_std_from_config(cfg)

    grid_size = int(model.backbone.grid_size if hasattr(model, "backbone") else model.grid_size)
    rows: list[dict[str, Any]] = []
    processed = 0
    started = time.time()

    for images, images_aug, labels, gt_masks, valid_masks, sample_ids in tqdm(loader, desc=f"G-LSAS ViT {mode}/{target}"):
        if max_samples is not None and processed >= max_samples:
            break
        if max_samples is not None and images.shape[0] > max_samples - processed:
            keep = max_samples - processed
            images = images[:keep]
            images_aug = images_aug[:keep]
            labels = labels[:keep]
            gt_masks = gt_masks[:keep]
            valid_masks = valid_masks[:keep]
            sample_ids = sample_ids[:keep]

        images = images.to(device, non_blocking=True)
        images_aug = images_aug.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = int(images.shape[0])

        if target == "predicted":
            target_classes = None
        else:
            with torch.no_grad():
                logits_for_target = model(images)
            target_classes = _resolve_target_classes(target, labels, logits_for_target)

        if vit_xai_method == "token_gradient":
            with torch.enable_grad():
                xai_maps_by_layer, logits, patch_tokens_by_layer = generate_vit_token_saliency(
                    model=model,
                    images=images,
                    layers=layers,
                    grid_size=grid_size,
                    target_classes=target_classes,
                    use_abs=_as_bool(gcfg.get("token_gradient_abs", True), default=True),
                )
        elif vit_xai_method == "attention_rollout":
            with torch.no_grad():
                xai_maps_by_layer, logits, patch_tokens_by_layer = generate_vit_attention_rollout(
                    model=model,
                    images=images,
                    layers=layers,
                    grid_size=grid_size,
                    discard_ratio=float(gcfg.get("attention_discard_ratio", eval_cfg.get("attention_discard_ratio", 0.0))),
                    head_fusion=str(gcfg.get("attention_head_fusion", eval_cfg.get("attention_head_fusion", "mean"))),
                )
        else:
            raise ValueError(f"Unsupported ViT G-LSAS xai method: {vit_xai_method}")

        preds = logits.argmax(dim=1).detach().cpu()
        labels_cpu = labels.detach().cpu()
        target_labels = preds if target == "predicted" else labels_cpu

        if stability:
            with torch.no_grad():
                _, aug_full_tokens_by_layer = model.forward_with_intermediates(
                    images_aug,
                    capture_layers=layers,
                    retain_grad=False,
                )
                aug_patch_tokens_by_layer = {
                    k: v[:, 1:, :].detach()
                    for k, v in aug_full_tokens_by_layer.items()
                }
        else:
            aug_patch_tokens_by_layer = {}

        gt_np = gt_masks.detach().cpu().numpy()
        valid_np = valid_masks.detach().cpu().numpy()

        for layer_name in layers:
            pca_maps = pca_heatmaps_from_tokens(
                tokens=patch_tokens_by_layer[layer_name],
                grid_size=grid_size,
                output_size=tuple(images.shape[-2:]),
                component=int(gcfg.get("pca_component", 0)),
                use_abs=_as_bool(gcfg.get("pca_use_abs", True), default=True),
            )
            xai_maps = xai_maps_by_layer[layer_name]
            if stability and layer_name in aug_patch_tokens_by_layer:
                pca_aug_maps = pca_heatmaps_from_tokens(
                    tokens=aug_patch_tokens_by_layer[layer_name],
                    grid_size=grid_size,
                    output_size=tuple(images.shape[-2:]),
                    component=int(gcfg.get("pca_component", 0)),
                    use_abs=_as_bool(gcfg.get("pca_use_abs", True), default=True),
                )
            else:
                pca_aug_maps = [None] * batch_size

            for i in range(batch_size):
                global_idx = processed + i
                metrics = grounded_lsas(
                    pca_map=pca_maps[i],
                    xai_map=xai_maps[i],
                    gt_mask=gt_np[i],
                    valid_mask=valid_np[i],
                    pca_aug_map=None if pca_aug_maps[i] is None else pca_aug_maps[i],
                    top_frac=top_frac,
                )
                rows.append(
                    {
                        "sample_idx": global_idx,
                        "sample_id": str(sample_ids[i]),
                        "mode": mode,
                        "architecture": architecture_name(cfg),
                        "xai_method": vit_xai_method,
                        "target": target,
                        "layer": layer_name,
                        "true_label": int(labels_cpu[i].item()),
                        "pred_label": int(preds[i].item()),
                        "target_label": int(target_labels[i].item()),
                        **metrics,
                    }
                )

                if save_visualizations and global_idx < num_visualizations:
                    safe_layer = layer_name.replace(".", "_")
                    save_grounded_map_grid(
                        image=images[i].detach().cpu(),
                        gt_mask=gt_np[i],
                        pca_map=pca_maps[i],
                        xai_map=xai_maps[i],
                        output_path=figure_dir / f"sample_{global_idx:04d}_{safe_layer}.png",
                        title=f"{mode} | {layer_name} | {vit_xai_method} | pred={int(preds[i])}",
                        mean=mean,
                        std=std,
                    )

        processed += batch_size

    df = pd.DataFrame(rows)
    ensure_dir(output_csv.parent)
    df.to_csv(output_csv, index=False)
    print(f"Saved G-LSAS rows: {output_csv} ({len(df)} rows, {processed} images, {time.time() - started:.1f}s)")
    return df


def evaluate_grounded_alignment(config_path: str, mode: str, target_override: str | None = None) -> None:
    cfg = load_yaml(config_path)
    mode = _target_name(mode)
    if target_override is not None:
        cfg.setdefault("grounded_evaluation", {})["target"] = target_override
    target = str(cfg.get("grounded_evaluation", {}).get("target", "predicted")).lower()

    exp = prepare_experiment(
        cfg=cfg,
        config_path=config_path,
        run_kind=f"evaluate_grounded_alignment_{mode}_{target}",
        mode=mode,
    )

    try:
        set_seed(int(cfg["project"].get("seed", 42)))
        torch.backends.cudnn.benchmark = True
        device = get_device()
        print(f"Using device: {device}")

        # Use resolved config so experiment checkpoint relocation is respected.
        cfg = load_yaml(exp.resolved_config_path)
        cfg.setdefault("grounded_evaluation", {})["target"] = target
        output_csv, figure_dir = _resolve_output_paths(cfg, exp, mode, target)
        ensure_dir(figure_dir)

        print("Grounded alignment settings:")
        print(f"  mode: {mode}")
        print(f"  target: {target}")
        print(f"  architecture: {architecture_name(cfg)}")
        print(f"  checkpoint: {cfg['evaluation']['checkpoint_path']}")
        print(f"  output_csv: {output_csv}")
        print(f"  figure_dir: {figure_dir}")
        print(f"  grounded dataset: {cfg.get('grounded_evaluation', {}).get('dataset')}")

        loader = build_mask_loader(cfg)
        model = load_classifier_from_checkpoint(
            cfg=cfg,
            mode=mode,
            device=device,
            requires_grad=True,
        )

        arch = architecture_name(cfg)
        if arch.startswith("vit"):
            df = evaluate_vit_grounded(cfg, mode, target, loader, model, device, output_csv, figure_dir)
        else:
            df = evaluate_resnet_grounded(cfg, mode, target, loader, model, device, output_csv, figure_dir)

        curve_path = output_csv.parent / f"{output_csv.stem}_curve.png"
        save_grounded_layer_curve(
            csv_path=output_csv,
            output_path=curve_path,
            metric="g_lsas",
            title=f"G-LSAS: {mode} | target={target}",
        )

        lei_csv = output_csv.parent / f"{output_csv.stem}_lei.csv"
        lei_yaml = output_csv.parent / f"{output_csv.stem}_lei.yaml"
        lei = compute_layer_emergence_index(
            csv_path=output_csv,
            metric="g_lsas",
            threshold=float(cfg.get("grounded_evaluation", {}).get("lei_threshold", 0.30)),
            output_csv=lei_csv,
            output_yaml=lei_yaml,
        )

        print("\nG-LSAS completed.")
        print(f"Saved CSV: {output_csv}")
        print(f"Saved curve: {curve_path}")
        print(f"Saved LEI CSV: {lei_csv}")
        print(f"Saved LEI YAML: {lei_yaml}")
        print("Mean metrics per layer:")
        cols = ["g_lsas", "corr_pca_xai", "iou_pca_gt", "iou_xai_gt", "pca_stability", "object_focus_pca", "object_focus_xai"]
        print(df.groupby("layer")[[c for c in cols if c in df.columns]].mean())
        print("LEI:", lei)

        update_manifest(
            exp=exp,
            cfg=cfg,
            status="completed",
            extra={
                "note": "grounded G-LSAS evaluation completed",
                "mode": mode,
                "target": target,
                "output_csv": str(output_csv),
                "curve_path": str(curve_path),
                "lei_csv": str(lei_csv),
                "lei_yaml": str(lei_yaml),
                "lei": lei,
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
    parser.add_argument("--mode", required=True, choices=["supervised", "lejepa", "vit_supervised", "vit_lejepa"])
    parser.add_argument("--target", default=None, choices=[None, "predicted", "true"])
    args = parser.parse_args()
    evaluate_grounded_alignment(args.config, mode=args.mode, target_override=args.target)


if __name__ == "__main__":
    main()
