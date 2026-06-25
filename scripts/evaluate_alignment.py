from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from src.data.dataloaders import build_cifar10_loaders
from src.evaluation.sas import lsas
from src.features.pca_masks import pca_heatmaps_from_activation
from src.networks.resnet import (
    LinearProbeResNet18Cifar,
    ResNet18CifarBackbone,
    SupervisedResNet18Cifar,
)
from src.training.checkpointing import load_checkpoint
from src.utils import (
    ensure_dir,
    get_device,
    get_module_by_name,
    load_yaml,
    set_seed,
)
from src.visualization.plot_layer_curves import save_layer_curve
from src.visualization.plot_maps import save_map_grid
from src.xai.gradcam import MultiLayerGradCAM
from src.experiment import prepare_experiment, update_manifest


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


def load_stage1_model(config: dict, mode: str, device: torch.device) -> torch.nn.Module:
    checkpoint_path = config["evaluation"]["checkpoint_path"]

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path, map_location=device)

    if mode == "supervised":
        model = SupervisedResNet18Cifar(num_classes=int(config["model"]["num_classes"]))
        model.load_state_dict(checkpoint["model_state"])
    elif mode == "lejepa":
        backbone = ResNet18CifarBackbone()
        model = LinearProbeResNet18Cifar(
            backbone,
            num_classes=int(config["model"]["num_classes"]),
        )
        model.load_state_dict(checkpoint["model_state"])
    else:
        raise ValueError("mode must be 'supervised' or 'lejepa'.")

    # Grad-CAM needs gradients.
    for p in model.parameters():
        p.requires_grad_(True)

    return model.to(device).eval()


def evaluate_layer_alignment(config_path: str, mode: str) -> None:
    cfg = load_yaml(config_path)

    exp = prepare_experiment(
        cfg=cfg,
        config_path=config_path,
        run_kind=f"evaluate_{mode}",
        mode=mode,
    )

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
    expected_rows = max_samples * len(layers)

    print("Evaluation settings:")
    print(f"  mode: {mode}")
    print(f"  config: {config_path}")
    print(f"  checkpoint: {cfg['evaluation']['checkpoint_path']}")
    print(f"  data root: {cfg['data']['root']}")
    print(f"  max_samples: {max_samples}")
    print(f"  batch_size: {eval_batch_size}")
    print(f"  layers: {layers}")
    print(f"  expected rows: {expected_rows}")
    print(f"  save_visualizations: {save_visualizations}")
    print(f"  num_visualizations: {num_visualizations}")

    loaders = build_cifar10_loaders(
        root=cfg["data"]["root"],
        batch_size=eval_batch_size,
        num_workers=int(cfg["data"]["num_workers"]),
        val_fraction=float(cfg["data"]["val_fraction"]),
        seed=int(cfg["project"]["seed"]),
        self_supervised=False,
    )

    model = load_stage1_model(cfg, mode=mode, device=device)

    output_csv = Path(cfg["evaluation"]["output_csv"])
    figure_dir = ensure_dir(cfg["evaluation"]["figure_dir"])

    target_layers = {
        layer_name: get_module_by_name(model, layer_name)
        for layer_name in layers
    }

    rows: list[dict] = []
    processed = 0
    correct = 0
    started_at = time.time()

    progress = tqdm(loaders.test, desc=f"evaluate {mode}", dynamic_ncols=True)

    for batch_idx, (images, labels) in enumerate(progress):
        if processed >= max_samples:
            break

        remaining = max_samples - processed
        if images.shape[0] > remaining:
            images = images[:remaining]
            labels = labels[:remaining]

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        current_batch_size = images.shape[0]

        with torch.enable_grad():
            with MultiLayerGradCAM(model, target_layers) as gradcam:
                cam_maps_by_layer, logits, activations_by_layer = gradcam.generate(images)

        preds = logits.argmax(dim=1).detach().cpu()
        true_labels = labels.detach().cpu()

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

                metrics = lsas(pca_maps[i], xai_maps[i])

                rows.append(
                    {
                        "sample_idx": global_idx,
                        "mode": mode,
                        "layer": layer_name,
                        "true_label": int(true_labels[i].item()),
                        "pred_label": int(preds[i].item()),
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
                            f"{mode} | {layer_name} | "
                            f"true={int(true_labels[i].item())} "
                            f"pred={int(preds[i].item())}"
                        ),
                    )

        processed += current_batch_size
        elapsed = time.time() - started_at
        img_per_sec = processed / max(elapsed, 1e-8)

        progress.set_postfix(
            {
                "images": processed,
                "rows": len(rows),
                "img/s": f"{img_per_sec:.2f}",
            }
        )

    ensure_dir(output_csv.parent)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    curve_path = output_csv.parent / f"{output_csv.stem}_curve.png"
    save_layer_curve(output_csv, curve_path, title=f"Stage 1 LSAS curve: {mode}")

    elapsed = time.time() - started_at
    accuracy = correct / max(processed, 1)

    print()
    print("Evaluation completed.")
    print(f"Saved CSV: {output_csv}")
    print(f"Saved curve: {curve_path}")
    print(f"Processed images: {processed}")
    print(f"Rows: {len(rows)}")
    print(f"Expected rows: {processed * len(layers)}")
    print(f"Accuracy on evaluated subset: {accuracy:.4f}")
    print(f"Elapsed: {elapsed / 60:.2f} min")
    print(f"Throughput: {processed / max(elapsed, 1e-8):.2f} images/sec")

    print()
    print("Rows per layer:")
    print(df.groupby("layer").size())

    print()
    print("Mean metrics per layer:")
    metric_cols = [
        col for col in [
            "lsas",
            "corr_pca_xai",
            "soft_iou_pca_xai",
        ]
        if col in df.columns
    ]
    print(df.groupby("layer")[metric_cols].mean())

    if processed != max_samples:
        print()
        print(f"WARNING: requested max_samples={max_samples}, but processed={processed}.")

    if len(rows) != processed * len(layers):
        raise RuntimeError(
            f"Wrong number of rows. Got {len(rows)}, expected {processed * len(layers)}."
        )

    update_manifest(
        exp=exp,
        cfg=cfg,
        status="completed",
        extra={
            "processed_images": processed,
            "rows": len(rows),
            "accuracy_on_evaluated_subset": accuracy if "accuracy" in locals() else None,
            "output_csv": str(output_csv),
            "curve_path": str(curve_path),
            "elapsed_minutes": elapsed / 60,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True, choices=["supervised", "lejepa"])
    args = parser.parse_args()

    evaluate_layer_alignment(config_path=args.config, mode=args.mode)


if __name__ == "__main__":
    main()