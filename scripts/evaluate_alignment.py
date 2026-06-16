from __future__ import annotations

import argparse
import time
from pathlib import Path

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


def load_stage1_model(config: dict, mode: str, device: torch.device) -> torch.nn.Module:
    checkpoint_path = config["evaluation"]["checkpoint_path"]
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

    for p in model.parameters():
        p.requires_grad_(True)

    return model.to(device).eval()


def evaluate_layer_alignment(config_path: str, mode: str) -> None:
    cfg = load_yaml(config_path)
    set_seed(int(cfg["project"]["seed"]))

    device = get_device()
    print(f"Using device: {device}")
    if device.type != "cuda":
        print("WARNING: CUDA is not available. Evaluation may be very slow on CPU.")

    eval_batch_size = int(cfg["evaluation"].get("batch_size", 32))
    max_samples = int(cfg["evaluation"].get("max_samples", 256))
    num_visualizations = int(cfg["evaluation"].get("num_visualizations", 4))
    save_visualizations = bool(cfg["evaluation"].get("save_visualizations", True))

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
    layers = list(cfg["evaluation"]["layers"])

    target_layers = {
        layer_name: get_module_by_name(model, layer_name)
        for layer_name in layers
    }

    rows: list[dict] = []
    processed = 0
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

        batch_size = images.shape[0]

        with MultiLayerGradCAM(model, target_layers) as gradcam:
            cam_maps_by_layer, logits, activations_by_layer = gradcam.generate(images)

        preds = logits.argmax(dim=1).detach().cpu()
        true_labels = labels.detach().cpu()

        for layer_name in layers:
            activation = activations_by_layer[layer_name]

            pca_maps = pca_heatmaps_from_activation(
                activation=activation,
                output_size=tuple(images.shape[-2:]),
                component=0,
                use_abs=True,
            )

            xai_maps = cam_maps_by_layer[layer_name]

            for i in range(batch_size):
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

        processed += batch_size
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
    print(f"Saved {len(rows)} rows for {processed} images to {output_csv}")
    print(f"Saved layer curve to {curve_path}")
    print(f"Elapsed: {elapsed / 60:.2f} min")
    print(f"Throughput: {processed / max(elapsed, 1e-8):.2f} images/sec")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True, choices=["supervised", "lejepa"])
    args = parser.parse_args()

    evaluate_layer_alignment(config_path=args.config, mode=args.mode)


if __name__ == "__main__":
    main()