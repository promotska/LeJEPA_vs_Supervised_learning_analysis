from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from src.data.dataloaders import build_cifar10_loaders
from src.evaluation.sas import lsas
from src.features.pca_masks import pca_heatmap_from_activation
from src.networks.resnet import LinearProbeResNet18Cifar, ResNet18CifarBackbone, SupervisedResNet18Cifar
from src.training.checkpointing import load_checkpoint
from src.utils import ensure_dir, get_device, get_module_by_name, load_yaml, set_seed
from src.visualization.plot_layer_curves import save_layer_curve
from src.visualization.plot_maps import save_map_grid
from src.xai.gradcam import GradCAM


def load_stage1_model(config: dict, mode: str, device: torch.device) -> torch.nn.Module:
    checkpoint_path = config["evaluation"]["checkpoint_path"]
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)

    if mode == "supervised":
        model = SupervisedResNet18Cifar(num_classes=int(config["model"]["num_classes"]))
        model.load_state_dict(checkpoint["model_state"])
    elif mode == "lejepa":
        backbone = ResNet18CifarBackbone()
        model = LinearProbeResNet18Cifar(backbone, num_classes=int(config["model"]["num_classes"]))
        model.load_state_dict(checkpoint["model_state"])
    else:
        raise ValueError("mode must be 'supervised' or 'lejepa'.")

    # Grad-CAM needs gradients through the backbone even for the linear-probe model.
    for p in model.parameters():
        p.requires_grad_(True)
    return model.to(device).eval()


def evaluate_layer_alignment(config_path: str, mode: str) -> None:
    cfg = load_yaml(config_path)
    set_seed(int(cfg["project"]["seed"]))
    device = get_device()

    eval_batch_size = int(cfg["evaluation"].get("batch_size", 1))
    if eval_batch_size != 1:
        raise ValueError("Stage-1 layer evaluation expects batch_size=1 for per-image PCA.")

    loaders = build_cifar10_loaders(
        root=cfg["data"]["root"],
        batch_size=1,
        num_workers=int(cfg["data"]["num_workers"]),
        val_fraction=float(cfg["data"]["val_fraction"]),
        seed=int(cfg["project"]["seed"]),
        self_supervised=False,
    )
    model = load_stage1_model(cfg, mode=mode, device=device)

    output_csv = Path(cfg["evaluation"]["output_csv"])
    figure_dir = ensure_dir(cfg["evaluation"]["figure_dir"])
    max_samples = int(cfg["evaluation"].get("max_samples", 64))
    layers = list(cfg["evaluation"]["layers"])

    rows: list[dict] = []
    sample_count = 0

    for sample_idx, (images, labels) in enumerate(tqdm(loaders.test, desc=f"evaluate {mode}")):
        if sample_idx >= max_samples:
            break
        images = images.to(device)
        labels = labels.to(device)

        for layer_name in layers:
            target_layer = get_module_by_name(model, layer_name)
            with GradCAM(model, target_layer) as gradcam:
                cam_maps, logits = gradcam.generate(images)
                activation = gradcam.hook.data.activation
                if activation is None:
                    raise RuntimeError("Missing activation after Grad-CAM forward pass.")

                pca_map = pca_heatmap_from_activation(
                    activation=activation,
                    output_size=tuple(images.shape[-2:]),
                    component=0,
                    use_abs=True,
                )
                xai_map = cam_maps[0]
                metrics = lsas(pca_map, xai_map)
                pred = int(logits.argmax(dim=1).item())
                true = int(labels.item())

                row = {
                    "sample_idx": sample_idx,
                    "mode": mode,
                    "layer": layer_name,
                    "true_label": true,
                    "pred_label": pred,
                    **metrics,
                }
                rows.append(row)

                if sample_idx < 8:
                    safe_layer = layer_name.replace(".", "_")
                    fig_path = figure_dir / f"sample_{sample_idx:03d}_{safe_layer}.png"
                    save_map_grid(
                        image=images[0].detach().cpu(),
                        pca_map=pca_map,
                        xai_map=xai_map,
                        output_path=fig_path,
                        title=f"{mode} | {layer_name} | true={true} pred={pred}",
                    )
        sample_count += 1

    ensure_dir(output_csv.parent)
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    curve_path = output_csv.parent / f"{output_csv.stem}_curve.png"
    save_layer_curve(output_csv, curve_path, title=f"Stage 1 LSAS curve: {mode}")
    print(f"Saved {len(rows)} rows for {sample_count} images to {output_csv}")
    print(f"Saved layer curve to {curve_path}")
