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

from src.data.dataloaders import build_cifar10_loaders
from src.evaluation.sas import lsas
from src.experiment import prepare_experiment, update_manifest
from src.features.pca_masks import pca_heatmaps_from_tokens
from src.networks.vit import SupervisedViTCifar
from src.training.checkpointing import load_checkpoint
from src.utils import ensure_dir, get_device, load_yaml, set_seed
from src.visualization.plot_layer_curves import save_layer_curve
from src.visualization.plot_maps import save_map_grid
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


def _resolve_target_classes(
    gradcam_target: str,
    labels: torch.Tensor,
) -> torch.Tensor | None:
    gradcam_target = gradcam_target.lower().strip()

    if gradcam_target == "predicted":
        return None

    if gradcam_target in {"true", "gt", "label", "ground_truth", "ground-truth"}:
        return labels

    raise ValueError(
        f"Unknown gradcam_target={gradcam_target!r}. Use 'predicted' or 'true'."
    )


def build_vit_model_from_config(cfg: dict) -> SupervisedViTCifar:
    model_cfg = cfg["model"]

    return SupervisedViTCifar(
        image_size=int(model_cfg.get("image_size", 32)),
        patch_size=int(model_cfg.get("patch_size", 4)),
        num_classes=int(model_cfg["num_classes"]),
        embed_dim=int(model_cfg.get("embed_dim", 192)),
        depth=int(model_cfg.get("depth", 6)),
        num_heads=int(model_cfg.get("num_heads", 3)),
        mlp_ratio=float(model_cfg.get("mlp_ratio", 4.0)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        attn_dropout=float(model_cfg.get("attn_dropout", 0.1)),
    )


def load_vit_model(config: dict, device: torch.device) -> SupervisedViTCifar:
    checkpoint_path = config["evaluation"]["checkpoint_path"]

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path, map_location=device)

    model = build_vit_model_from_config(config)
    model.load_state_dict(checkpoint["model_state"])

    for p in model.parameters():
        p.requires_grad_(True)

    return model.to(device).eval()


def evaluate_vit_alignment(
    config_path: str,
    gradcam_target_override: str | None = None,
) -> None:
    cfg = load_yaml(config_path)

    if gradcam_target_override is not None:
        cfg.setdefault("evaluation", {})
        cfg["evaluation"]["gradcam_target"] = gradcam_target_override

    gradcam_target = str(
        cfg["evaluation"].get("gradcam_target", "predicted")
    ).lower().strip()

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
        run_kind=f"evaluate_vit_supervised_{gradcam_target}",
        mode="vit_supervised",
    )

    try:
        set_seed(int(cfg["project"]["seed"]))
        torch.backends.cudnn.benchmark = True

        device = get_device()
        print(f"Using device: {device}")

        if device.type != "cuda":
            print("WARNING: CUDA is not available. Evaluation may be slow on CPU.")

        eval_batch_size = int(cfg["evaluation"].get("batch_size", 32))
        max_samples = int(cfg["evaluation"].get("max_samples", 256))
        num_visualizations = int(cfg["evaluation"].get("num_visualizations", 4))
        save_visualizations = _as_bool(
            cfg["evaluation"].get("save_visualizations", True),
            default=True,
        )

        layers = list(cfg["evaluation"]["layers"])

        print("ViT alignment evaluation settings:")
        print(f"  config: {config_path}")
        print(f"  checkpoint: {cfg['evaluation']['checkpoint_path']}")
        print(f"  gradcam_target: {gradcam_target}")
        print(f"  layers: {layers}")
        print(f"  max_samples: {max_samples}")
        print(f"  batch_size: {eval_batch_size}")
        print(f"  output_csv: {cfg['evaluation']['output_csv']}")

        loaders = build_cifar10_loaders(
            root=cfg["data"]["root"],
            batch_size=eval_batch_size,
            num_workers=int(cfg["data"]["num_workers"]),
            val_fraction=float(cfg["data"]["val_fraction"]),
            seed=int(cfg["project"]["seed"]),
            self_supervised=False,
        )

        model = load_vit_model(cfg, device=device)

        output_csv = Path(cfg["evaluation"]["output_csv"])
        figure_dir = ensure_dir(cfg["evaluation"]["figure_dir"])

        rows: list[dict] = []
        processed = 0
        correct = 0
        started_at = time.time()

        progress = tqdm(
            loaders.test,
            desc=f"evaluate vit/{gradcam_target}",
            dynamic_ncols=True,
        )

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

            target_classes = _resolve_target_classes(
                gradcam_target=gradcam_target,
                labels=labels,
            )

            with torch.enable_grad():
                saliency_by_layer, logits, tokens_by_layer = generate_vit_token_saliency(
                    model=model,
                    images=images,
                    layers=layers,
                    grid_size=int(model.backbone.grid_size),
                    target_classes=target_classes,
                    use_abs=True,
                )

            preds = logits.argmax(dim=1).detach().cpu()
            true_labels = labels.detach().cpu()

            if target_classes is None:
                target_labels = preds
            else:
                target_labels = target_classes.detach().cpu()

            correct += int((preds == true_labels).sum().item())

            for layer_name in layers:
                tokens = tokens_by_layer[layer_name]

                pca_maps = pca_heatmaps_from_tokens(
                    tokens=tokens,
                    grid_size=int(model.backbone.grid_size),
                    output_size=tuple(images.shape[-2:]),
                    component=0,
                    use_abs=True,
                )

                xai_maps = saliency_by_layer[layer_name]

                for i in range(current_batch_size):
                    global_idx = processed + i
                    metrics = lsas(pca_maps[i], xai_maps[i])

                    rows.append(
                        {
                            "sample_idx": global_idx,
                            "mode": "vit_supervised",
                            "xai_method": "token_gradient",
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
                                f"ViT supervised | {layer_name} | target={gradcam_target} | "
                                f"true={int(true_labels[i].item())} "
                                f"pred={int(preds[i].item())} "
                                f"cam={int(target_labels[i].item())}"
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
        save_layer_curve(
            output_csv,
            curve_path,
            title=f"Stage 2 ViT supervised LSAS | target={gradcam_target}",
        )

        elapsed = time.time() - started_at
        accuracy = correct / max(processed, 1)

        print()
        print("ViT alignment evaluation completed.")
        print(f"Saved CSV: {output_csv}")
        print(f"Saved curve: {curve_path}")
        print(f"Processed images: {processed}")
        print(f"Rows: {len(rows)}")
        print(f"Expected rows: {processed * len(layers)}")
        print(f"Accuracy on evaluated subset: {accuracy:.4f}")
        print(f"Elapsed: {elapsed / 60:.2f} min")

        print()
        print("Mean metrics per layer:")
        print(df.groupby("layer")[["lsas", "corr_pca_xai", "soft_iou_pca_xai"]].mean())

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
                "accuracy_on_evaluated_subset": accuracy,
                "xai_method": "token_gradient",
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
            extra={
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/cifar10_vit_supervised.yaml",
    )
    parser.add_argument(
        "--gradcam-target",
        choices=["predicted", "true"],
        default=None,
    )
    args = parser.parse_args()

    evaluate_vit_alignment(
        config_path=args.config,
        gradcam_target_override=args.gradcam_target,
    )


if __name__ == "__main__":
    main()