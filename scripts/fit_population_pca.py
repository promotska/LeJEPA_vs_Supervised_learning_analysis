from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataloaders import build_loaders
from src.features.population_pca import fit_population_pca_from_activations
from src.networks.factory import load_classifier_from_checkpoint
from src.utils import get_device, get_module_by_name, load_yaml, set_seed
from src.xai.gradcam import _LayerHook  # reuse the existing forward-hook


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True, choices=["supervised", "lejepa"])
    parser.add_argument("--layers", nargs="+", required=True)
    parser.add_argument("--num-batches", type=int, default=16)
    parser.add_argument("--num-components", type=int, default=8)
    parser.add_argument("--output", required=True, help="e.g. experiments/<name>/artifacts/population_pca.pt")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg["project"]["seed"]))
    device = get_device()

    model = load_classifier_from_checkpoint(cfg, mode=args.mode, device=device, requires_grad=False)
    loaders = build_loaders(cfg, self_supervised=False)

    hooks = {name: _LayerHook(get_module_by_name(model, name)) for name in args.layers}
    pooled = {name: [] for name in args.layers}

    with torch.no_grad():
        for batch_idx, (images, _labels) in enumerate(loaders.test):
            if batch_idx >= args.num_batches:
                break
            model(images.to(device))
            for name in args.layers:
                pooled[name].append(hooks[name].data.activation.detach())

    for hook in hooks.values():
        hook.close()

    bases = {
        name: fit_population_pca_from_activations(acts, num_components=args.num_components)
        for name, acts in pooled.items()
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(bases, args.output)
    print(f"Saved population PCA bases for layers {args.layers} to {args.output}")


if __name__ == "__main__":
    main()