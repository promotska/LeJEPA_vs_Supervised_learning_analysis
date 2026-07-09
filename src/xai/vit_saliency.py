from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _normalize_maps(maps: torch.Tensor) -> np.ndarray:
    """
    maps: [B, 1, H, W]
    returns numpy [B, H, W]
    """
    maps = maps.detach().float()
    b = maps.shape[0]
    flat = maps.view(b, -1)

    min_v = flat.min(dim=1, keepdim=True).values
    max_v = flat.max(dim=1, keepdim=True).values
    flat = (flat - min_v) / (max_v - min_v + 1e-8)

    return flat.view(b, *maps.shape[-2:]).cpu().numpy()


def _token_scores_to_maps(
    scores: torch.Tensor,
    grid_size: int,
    output_size: tuple[int, int],
) -> np.ndarray:
    """
    scores: [B, N]
    returns numpy [B, output_H, output_W]
    """
    if scores.ndim != 2:
        raise ValueError(f"Expected scores [B,N], got {tuple(scores.shape)}")

    b, n = scores.shape
    if n != grid_size * grid_size:
        raise ValueError(f"Token count N={n} does not match grid_size^2={grid_size * grid_size}")

    maps = scores.reshape(b, 1, grid_size, grid_size)
    maps = F.interpolate(maps, size=output_size, mode="bilinear", align_corners=False)
    return _normalize_maps(maps)


def generate_vit_token_saliency(
    model: torch.nn.Module,
    images: torch.Tensor,
    layers: list[str],
    grid_size: int,
    target_classes: torch.Tensor | None = None,
    use_abs: bool = True,
    num_registers: int = 0,
) -> tuple[dict[str, np.ndarray], torch.Tensor, dict[str, torch.Tensor]]:
    """
    Generates token-gradient saliency maps for selected ViT layers.

    The model returns captured full token tensors [B, 1+N, D]. This function
    removes the CLS token and computes maps over patch tokens only.

    Returns:
        saliency_maps: layer_name -> numpy [B,H,W]
        logits: [B,num_classes]
        patch_tokens_by_layer: layer_name -> tensor [B,N,D]
    """
    model.eval()
    model.zero_grad(set_to_none=True)

    logits, full_tokens_by_layer = model.forward_with_intermediates(
        images,
        capture_layers=layers,
        retain_grad=True,
    )

    if target_classes is None:
        target_classes = logits.argmax(dim=1)

    selected = logits.gather(1, target_classes.view(-1, 1)).sum()
    selected.backward(retain_graph=False)

    saliency_maps: dict[str, np.ndarray] = {}
    patch_tokens_by_layer: dict[str, torch.Tensor] = {}

    for layer_name, full_tokens in full_tokens_by_layer.items():
        grads = full_tokens.grad

        if grads is None:
            raise RuntimeError(
                f"No gradient captured for {layer_name}. "
                "The captured tensor must be the full token tensor used by later blocks, not an unused slice."
            )

        num_prefix = 1 + num_registers
        patch_tokens = full_tokens[:, num_prefix:, :]
        patch_grads = grads[:, num_prefix:, :]

        if use_abs:
            scores = (patch_tokens * patch_grads).sum(dim=-1).abs()
        else:
            scores = torch.relu((patch_tokens * patch_grads).sum(dim=-1))

        if torch.allclose(scores.detach().abs().sum(), torch.tensor(0.0, device=scores.device)):
            scores = patch_grads.norm(dim=-1)

        saliency_maps[layer_name] = _token_scores_to_maps(
            scores=scores,
            grid_size=grid_size,
            output_size=tuple(images.shape[-2:]),
        )
        patch_tokens_by_layer[layer_name] = patch_tokens.detach()

    return saliency_maps, logits.detach(), patch_tokens_by_layer
