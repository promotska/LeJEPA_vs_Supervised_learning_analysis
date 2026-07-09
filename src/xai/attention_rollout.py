from __future__ import annotations

import re

import numpy as np
import torch
import torch.nn.functional as F


def _normalize_maps(maps: torch.Tensor) -> np.ndarray:
    maps = maps.detach().float()
    b = maps.shape[0]
    flat = maps.view(b, -1)
    min_v = flat.min(dim=1, keepdim=True).values
    max_v = flat.max(dim=1, keepdim=True).values
    flat = (flat - min_v) / (max_v - min_v + 1e-8)
    return flat.view(b, *maps.shape[-2:]).cpu().numpy()


def _block_index(layer_name: str) -> int:
    match = re.fullmatch(r"blocks\.(\d+)", layer_name)
    if not match:
        raise ValueError(f"Attention rollout supports block layers like 'blocks.3', got {layer_name!r}")
    return int(match.group(1))


def _scores_to_maps(scores: torch.Tensor, grid_size: int, output_size: tuple[int, int]) -> np.ndarray:
    b, n = scores.shape
    if n != grid_size * grid_size:
        raise ValueError(f"Token count N={n} does not match grid_size^2={grid_size * grid_size}")
    maps = scores.reshape(b, 1, grid_size, grid_size)
    maps = F.interpolate(maps, size=output_size, mode="bilinear", align_corners=False)
    return _normalize_maps(maps)


@torch.no_grad()
def generate_vit_attention_rollout(
    model: torch.nn.Module,
    images: torch.Tensor,
    layers: list[str],
    grid_size: int,
    discard_ratio: float = 0.0,
    head_fusion: str = "mean",
    num_registers: int = 0,
) -> tuple[dict[str, np.ndarray], torch.Tensor, dict[str, torch.Tensor]]:
    """
    Attention Rollout for ViT.

    Returns maps for each requested block layer. The map is CLS-token attention
    rolled out up to that block and projected onto patch tokens.

    Attention Rollout is not class-target-specific. Use GMAR/token-gradient when
    class-conditioned ViT explanations are required.
    """
    model.eval()
    logits, full_tokens_by_layer, attentions_by_layer = model.forward_with_intermediates(
        images,
        capture_layers=layers,
        retain_grad=False,
        return_attentions=True,
    )

    max_idx = max(_block_index(layer) for layer in layers)
    requested = {layer: _block_index(layer) for layer in layers}
    rollout: torch.Tensor | None = None
    maps: dict[str, np.ndarray] = {}
    patch_tokens_by_layer: dict[str, torch.Tensor] = {}

    for idx in range(max_idx + 1):
        layer_name = f"blocks.{idx}"
        attn = attentions_by_layer[layer_name]  # [B,H,T,T]

        if head_fusion == "mean":
            attn_fused = attn.mean(dim=1)
        elif head_fusion == "max":
            attn_fused = attn.max(dim=1).values
        elif head_fusion == "min":
            attn_fused = attn.min(dim=1).values
        else:
            raise ValueError(f"Unknown head_fusion={head_fusion!r}")

        if discard_ratio > 0:
            b, t, _ = attn_fused.shape
            flat = attn_fused.reshape(b, -1)
            num_discard = int(flat.shape[1] * discard_ratio)
            if num_discard > 0:
                _, indices = flat.topk(num_discard, dim=1, largest=False)
                flat.scatter_(1, indices, 0.0)
                attn_fused = flat.reshape(b, t, t)

        eye = torch.eye(attn_fused.shape[-1], device=attn_fused.device, dtype=attn_fused.dtype)
        attn_aug = attn_fused + eye.unsqueeze(0)
        attn_aug = attn_aug / attn_aug.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        rollout = attn_aug if rollout is None else attn_aug @ rollout

        for req_layer, req_idx in requested.items():
            if req_idx == idx:
                cls_to_patch = rollout[:, 0, 1 + num_registers:]
                maps[req_layer] = _scores_to_maps(
                    scores=cls_to_patch,
                    grid_size=grid_size,
                    output_size=tuple(images.shape[-2:]),
                )
                patch_tokens_by_layer[req_layer] = full_tokens_by_layer[req_layer][:, 1 + num_registers:, :].detach()

    return maps, logits.detach(), patch_tokens_by_layer
