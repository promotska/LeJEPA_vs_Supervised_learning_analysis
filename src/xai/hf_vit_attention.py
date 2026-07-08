from __future__ import annotations

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


def _scores_to_maps(scores: torch.Tensor, grid_size: int, output_size: tuple[int, int]) -> np.ndarray:
    b, n = scores.shape
    if n != grid_size * grid_size:
        raise ValueError(f"Token count N={n} does not match grid_size^2={grid_size * grid_size}")
    maps = scores.reshape(b, 1, grid_size, grid_size)
    maps = F.interpolate(maps, size=output_size, mode="bilinear", align_corners=False)
    return _normalize_maps(maps)


@torch.no_grad()
def generate_hf_vit_last_attention(
    model: torch.nn.Module,
    images: torch.Tensor,
    layers: list[str],
    grid_size: int,
    head_fusion: str = "mean",
) -> tuple[dict[str, np.ndarray], torch.Tensor, dict[str, torch.Tensor]]:
    """Final-block CLS-to-patch attention for public HF ViT checkpoints.

    This is not full Attention Rollout, because the OK-AI HF output exposes
    `last_self_attention` rather than attention from every transformer block.
    It is still a useful public-pretrained attention saliency baseline.
    """
    model.eval()
    logits, full_tokens_by_layer, attentions_by_layer = model.forward_with_intermediates(
        images,
        capture_layers=layers,
        retain_grad=False,
        return_attentions=True,
    )

    maps: dict[str, np.ndarray] = {}
    patch_tokens_by_layer: dict[str, torch.Tensor] = {}

    for layer in layers:
        attn = attentions_by_layer[layer]
        if attn.ndim == 3:
            attn_fused = attn
        elif attn.ndim == 4:
            if head_fusion == "mean":
                attn_fused = attn.mean(dim=1)
            elif head_fusion == "max":
                attn_fused = attn.max(dim=1).values
            elif head_fusion == "min":
                attn_fused = attn.min(dim=1).values
            else:
                raise ValueError(f"Unknown head_fusion={head_fusion!r}")
        else:
            raise ValueError(f"Expected attention [B,T,T] or [B,H,T,T], got {tuple(attn.shape)}")

        cls_to_patch = attn_fused[:, 0, 1:]
        maps[layer] = _scores_to_maps(cls_to_patch, grid_size=grid_size, output_size=tuple(images.shape[-2:]))
        patch_tokens_by_layer[layer] = full_tokens_by_layer[layer][:, 1:, :].detach()

    return maps, logits.detach(), patch_tokens_by_layer
