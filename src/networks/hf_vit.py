from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _get_output(outputs: Any, key: str):
    if isinstance(outputs, dict):
        return outputs[key]
    if hasattr(outputs, key):
        return getattr(outputs, key)
    try:
        return outputs[key]
    except Exception as exc:
        raise KeyError(f"Could not read output key {key!r} from HF model outputs of type {type(outputs)}") from exc


class HFViTBackbone(nn.Module):
    """Hugging Face ViT-like feature extractor wrapper.

    Designed for OK-AI LeJEPA/DINO/iBOT ViT checkpoints that return:
      - latent: CLS feature [B, D]
      - patch_latent: patch features [B, N, D]
      - last_self_attention: attention from final block [B, H, T, T]

    Notes:
      * Most OK-AI checkpoints expose only final patch tokens and final attention
        through the public HF forward output. Therefore this wrapper supports a
        final-layer public-pretrained comparison, not a full layer-emergence curve.
      * For layer-wise emergence, use locally defined ViT checkpoints where all
        block activations/attentions are available.
    """

    def __init__(
        self,
        model_id: str,
        revision: str | None = None,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int | None = None,
        local_files_only: bool = False,
        trust_remote_code: bool = True,
        torch_dtype: str | None = None,
        freeze: bool = True,
    ):
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError(
                "HFViTBackbone requires transformers. Install with: pip install transformers safetensors"
            ) from exc

        kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
        }
        if revision:
            kwargs["revision"] = revision
        if torch_dtype:
            dtype_name = str(torch_dtype).lower()
            if dtype_name in {"bf16", "bfloat16"}:
                kwargs["torch_dtype"] = torch.bfloat16
            elif dtype_name in {"fp16", "float16", "half"}:
                kwargs["torch_dtype"] = torch.float16
            elif dtype_name in {"fp32", "float32"}:
                kwargs["torch_dtype"] = torch.float32
            elif dtype_name in {"auto"}:
                kwargs["torch_dtype"] = "auto"
            else:
                raise ValueError(f"Unsupported torch_dtype={torch_dtype!r}")

        self.model_id = model_id
        self.revision = revision
        self.hf_model = AutoModel.from_pretrained(model_id, **kwargs)

        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.grid_size = self.image_size // self.patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.embed_dim = int(embed_dim or self._infer_embed_dim_from_config(default=768))

        if freeze:
            self.freeze()

    def _infer_embed_dim_from_config(self, default: int) -> int:
        cfg = getattr(self.hf_model, "config", None)
        for key in ("hidden_size", "embed_dim", "hidden_dim", "dim"):
            value = getattr(cfg, key, None) if cfg is not None else None
            if value is not None:
                return int(value)
        return int(default)

    def freeze(self) -> None:
        for p in self.hf_model.parameters():
            p.requires_grad_(False)
        self.hf_model.eval()

    def train(self, mode: bool = True):
        # Keep HF backbone frozen/eval during linear probing. The wrapper's head,
        # if any, is handled by the parent module.
        super().train(mode)
        self.hf_model.eval()
        return self

    def _forward_outputs(self, pixel_values: torch.Tensor):
        try:
            return self.hf_model(pixel_values=pixel_values)
        except TypeError:
            return self.hf_model(pixel_values)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self._forward_outputs(x)
        latent = _get_output(outputs, "latent")
        return latent.float()

    def forward_patch_tokens_and_attention(self, x: torch.Tensor):
        outputs = self._forward_outputs(x)
        latent = _get_output(outputs, "latent").float()
        patch_latent = _get_output(outputs, "patch_latent").float()
        last_self_attention = _get_output(outputs, "last_self_attention")
        return latent, patch_latent, last_self_attention

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


class HFViTLinearProbe(nn.Module):
    """Frozen public HF ViT backbone + trainable linear classifier."""

    def __init__(self, backbone: HFViTBackbone, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(int(backbone.embed_dim), int(num_classes))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)

    def forward_with_intermediates(
        self,
        x: torch.Tensor,
        capture_layers: list[str],
        retain_grad: bool = False,
        return_attentions: bool = False,
    ):
        latent, patch_latent, last_self_attention = self.backbone.forward_patch_tokens_and_attention(x)
        logits = self.classifier(latent)

        full_tokens = torch.cat([latent.unsqueeze(1), patch_latent], dim=1)
        if retain_grad and torch.is_grad_enabled():
            full_tokens.retain_grad()

        # Public OK-AI HF output exposes final patch tokens. We allow aliases to
        # keep configs explicit.
        tokens_by_layer: dict[str, torch.Tensor] = {}
        attentions_by_layer: dict[str, torch.Tensor] = {}
        for layer in capture_layers:
            layer_l = str(layer).lower()
            if layer_l not in {"patch_latent", "last", "final", "last_self_attention"}:
                raise ValueError(
                    "Public HF ViT wrapper only exposes final patch tokens/attention. "
                    "Use evaluation.layers: ['patch_latent'] for OK-AI public checkpoints. "
                    f"Got layer={layer!r}."
                )
            tokens_by_layer[layer] = full_tokens
            if return_attentions:
                attentions_by_layer[layer] = last_self_attention

        if return_attentions:
            return logits, tokens_by_layer, attentions_by_layer
        return logits, tokens_by_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))
