from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.networks.heads import MLPHead


@dataclass(frozen=True)
class ViTConfig:
    image_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    num_classes: int = 10
    embed_dim: int = 192
    depth: int = 6
    num_heads: int = 3
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    attn_dropout: float = 0.1


class PatchEmbed(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 192,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size={image_size} must be divisible by patch_size={patch_size}")
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B,C,H,W] -> [B,D,Gh,Gw] -> [B,N,D]
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class MLP(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        attn_dropout: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.drop_path = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim=embed_dim, hidden_dim=int(embed_dim * mlp_ratio), dropout=dropout)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(
            x_norm,
            x_norm,
            x_norm,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        x = x + self.drop_path(attn_out)
        x = x + self.mlp(self.norm2(x))
        if return_attention:
            # [B, heads, T, T]
            return x, attn_weights
        return x


class ViTCifarBackbone(nn.Module):
    """
    Small CIFAR-friendly ViT.

    Captured tensors are full token tensors [B, 1+N, D]. Downstream saliency/PCA
    code removes the CLS token. This avoids retaining gradients on unused slices.
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        num_registers: int = 0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = self.patch_embed.grid_size
        self.num_patches = self.patch_embed.num_patches
        self.embed_dim = embed_dim
        self.depth = depth

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.num_registers = int(num_registers)
        self.num_prefix_tokens = 1 + self.num_registers
        if self.num_registers > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, self.num_registers, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.num_registers > 0:
            nn.init.trunc_normal_(self.register_tokens, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_tokens(
        self,
        x: torch.Tensor,
        capture_layers: list[str] | None = None,
        retain_grad: bool = False,
        return_attentions: bool = False,
    ):
        capture_set = set(capture_layers or [])
        x = self.patch_embed(x)
        batch_size = x.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        if self.num_registers > 0:
            regs = self.register_tokens.expand(batch_size, -1, -1)
            x = torch.cat([x[:, :1], regs, x[:, 1:]], dim=1)
        x = self.pos_drop(x)

        captured: dict[str, torch.Tensor] = {}
        attentions: dict[str, torch.Tensor] = {}

        for idx, block in enumerate(self.blocks):
            layer_name = f"blocks.{idx}"
            if return_attentions:
                x, attn = block(x, return_attention=True)
                attentions[layer_name] = attn
            else:
                x = block(x, return_attention=False)

            if layer_name in capture_set:
                if retain_grad and torch.is_grad_enabled():
                    x.retain_grad()
                captured[layer_name] = x

        x = self.norm(x)
        if "norm" in capture_set:
            if retain_grad and torch.is_grad_enabled():
                x.retain_grad()
            captured["norm"] = x

        if return_attentions:
            return x, captured, attentions
        return x, captured

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        tokens, _ = self.forward_tokens(x)
        return tokens[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


class SupervisedViTCifar(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        num_classes: int = 10,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        num_registers: int = 0,
    ):
        super().__init__()
        self.backbone = ViTCifarBackbone(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=3,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attn_dropout=attn_dropout,
            num_registers=num_registers,
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)

    def forward_with_intermediates(
        self,
        x: torch.Tensor,
        capture_layers: list[str],
        retain_grad: bool = False,
        return_attentions: bool = False,
    ):
        if return_attentions:
            tokens, captured, attentions = self.backbone.forward_tokens(
                x,
                capture_layers=capture_layers,
                retain_grad=retain_grad,
                return_attentions=True,
            )
            logits = self.classifier(tokens[:, 0])
            return logits, captured, attentions

        tokens, captured = self.backbone.forward_tokens(
            x,
            capture_layers=capture_layers,
            retain_grad=retain_grad,
            return_attentions=False,
        )
        logits = self.classifier(tokens[:, 0])
        return logits, captured

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))


class LeJEPAViTCifar(nn.Module):
    """ViT backbone + projector/predictor for LeJEPA-SIGReg pretraining."""

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        projection_dim: int = 256,
        prediction_dim: int = 512,
        num_registers: int = 0,
    ):
        super().__init__()
        self.backbone = ViTCifarBackbone(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=3,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attn_dropout=attn_dropout,
            num_registers=num_registers,
        )
        self.projector = MLPHead(embed_dim, prediction_dim, projection_dim)
        self.predictor = MLPHead(projection_dim, prediction_dim, projection_dim)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)

    def forward_projected(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(self.forward_features(x))

    def forward_views(self, views: list[torch.Tensor]) -> dict[str, list[torch.Tensor] | torch.Tensor]:
        z = [self.forward_projected(v) for v in views]
        p = [self.predictor(zi) for zi in z]
        return {"z": z, "p": p, "embeddings_for_sigreg": torch.cat(z, dim=0)}


class LinearProbeViTCifar(nn.Module):
    def __init__(self, backbone: ViTCifarBackbone, num_classes: int = 10):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.embed_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)

    def forward_with_intermediates(
        self,
        x: torch.Tensor,
        capture_layers: list[str],
        retain_grad: bool = False,
        return_attentions: bool = False,
    ):
        if return_attentions:
            tokens, captured, attentions = self.backbone.forward_tokens(
                x,
                capture_layers=capture_layers,
                retain_grad=retain_grad,
                return_attentions=True,
            )
            logits = self.classifier(tokens[:, 0])
            return logits, captured, attentions
        tokens, captured = self.backbone.forward_tokens(
            x,
            capture_layers=capture_layers,
            retain_grad=retain_grad,
        )
        logits = self.classifier(tokens[:, 0])
        return logits, captured

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))
