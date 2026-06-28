from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


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
            raise ValueError(
                f"image_size={image_size} must be divisible by patch_size={patch_size}"
            )

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] -> [B, D, Gh, Gw] -> [B, N, D]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


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
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim=embed_dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(
            x_norm,
            x_norm,
            x_norm,
            need_weights=False,
        )
        x = x + self.drop_path(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x


class ViTCifarBackbone(nn.Module):
    """
    Small CIFAR-friendly ViT.

    Important for your project:
    - The backbone exposes patch tokens at selected transformer blocks.
    - These patch tokens are used for:
        1. PCA token maps
        2. token-gradient saliency maps
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
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Returns:
            final tokens: [B, 1+N, D]
            captured patch tokens:
                layer name -> [B, N, D]

        Layer names are:
            blocks.0
            blocks.1
            ...
            blocks.{depth-1}
        """
        capture_set = set(capture_layers or [])

        x = self.patch_embed(x)
        batch_size = x.shape[0]

        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        captured: dict[str, torch.Tensor] = {}

        for idx, block in enumerate(self.blocks):
            x = block(x)
            layer_name = f"blocks.{idx}"

            if layer_name in capture_set:
                patch_tokens = x[:, 1:, :]
                if retain_grad and torch.is_grad_enabled():
                    patch_tokens.retain_grad()
                captured[layer_name] = patch_tokens

        x = self.norm(x)

        if "norm" in capture_set:
            patch_tokens = x[:, 1:, :]
            if retain_grad and torch.is_grad_enabled():
                patch_tokens.retain_grad()
            captured["norm"] = patch_tokens

        return x, captured

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        tokens, _ = self.forward_tokens(x)
        cls = tokens[:, 0]
        return cls

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
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)

    def forward_with_intermediates(
        self,
        x: torch.Tensor,
        capture_layers: list[str],
        retain_grad: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        tokens, captured = self.backbone.forward_tokens(
            x,
            capture_layers=capture_layers,
            retain_grad=retain_grad,
        )
        logits = self.classifier(tokens[:, 0])
        return logits, captured

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))