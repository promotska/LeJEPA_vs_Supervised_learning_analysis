from __future__ import annotations

import torch
from torch import nn


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, use_bn: bool = True):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.extend([nn.ReLU(inplace=True), nn.Linear(hidden_dim, out_dim)])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
