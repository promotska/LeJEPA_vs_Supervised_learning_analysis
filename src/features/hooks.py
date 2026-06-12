from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class HookData:
    activation: Optional[torch.Tensor] = None
    gradient: Optional[torch.Tensor] = None


class ActivationGradientHook:
    """Capture activations and gradients from a module."""

    def __init__(self, module: torch.nn.Module):
        self.data = HookData()
        self._forward_handle = module.register_forward_hook(self._forward_hook)
        self._backward_handle = module.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.data.activation = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.data.gradient = grad_output[0].detach()

    def close(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()
