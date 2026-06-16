from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class _HookData:
    activation: torch.Tensor | None = None
    gradient: torch.Tensor | None = None


class _LayerHook:
    def __init__(self, layer: torch.nn.Module):
        self.data = _HookData()
        self.forward_handle = layer.register_forward_hook(self._forward_hook)
        self.backward_handle = layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.data.activation = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.data.gradient = grad_output[0]

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()


def _normalize_cams(cams: torch.Tensor) -> np.ndarray:
    """
    cams: [B, 1, H, W]
    returns numpy [B, H, W] normalized per image to [0, 1]
    """
    cams = cams.detach().float()
    b = cams.shape[0]
    cams = cams.view(b, -1)

    min_v = cams.min(dim=1, keepdim=True).values
    max_v = cams.max(dim=1, keepdim=True).values
    cams = (cams - min_v) / (max_v - min_v + 1e-8)

    return cams.view(b, 1, *cams.shape[1:]).cpu().numpy()


class GradCAM:
    """
    Single-layer Grad-CAM.

    Kept for backward compatibility with your previous code.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.hook = _LayerHook(target_layer)

    def close(self) -> None:
        self.hook.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def generate(
        self,
        images: torch.Tensor,
        target_classes: torch.Tensor | None = None,
    ) -> tuple[np.ndarray, torch.Tensor]:
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        logits = self.model(images)

        if target_classes is None:
            target_classes = logits.argmax(dim=1)

        selected = logits.gather(1, target_classes.view(-1, 1)).sum()
        selected.backward(retain_graph=False)

        activations = self.hook.data.activation
        gradients = self.hook.data.gradient

        if activations is None or gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activation/gradient.")

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cams = (weights * activations).sum(dim=1, keepdim=True)
        cams = F.relu(cams)
        cams = F.interpolate(
            cams,
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cams_np = cams.detach().float().cpu().numpy()[:, 0]
        for i in range(cams_np.shape[0]):
            min_v = cams_np[i].min()
            max_v = cams_np[i].max()
            cams_np[i] = (cams_np[i] - min_v) / (max_v - min_v + 1e-8)

        return cams_np, logits.detach()


class MultiLayerGradCAM:
    """
    Efficient multi-layer Grad-CAM.

    One forward/backward pass gives Grad-CAM maps for all requested layers.
    This is much faster than running one Grad-CAM pass per layer.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layers: Mapping[str, torch.nn.Module],
    ):
        if not target_layers:
            raise ValueError("target_layers must not be empty.")

        self.model = model
        self.target_layers = dict(target_layers)
        self.hooks = {
            name: _LayerHook(layer)
            for name, layer in self.target_layers.items()
        }

    def close(self) -> None:
        for hook in self.hooks.values():
            hook.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def generate(
        self,
        images: torch.Tensor,
        target_classes: torch.Tensor | None = None,
    ) -> tuple[dict[str, np.ndarray], torch.Tensor, dict[str, torch.Tensor]]:
        """
        Returns:
            cam_maps:
                dict layer_name -> numpy [B, H, W]
            logits:
                tensor [B, num_classes]
            activations:
                dict layer_name -> tensor [B, C, h, w], detached
        """
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        logits = self.model(images)

        if target_classes is None:
            target_classes = logits.argmax(dim=1)

        selected = logits.gather(1, target_classes.view(-1, 1)).sum()
        selected.backward(retain_graph=False)

        cam_maps: dict[str, np.ndarray] = {}
        activations_out: dict[str, torch.Tensor] = {}

        for name, hook in self.hooks.items():
            activations = hook.data.activation
            gradients = hook.data.gradient

            if activations is None or gradients is None:
                raise RuntimeError(f"Grad-CAM hook failed for layer: {name}")

            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cams = (weights * activations).sum(dim=1, keepdim=True)
            cams = F.relu(cams)
            cams = F.interpolate(
                cams,
                size=images.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            cams_np = cams.detach().float().cpu().numpy()[:, 0]
            for i in range(cams_np.shape[0]):
                min_v = cams_np[i].min()
                max_v = cams_np[i].max()
                cams_np[i] = (cams_np[i] - min_v) / (max_v - min_v + 1e-8)

            cam_maps[name] = cams_np
            activations_out[name] = activations.detach()

        return cam_maps, logits.detach(), activations_out