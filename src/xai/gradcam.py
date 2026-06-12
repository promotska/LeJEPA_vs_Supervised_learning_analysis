from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.features.hooks import ActivationGradientHook


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.hook = ActivationGradientHook(target_layer)

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
        """Generate Grad-CAM maps.

        Returns:
            cams: numpy array [B,H,W] normalized to [0,1]
            logits: raw model outputs [B,num_classes]
        """
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
        cams = F.interpolate(cams, size=images.shape[-2:], mode="bilinear", align_corners=False)

        cams_np = cams.detach().float().cpu().numpy()[:, 0]
        for i in range(cams_np.shape[0]):
            min_v = cams_np[i].min()
            max_v = cams_np[i].max()
            cams_np[i] = (cams_np[i] - min_v) / (max_v - min_v + 1e-8)
        return cams_np, logits.detach()
