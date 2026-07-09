# Takeaway: LeJEPA vs supervised

- **Overall alignment (LSAS)** increased at deeper layers (mean Δ over backbone.layer3, backbone.layer4 = +0.036).
- **Decomposition:** correlation Δ = +0.042 on average, overlap (soft-IoU) Δ = -0.013 — the two move in opposite directions.
- **Significant layers (FDR):** backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4; n = 752 images/layer, single seed.

_Caveats: token-gradient saliency only (attention-rollout not yet run); single seed; deeper-layer token gradients are noisy by construction._