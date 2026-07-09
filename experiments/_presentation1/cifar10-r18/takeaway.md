# Takeaway: LeJEPA vs supervised

- **Overall alignment (LSAS)** was largely unchanged at deeper layers (mean Δ over backbone.layer3, backbone.layer4 = +0.006).
- **Decomposition:** correlation Δ = +0.037 on average, overlap (soft-IoU) Δ = -0.020 — the two move in opposite directions.
- **Significant layers (FDR):** backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4; n = 1000 images/layer, single seed.

_Caveats: token-gradient saliency only (attention-rollout not yet run); single seed; deeper-layer token gradients are noisy by construction._