# Takeaway: LeJEPA vs supervised

- **Overall alignment (LSAS)** increased at deeper layers (mean Δ over blocks.8, blocks.11 = +0.029).
- **Decomposition:** correlation Δ = +0.056 on average, overlap (soft-IoU) Δ = -0.005 — the two move in opposite directions.
- **Significant layers (FDR):** blocks.2, blocks.5, blocks.8; n = 1000 images/layer, single seed.

_Caveats: token-gradient saliency only (attention-rollout not yet run); single seed; deeper-layer token gradients are noisy by construction._