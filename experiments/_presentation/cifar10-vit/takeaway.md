# Takeaway: LeJEPA vs supervised

- **Overall alignment (LSAS)** increased at deeper layers (mean Δ over blocks.1, blocks.3 = +0.206).
- **Decomposition:** correlation Δ = +0.355 on average, overlap (soft-IoU) Δ = +0.056 — the two move in the same directions.
- **Significant layers (FDR):** blocks.1, blocks.3; n = 1000 images/layer, single seed.

_Caveats: token-gradient saliency only (attention-rollout not yet run); single seed; deeper-layer token gradients are noisy by construction._