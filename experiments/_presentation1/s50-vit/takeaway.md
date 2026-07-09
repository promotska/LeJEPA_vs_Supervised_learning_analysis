# Takeaway: LeJEPA vs supervised

- **Overall alignment (LSAS)** was largely unchanged at deeper layers (mean Δ over blocks.8, blocks.11 = +0.002).
- **Decomposition:** correlation Δ = +0.056 on average, overlap (soft-IoU) Δ = -0.074 — the two move in opposite directions.
- **Significant layers (FDR):** blocks.2; n = 752 images/layer, single seed.

_Caveats: token-gradient saliency only (attention-rollout not yet run); single seed; deeper-layer token gradients are noisy by construction._