# Takeaway: registers (4) vs no registers

- **Overall alignment (LSAS)** decreased at deeper layers (mean Δ over blocks.3, blocks.4 = -0.073).
- **Decomposition:** correlation Δ = -0.103 on average, overlap (soft-IoU) Δ = +0.033 — the two move in opposite directions.
- **Significant layers (FDR):** blocks.1, blocks.3, blocks.4; n = 1000 images/layer, single seed.

_Caveats: token-gradient saliency only (attention-rollout not yet run); single seed; deeper-layer token gradients are noisy by construction._