# Public OK-AI LeJEPA ViT-B/16 integration

This code supports `OK-AI/lejepa-vitb16-pretrain-in1k` as a frozen public pretrained feature extractor.

Important limitations:

- The public HF forward output exposes final `patch_latent` and `last_self_attention`.
- Therefore this stage computes final-token PCA vs final self-attention LSAS, not a full block-by-block LEI curve.
- Train a linear probe on the target dataset before running predicted/true label diagnostics.
- Use this as a public-pretrained comparison stage, separate from controlled from-scratch LeJEPA/Supervised runs.
