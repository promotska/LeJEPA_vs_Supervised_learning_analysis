from __future__ import annotations

"""Sanity-check baselines for the PCA<->XAI alignment metric.

These control for the fact that on centered datasets (like CIFAR) both PCA and
saliency maps tend to peak in the middle, so a chunk of raw LSAS can be explained
by shared center bias rather than genuine image-specific co-localization.

Two controls:
  * center prior   -- a fixed Gaussian blob at the image center. LSAS(map, center)
                      is the "how center-biased is this map on its own" floor.
  * shuffled pairs -- LSAS(PCA_i, XAI_j) for i != j. If alignment were only center
                      bias, mismatched pairs would score just as high; the
                      matched-minus-shuffled gap is the real, image-specific signal.
"""

import numpy as np


def center_prior_map(height: int, width: int, sigma_frac: float = 0.25) -> np.ndarray:
    """Normalized 2-D Gaussian centered on the image, in [0, 1]."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    sigma = sigma_frac * float(min(height, width))
    g = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma * sigma)))
    return (g - g.min()) / (g.max() - g.min() + 1e-8)


def derangement_index(n: int, seed: int = 0) -> np.ndarray:
    """A permutation of range(n) with no fixed points (for n > 1)."""
    if n <= 1:
        return np.zeros(n, dtype=int)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    fixed = np.where(perm == np.arange(n))[0]
    for idx in fixed:
        swap = (idx + 1) % n
        perm[idx], perm[swap] = perm[swap], perm[idx]
    return perm
