from __future__ import annotations
"""Compare single-layer alignment results across models (e.g. off-the-shelf ViT-B
supervised vs LeJEPA, optionally alongside our own model's final layer).

Each --arm is one CSV with lsas/corr_pca_xai/soft_iou_pca_xai + a 'layer' column.
Multi-layer arms use their deepest layer (or --layer). Arms align on sample_idx;
the first two arms get paired stats (bootstrap CI, Cohen's d, Wilcoxon)."""
import argparse, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
try:
    from src.evaluation.lei import layer_sort_key
except Exception:
    def layer_sort_key(l):
        p = str(l).replace("_", ".").split("."); n = [int(x) for x in p if x.isdigit()]
        return (0, n[-1], str(l)) if n else (1, 0, str(l))
try:
    from scipy.stats import wilcoxon; HAVE = True
except Exception:
    HAVE = False

METRICS = ["lsas", "corr_pca_xai", "soft_iou_pca_xai"]
PRETTY = {"lsas": "LSAS", "corr_pca_xai": "correlation", "soft_iou_pca_xai": "overlap (soft-IoU)"}
COLORS = ["#7f7f7f", "#0072B2", "#009E73", "#D55E00", "#9467bd"]

def load_arm(spec, prefer_layer):
    label, path = spec.split("=", 1)
    df = pd.read_csv(path)
    layers = sorted(df["layer"].astype(str).unique(), key=layer_sort_key)
    lyr = prefer_layer if (prefer_layer in layers) else layers[-1]
    d = df[df["layer"].astype(str) == lyr][["sample_idx"] + METRICS].drop_duplicates("sample_idx")
    return label, lyr, d

def mean_ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if x.size < 2: return (float(x.mean()) if x.size else float("nan")), 0.0
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(x.size))

def boot(diff, nb=5000, seed=0):
    d = np.asarray(diff, float); d = d[~np.isnan(d)]
    if d.size < 2: return float("nan"), float("nan"), float("nan")
    r = np.random.default_rng(seed); m = d[r.integers(0, d.size, size=(nb, d.size))].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

def cohend(diff):
    d = np.asarray(diff, float); d = d[~np.isnan(d)]; sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True,
                    help="LABEL=path.csv (repeatable; first two get paired stats)")
    ap.add_argument("--layer", default=None, help="multi-layer arms: which layer (default deepest)")
    ap.add_argument("--title", default="Off-the-shelf ViT-B/16: supervised vs LeJEPA")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    arms = [load_arm(s, args.layer) for s in args.arm]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    lines = []; emit = lambda s="": (print(s), lines.append(s))

    stats = {}
    for label, lyr, d in arms:
        stats[label] = {m: mean_ci(d[m]) for m in METRICS}
        stats[label]["_layer"], stats[label]["_n"] = lyr, len(d)

    x = np.arange(len(METRICS)); w = 0.8 / len(arms)
    fig, ax = plt.subplots(figsize=(1.9 * len(METRICS) + 2, 4.6))
    for k, (label, lyr, d) in enumerate(arms):
        means = [stats[label][m][0] for m in METRICS]; cis = [stats[label][m][1] for m in METRICS]
        ax.bar(x + k * w - 0.4 + w / 2, means, w, yerr=cis, capsize=4,
               color=COLORS[k % len(COLORS)], label=f"{label} ({lyr})")
    ax.set_xticks(x); ax.set_xticklabels([PRETTY[m] for m in METRICS])
    ax.set_ylabel("value (mean, 95% CI)"); ax.set_title(args.title)
    ax.legend(frameon=False, fontsize=9); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(out / "fig_ots_bars.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    pd.DataFrame([{"model": label, "layer": stats[label]["_layer"], "n": stats[label]["_n"],
                   **{PRETTY[m]: round(stats[label][m][0], 4) for m in METRICS}}
                  for label, _, _ in arms]).to_csv(out / "table_ots.csv", index=False)

    emit("=" * 72); emit(args.title); emit("=" * 72)
    for label, lyr, d in arms:
        emit(f"{label:30s} layer={lyr:18s} n={stats[label]['_n']}  " +
             "  ".join(f"{PRETTY[m]}={stats[label][m][0]:.3f}" for m in METRICS))

    if len(arms) >= 2:
        (la, _, da), (lb, _, db) = arms[0], arms[1]
        merged = da.merge(db, on="sample_idx", suffixes=("_a", "_b"))
        emit(f"\nPaired  {lb} - {la}  (n_paired={len(merged)}):")
        for m in METRICS:
            diff = (merged[f"{m}_b"] - merged[f"{m}_a"]).to_numpy()
            md, lo, hi = boot(diff); dd = cohend(diff); p = ""
            if HAVE and np.count_nonzero(diff) and diff.size >= 10:
                try: p = f"  p={wilcoxon(diff).pvalue:.2e}"
                except Exception: p = ""
            emit(f"  {PRETTY[m]:20s} Δ={md:+.4f}  CI=[{lo:+.4f},{hi:+.4f}]  d={dd:+.2f}{p}")

    (out / "summary.txt").write_text("\n".join(lines))
    emit(f"\nWrote fig_ots_bars.png, table_ots.csv, summary.txt to {out}")

if __name__ == "__main__":
    main()