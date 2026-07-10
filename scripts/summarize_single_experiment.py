from __future__ import annotations
"""Single-experiment summary: center/shuffle control + MI alongside corr/IoU/LSAS.
Reads one experiment's per-sample LSAS CSV (produced with compute_baselines +
report_mi) and writes presentation figures + a table."""
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

LBL = {"lsas": "LSAS", "corr_pca_xai": "correlation", "soft_iou_pca_xai": "overlap (soft-IoU)",
       "nmi_pca_xai": "mutual info (NMI)"}
COL = {"lsas": "#333333", "corr_pca_xai": "#1f77b4", "soft_iou_pca_xai": "#2ca02c", "nmi_pca_xai": "#d62728"}

def discover(md, which):
    c = [p for p in sorted(md.glob(f"*_{which}.csv"))
         if "lei" not in p.name.lower() and not p.name.startswith("representation")]
    tg = [p for p in c if "token_gradient" in p.name]
    return (tg or c)[0] if c else None

def mean_ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if x.size < 2: return (float(x.mean()) if x.size else float("nan")), 0.0
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(x.size))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-dir", required=True)
    ap.add_argument("--which", default="predicted", choices=["predicted", "true"])
    ap.add_argument("--label", default="model")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    exp = Path(args.experiment_dir)
    csv = discover(exp / "metrics", args.which)
    if csv is None: raise SystemExit(f"no *_{args.which}.csv in {exp/'metrics'}")
    df = pd.read_csv(csv)
    out = Path(args.out_dir) if args.out_dir else exp / "single_summary"; out.mkdir(parents=True, exist_ok=True)
    layers = sorted(df["layer"].astype(str).unique(), key=layer_sort_key)
    x = np.arange(len(layers))
    metrics = [m for m in ["lsas", "corr_pca_xai", "soft_iou_pca_xai", "nmi_pca_xai"] if m in df.columns]

    # Figure 1: metrics incl. MI, per layer
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for m in metrics:
        ys = [mean_ci(df[df["layer"].astype(str) == l][m]) for l in layers]
        ax.errorbar(x, [a for a, _ in ys], yerr=[b for _, b in ys], marker="o", capsize=3,
                    color=COL.get(m), label=LBL.get(m, m))
    ax.set_xticks(x); ax.set_xticklabels(layers, rotation=30, ha="right")
    ax.set_ylabel("value (mean, 95% CI)"); ax.set_title(f"{args.label}: alignment metrics incl. mutual information")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(out / "fig_metrics_with_mi.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # Figure 2: center/shuffle control
    has_ctrl = "lsas_shuffled" in df.columns
    if has_ctrl:
        w = 0.27
        per = lambda c: [float(df[df["layer"].astype(str) == l][c].mean()) if c in df.columns else np.nan for l in layers]
        fig2, ax2 = plt.subplots(figsize=(7, 4.4))
        ax2.bar(x - w, per("lsas"), w, color="#2ca02c", label="matched (real)")
        ax2.bar(x, per("lsas_shuffled"), w, color="#7f7f7f", label="shuffled pairs (chance)")
        if "lsas_pca_center" in df.columns:
            ax2.bar(x + w, per("lsas_pca_center"), w, color="#d62728", alpha=0.8, label="PCA vs center prior")
        ax2.set_xticks(x); ax2.set_xticklabels(layers, rotation=30, ha="right")
        ax2.set_ylabel("LSAS"); ax2.set_title(f"{args.label}: alignment vs controls (matched − shuffled = real signal)")
        ax2.legend(frameon=False); ax2.grid(axis="y", alpha=0.25)
        fig2.tight_layout(); fig2.savefig(out / "fig_controls.png", dpi=200, bbox_inches="tight"); plt.close(fig2)

    rows = []
    for l in layers:
        d = df[df["layer"].astype(str) == l]; row = {"layer": l}
        for m in metrics: row[LBL[m]] = round(float(d[m].mean()), 4)
        if has_ctrl:
            row["shuffled"] = round(float(d["lsas_shuffled"].mean()), 4)
            if "lsas_pca_center" in df.columns: row["center"] = round(float(d["lsas_pca_center"].mean()), 4)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "table_single.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    if not has_ctrl: print("NOTE: no lsas_shuffled -> re-run eval with compute_baselines: true.")
    if "nmi_pca_xai" not in df.columns: print("NOTE: no nmi_pca_xai -> apply report_mi in sas.py and re-run eval.")
    print(f"\nWrote figures + table to {out}")

if __name__ == "__main__":
    main()