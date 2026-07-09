from __future__ import annotations

import argparse
import sys
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from src.evaluation.lei import layer_sort_key
except Exception:
    def layer_sort_key(layer: str):
        parts = str(layer).replace("_", ".").split(".")
        nums = [int(p) for p in parts if p.isdigit()]
        return (0, nums[-1], str(layer)) if nums else (1, 0, str(layer))

try:
    from scipy.stats import wilcoxon
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# Styling (Okabe-Ito colorblind-safe palette)
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13,
    "legend.fontsize": 11, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
})
COLORS = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p"]
C_DOWN, C_UP = "#D55E00", "#009E73" 
KEY_METRICS = ["lsas", "corr_pca_xai", "soft_iou_pca_xai"]
PRETTY = {"lsas": "LSAS", "corr_pca_xai": "correlation", "soft_iou_pca_xai": "overlap (soft-IoU)"}

# --------------------------------------------------------------------------- #
# Stats Tools
# --------------------------------------------------------------------------- #
def bootstrap_diff_ci(diffs, n_boot=5000, seed=0):
    d = np.asarray(diffs, float); d = d[~np.isnan(d)]
    if d.size < 2: return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def cohens_d(diffs):
    d = np.asarray(diffs, float); d = d[~np.isnan(d)]
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("nan")

def mean_ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if x.size < 2: return float(x.mean()) if x.size else float("nan"), 0.0
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(x.size))

def benjamini_hochberg(pvals, alpha=0.05):
    p = np.asarray(pvals, float); ok = ~np.isnan(p)
    adj = np.full_like(p, np.nan); rej = np.zeros(p.shape, bool)
    idx = np.where(ok)[0]
    if idx.size == 0: return adj, rej
    order = idx[np.argsort(p[idx])]; m = len(order); prev = 1.0
    for rank, j in enumerate(reversed(order)):
        val = min(prev, p[j] * m / (m - rank)); adj[j] = val; prev = val
    rej[order] = adj[order] <= alpha
    return adj, rej

def safe_fname(s: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]+', '_', s).strip('_').lower()

# --------------------------------------------------------------------------- #
# Data Loading
# --------------------------------------------------------------------------- #
def load_model_data(dir_path: str, which: str):
    metrics_dir = Path(dir_path) / "metrics"
    if not metrics_dir.exists(): return None, None
    cands = [p for p in sorted(metrics_dir.glob(f"*_{which}.csv")) if "lei" not in p.name.lower() and not p.name.startswith("representation")]
    if not cands: return None, None
    
    csv_file = ([p for p in cands if "token_gradient" in p.name] or cands)[0]
    raw_df = pd.read_csv(csv_file).drop_duplicates(subset=["sample_idx", "layer"])
    layers = sorted(raw_df["layer"].unique(), key=layer_sort_key)
    
    stats = []
    for lyr in layers:
        m = raw_df[raw_df["layer"] == lyr]
        row = {"layer": lyr}
        for metric in ["lsas", "corr_pca_xai", "soft_iou_pca_xai", "lsas_shuffled", "lsas_pca_center"]:
            if metric in m.columns:
                mean, ci = mean_ci(m[metric].to_numpy())
                row[metric] = mean; row[f"{metric}_ci"] = ci
        stats.append(row)
    return pd.DataFrame(stats), raw_df

# --------------------------------------------------------------------------- #
# Paired Processing (For same-architecture models)
# --------------------------------------------------------------------------- #
def merge_raw(raw_base, raw_comp, layers):
    metrics = [m for m in KEY_METRICS if m in raw_base.columns and m in raw_comp.columns]
    merged = raw_base[["sample_idx", "layer"] + metrics].merge(
        raw_comp[["sample_idx", "layer"] + metrics], on=["sample_idx", "layer"],
        suffixes=("_base", "_comp"), validate="one_to_one")
    return merged, metrics

def per_layer_stats(merged, metric, layers):
    rows, pvals = [], []
    for lyr in layers:
        m = merged[merged["layer"] == lyr]
        b, c = m[f"{metric}_base"].to_numpy(), m[f"{metric}_comp"].to_numpy()
        diffs = c - b
        bm, bci = mean_ci(b); cm, cci = mean_ci(c)
        md, lo, hi = bootstrap_diff_ci(diffs)
        p = float("nan")
        if _HAVE_SCIPY and np.count_nonzero(diffs) and diffs.size >= 10:
            try: p = float(wilcoxon(diffs).pvalue)
            except: pass
        rows.append(dict(layer=lyr, base_mean=bm, base_ci=bci, comp_mean=cm, comp_ci=cci,
                         diff=md, lo=lo, hi=hi, d=cohens_d(diffs), p=p, n=int(diffs.size)))
        pvals.append(p)
    df = pd.DataFrame(rows)
    df["sig_fdr"] = benjamini_hochberg(pvals)[1]
    return df

# --------------------------------------------------------------------------- #
# GLOBAL Figures (Indices on X-axis)
# --------------------------------------------------------------------------- #
def fig_lsas_global(models_stats, labels, out):
    fig, ax = plt.subplots(figsize=(max(7.2, len(labels) * 1.5), 5.0))
    for i, (stats, lab) in enumerate(zip(models_stats, labels)):
        if stats is None or "lsas" not in stats.columns: continue
        x = np.arange(len(stats))
        y = stats["lsas"].to_numpy(); e = stats["lsas_ci"].to_numpy()
        ax.plot(x, y, marker=MARKERS[i % len(MARKERS)], color=COLORS[i % len(COLORS)], lw=2.4, ms=8, label=lab)
        ax.fill_between(x, y - e, y + e, color=COLORS[i % len(COLORS)], alpha=0.15)
    ax.set_xlabel("Relative Layer Index (Shallow → Deep)")
    ax.set_ylabel("LSAS")
    ax.set_title("Global Layer-wise Alignment Comparison")
    ax.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)

def fig_trajectory(models_stats, labels, out):
    fig, ax = plt.subplots(figsize=(max(7.2, len(labels) * 1.5), 5.0))
    has_data = False
    for i, (stats, lab) in enumerate(zip(models_stats, labels)):
        if stats is None or "corr_pca_xai" not in stats.columns: continue
        has_data = True
        x = stats["corr_pca_xai"].to_numpy(); y = stats["soft_iou_pca_xai"].to_numpy()
        c = COLORS[i % len(COLORS)]
        ax.plot(x, y, marker=".", color=c, lw=2.0, ms=8, label=lab, alpha=0.8)
        ax.scatter(x[-1], y[-1], marker="*", facecolors=c, edgecolors="black", s=300, zorder=5)
    if not has_data: return
    ax.set_xlabel("Correlation"); ax.set_ylabel("Overlap (soft-IoU)")
    ax.set_title("Trajectory: Correlation vs Overlap (★ = Final Layer)")
    ax.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)

# --------------------------------------------------------------------------- #
# PAIRED Figures (Actual Layers on X-axis)
# --------------------------------------------------------------------------- #
def fig_arch_specific_lsas(members, layers, out):
    fig, ax = plt.subplots(figsize=(max(7.2, len(layers) * 0.5), 5.0))
    x = np.arange(len(layers))
    for i_global, stats, _, lab in members:
        y = stats["lsas"].to_numpy(); e = stats["lsas_ci"].to_numpy()
        c = COLORS[i_global % len(COLORS)]; mk = MARKERS[i_global % len(MARKERS)]
        ax.plot(x, y, marker=mk, color=c, lw=2.4, ms=8, label=lab)
        ax.fill_between(x, y - e, y + e, color=c, alpha=0.15)
    ax.set_xticks(x); ax.set_xticklabels(layers, rotation=45, ha="right")
    ax.set_ylabel("LSAS")
    ax.set_title("Architecture Match: Layer-wise Alignment")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)

def fig_arch_trade(stats_by_metric, layers, base_label, comp_label, out):
    x = np.arange(len(layers)); w = 0.38
    fig, ax = plt.subplots(figsize=(max(7.2, len(layers) * 0.6), 5.0))
    for off, metric, color in [(-w / 2, "corr_pca_xai", C_DOWN), (w / 2, "soft_iou_pca_xai", C_UP)]:
        if metric not in stats_by_metric: continue
        s = stats_by_metric[metric]
        d = s["diff"].to_numpy(); lo = s["lo"].to_numpy(); hi = s["hi"].to_numpy()
        ax.bar(x + off, d, w, color=color, label=f"Δ {PRETTY[metric]}")
        ax.errorbar(x + off, d, yerr=[d - lo, hi - d], fmt="none", ecolor="black", capsize=3, lw=1)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(layers, rotation=45, ha="right")
    ax.set_ylabel(f"Change ({comp_label} − {base_label})")
    ax.set_title(f"Trade-off: {comp_label} vs {base_label}")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)

# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def render_table(df_display, title, out_png, col_colors=None):
    n_rows, n_cols = df_display.shape
    fig, ax = plt.subplots(figsize=(min(2.5 * n_cols, 16), 0.6 * n_rows + 1.2))
    ax.axis("off"); ax.set_title(title, fontsize=15, pad=12)
    tbl = ax.table(cellText=df_display.values, colLabels=df_display.columns, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 1.6)
    for j in range(n_cols): tbl[0, j].set_facecolor("#2c3e50"); tbl[0, j].set_text_props(color="white", weight="bold")
    if col_colors:
        for (ri, ci), color in col_colors.items(): tbl[ri + 1, ci].set_facecolor(color)
    fig.tight_layout(); fig.savefig(out_png, dpi=200, bbox_inches="tight"); plt.close(fig)

def lsas_table_paired(stats_lsas, base_label, comp_label, out_dir, prefix):
    disp, colors = [], {}
    for i, r in stats_lsas.iterrows():
        star = "✓" if r["sig_fdr"] else ""
        disp.append([r["layer"], f"{r['base_mean']:.3f}", f"{r['comp_mean']:.3f}",
                     f"{r['diff']:+.3f}", f"[{r['lo']:+.3f}, {r['hi']:+.3f}]", f"{r['d']:+.2f}", star])
        colors[(i, 3)] = "#d7f0dd" if r["diff"] > 0 else "#fbe0d6"
    df = pd.DataFrame(disp, columns=["Layer", base_label, comp_label, "Δ", "95% CI", "Cohen d", "Sig."])
    df.to_csv(out_dir / f"{prefix}.csv", index=False)
    render_table(df, f"Paired LSAS: {comp_label} vs {base_label}", out_dir / f"{prefix}.png", colors)

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dirs", nargs="+", help="List of experiment directories")
    ap.add_argument("--labels", nargs="+", help="List of labels matching directories")
    ap.add_argument("--which", default="predicted", choices=["predicted", "true"])
    ap.add_argument("--out-dir", required=True)
    args, unknown = ap.parse_known_args() # allows skipping old legacy args nicely

    dirs, labels = args.dirs, args.labels
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Comparing {len(dirs)} models...")
    
    # Load all models
    loaded = [load_model_data(d, args.which) for d in dirs]
    models_stats = [x[0] if x else None for x in loaded]
    models_raw = [x[1] if x else None for x in loaded]

    # 1. Global Figures
    fig_lsas_global(models_stats, labels, out_dir / "fig1_lsas_global.png")
    fig_trajectory(models_stats, labels, out_dir / "fig2_trajectory.png")

    # 2. Group by Architecture & Build Paired Figures
    arch_groups = {}
    for i, (stats, raw, lab) in enumerate(zip(models_stats, models_raw, labels)):
        if stats is None: continue
        layer_tup = tuple(stats["layer"].tolist())
        if layer_tup not in arch_groups: arch_groups[layer_tup] = []
        arch_groups[layer_tup].append((i, stats, raw, lab))

    arch_idx = 1
    for layer_tup, members in arch_groups.items():
        if len(members) < 2: continue
        layers = list(layer_tup)
        
        # Determine a safe architecture name (based on first member's label)
        arch_prefix = safe_fname(members[0][3].split()[0]) # e.g. "R18" or "ViT"
        
        # Plot identical-layer LSAS
        fig_arch_specific_lsas(members, layers, out_dir / f"fig4_lsas_actual_layers_{arch_prefix}.png")
        print(f"  -> Generated specific architecture graph for '{arch_prefix}' layers.")

        # Compute specific pairwise metrics for every pair in this architecture group
        for idx1 in range(len(members)):
            for idx2 in range(idx1 + 1, len(members)):
                _, _, raw1, lab1 = members[idx1]
                _, _, raw2, lab2 = members[idx2]
                
                merged, metrics = merge_raw(raw1, raw2, layers)
                paired_stats = {m: per_layer_stats(merged, m, layers) for m in metrics}
                
                safe1, safe2 = safe_fname(lab1), safe_fname(lab2)
                pair_slug = f"{safe2}_vs_{safe1}"
                
                if "corr_pca_xai" in paired_stats:
                    fig_arch_trade(paired_stats, layers, lab1, lab2, out_dir / f"fig5_trade_{pair_slug}.png")
                if "lsas" in paired_stats:
                    lsas_table_paired(paired_stats["lsas"], lab1, lab2, out_dir, f"table_lsas_{pair_slug}")
        arch_idx += 1

    print(f"\n✅ All assets written to: {out_dir}")

if __name__ == "__main__":
    main()