from __future__ import annotations

"""
Distilled, PowerPoint-ready summary of a two-arm alignment comparison
(e.g. registers vs no-registers, or LeJEPA vs supervised).

Produces a small curated set in --out-dir:
  fig1_lsas_by_layer.png     headline: LSAS per layer, both arms, 95% CI band
  fig2_decomposition.png     why: per-layer Δcorrelation vs Δoverlap (the "trade")
  fig3_controls.png          (only if center/shuffle columns exist) matched vs null
  table_lsas.png / .csv      per-layer LSAS: base, reg, Δ, 95% CI, Cohen's d, sig
  table_accuracy.png / .csv  (if representation_*.yaml present) probe + kNN accuracy
  takeaway.md                auto-written one-paragraph verdict + the numbers

Stats: paired bootstrap 95% CI, Cohen's d, Wilcoxon p (if SciPy), Benjamini-Hochberg
FDR across layers. All computed per image then aggregated (n = images per layer).

Example:
  python scripts/make_presentation.py \
    --baseline-dir experiments/experiment-c10-vit-lejepa-v4-short-lr1e4-sig005 \
    --register-dir experiments/experiment-c10-vit-lejepa-v4-short-lr1e4-sig005-reg4 \
    --baseline-label "no registers" --register-label "registers (4)" \
    --which predicted
"""

import argparse
import sys
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

# presentation styling (large fonts, clean, colorblind-safe Okabe-Ito)
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13,
    "legend.fontsize": 11, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
})
C_BASE, C_REG = "#7f7f7f", "#0072B2"     # gray / blue
C_DOWN, C_UP = "#D55E00", "#009E73"       # orange-red / green
KEY_METRICS = ["lsas", "corr_pca_xai", "soft_iou_pca_xai"]
PRETTY = {"lsas": "LSAS", "corr_pca_xai": "correlation", "soft_iou_pca_xai": "overlap (soft-IoU)"}


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #
def bootstrap_diff_ci(diffs, n_boot=5000, seed=0):
    d = np.asarray(diffs, float); d = d[~np.isnan(d)]
    if d.size < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cohens_d(diffs):
    d = np.asarray(diffs, float); d = d[~np.isnan(d)]
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("nan")


def mean_ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if x.size < 2:
        return float(x.mean()) if x.size else float("nan"), 0.0
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(x.size))


def benjamini_hochberg(pvals, alpha=0.05):
    p = np.asarray(pvals, float); ok = ~np.isnan(p)
    adj = np.full_like(p, np.nan); rej = np.zeros(p.shape, bool)
    idx = np.where(ok)[0]
    if idx.size == 0:
        return adj, rej
    order = idx[np.argsort(p[idx])]; m = len(order); prev = 1.0
    for rank, j in enumerate(reversed(order)):
        val = min(prev, p[j] * m / (m - rank)); adj[j] = val; prev = val
    rej[order] = adj[order] <= alpha
    return adj, rej


# --------------------------------------------------------------------------- #
# load / pair
# --------------------------------------------------------------------------- #
def discover(metrics_dir: Path, which: str) -> Path | None:
    cands = [p for p in sorted(metrics_dir.glob(f"*_{which}.csv"))
             if "lei" not in p.name.lower() and not p.name.startswith("representation")]
    if not cands:
        return None
    tg = [p for p in cands if "token_gradient" in p.name]
    return (tg or cands)[0]


def load_paired(base_csv, reg_csv):
    base = pd.read_csv(base_csv).drop_duplicates(subset=["sample_idx", "layer"])
    reg = pd.read_csv(reg_csv).drop_duplicates(subset=["sample_idx", "layer"])
    metrics = [m for m in KEY_METRICS if m in base.columns and m in reg.columns]
    merged = base[["sample_idx", "layer"] + metrics].merge(
        reg[["sample_idx", "layer"] + metrics], on=["sample_idx", "layer"],
        suffixes=("_base", "_reg"), validate="one_to_one")
    layers = sorted(merged["layer"].unique(), key=layer_sort_key)
    return merged, metrics, layers


def per_layer_stats(merged, metric, layers):
    rows, pvals = [], []
    for lyr in layers:
        m = merged[merged["layer"] == lyr]
        b, r = m[f"{metric}_base"].to_numpy(), m[f"{metric}_reg"].to_numpy()
        diffs = r - b
        bm, bci = mean_ci(b); rm, rci = mean_ci(r)
        md, lo, hi = bootstrap_diff_ci(diffs)
        p = float("nan")
        if _HAVE_SCIPY and np.count_nonzero(diffs) and diffs.size >= 10:
            try:
                p = float(wilcoxon(diffs).pvalue)
            except Exception:
                p = float("nan")
        rows.append(dict(layer=lyr, base_mean=bm, base_ci=bci, reg_mean=rm, reg_ci=rci,
                         diff=md, lo=lo, hi=hi, d=cohens_d(diffs), p=p, n=int(diffs.size)))
        pvals.append(p)
    df = pd.DataFrame(rows)
    _, rej = benjamini_hochberg(pvals)
    df["sig_fdr"] = rej
    return df


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def fig_lsas(stats_lsas, layers, base_label, reg_label, out):
    x = np.arange(len(layers))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for col, ci, color, lab, mk, ls in [("base_mean", "base_ci", C_BASE, base_label, "o", "--"),
                                        ("reg_mean", "reg_ci", C_REG, reg_label, "s", "-")]:
        y = stats_lsas[col].to_numpy(); e = stats_lsas[ci].to_numpy()
        ax.plot(x, y, marker=mk, ls=ls, color=color, lw=2.4, ms=8, label=lab)
        ax.fill_between(x, y - e, y + e, color=color, alpha=0.15)
    for i, sig in enumerate(stats_lsas["sig_fdr"]):
        if sig:
            ytop = max(stats_lsas["base_mean"][i], stats_lsas["reg_mean"][i]) + 0.02
            ax.annotate("*", (x[i], ytop), ha="center", fontsize=18, color="black")
    ax.set_xticks(x); ax.set_xticklabels(layers)
    ax.set_ylabel("LSAS (PCA–saliency alignment)")
    ax.set_title("Layer-wise alignment: registers vs baseline")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=0.25)
    ax.text(0.99, -0.16, "* = significant after FDR correction  ·  band = 95% CI of the mean",
            transform=ax.transAxes, ha="right", fontsize=9, color="#555")
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_decomposition(stats_by_metric, layers, out):
    """Per-layer Δ (register − baseline) for correlation vs overlap: the 'trade' story."""
    x = np.arange(len(layers)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for off, metric, color in [(-w / 2, "corr_pca_xai", C_DOWN), (w / 2, "soft_iou_pca_xai", C_UP)]:
        if metric not in stats_by_metric:
            continue
        s = stats_by_metric[metric]
        d = s["diff"].to_numpy(); lo = s["lo"].to_numpy(); hi = s["hi"].to_numpy()
        ax.bar(x + off, d, w, color=color, label=f"Δ {PRETTY[metric]}")
        ax.errorbar(x + off, d, yerr=[d - lo, hi - d], fmt="none", ecolor="black", capsize=3, lw=1)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(layers)
    ax.set_ylabel("change with registers (register − baseline)")
    ax.set_title("Registers trade correlation for overlap")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_controls(merged_raw_base, layers, label, out):
    have = [c for c in ["lsas", "lsas_shuffled", "lsas_pca_center"] if c in merged_raw_base.columns]
    if "lsas_shuffled" not in have:
        return False
    x = np.arange(len(layers)); w = 0.27
    def per(col):
        return [float(merged_raw_base[merged_raw_base["layer"] == l][col].mean())
                if col in merged_raw_base.columns else np.nan for l in layers]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(x - w, per("lsas"), w, color=C_UP, label="matched (real)")
    ax.bar(x, per("lsas_shuffled"), w, color=C_BASE, label="shuffled pairs (chance)")
    if "lsas_pca_center" in merged_raw_base.columns:
        ax.bar(x + w, per("lsas_pca_center"), w, color=C_DOWN, alpha=0.8, label="center prior")
    ax.set_xticks(x); ax.set_xticklabels(layers)
    ax.set_ylabel("LSAS"); ax.set_title(f"Alignment is real, not center bias — {label}")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def render_table(df_display, title, out_png, col_colors=None):
    n = len(df_display)
    fig, ax = plt.subplots(figsize=(min(2.2 * len(df_display.columns), 11), 0.6 * n + 1.1))
    ax.axis("off"); ax.set_title(title, fontsize=15, pad=12)
    tbl = ax.table(cellText=df_display.values, colLabels=df_display.columns,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(12); tbl.scale(1, 1.6)
    for j in range(len(df_display.columns)):  # header styling
        c = tbl[0, j]; c.set_facecolor("#2c3e50"); c.set_text_props(color="white", weight="bold")
    if col_colors:
        for (ri, ci), color in col_colors.items():
            tbl[ri + 1, ci].set_facecolor(color)
    fig.tight_layout(); fig.savefig(out_png, dpi=200, bbox_inches="tight"); plt.close(fig)


def lsas_table(stats_lsas, base_label, reg_label, out_dir):
    disp, colors = [], {}
    for i, r in stats_lsas.iterrows():
        star = "✓" if r["sig_fdr"] else ""
        disp.append([r["layer"], f"{r['base_mean']:.3f}", f"{r['reg_mean']:.3f}",
                     f"{r['diff']:+.3f}", f"[{r['lo']:+.3f}, {r['hi']:+.3f}]",
                     f"{r['d']:+.2f}", star])
        colors[(i, 3)] = "#d7f0dd" if r["diff"] > 0 else "#fbe0d6"  # Δ cell green/red
    cols = ["layer", base_label, reg_label, "Δ", "95% CI", "Cohen d", "sig."]
    df = pd.DataFrame(disp, columns=cols)
    df.to_csv(out_dir / "table_lsas.csv", index=False)
    render_table(df, "LSAS by layer (register vs baseline)", out_dir / "table_lsas.png", colors)


def accuracy_table(base_dir, reg_dir, base_label, reg_label, out_dir):
    def rep(d):
        hits = list((Path(d) / "metrics").glob("representation_*.yaml"))
        if not hits:
            return None
        y = yaml.safe_load(hits[0].read_text())
        return (y.get("classifier_or_probe_accuracy") or {}).get("accuracy"), \
               ((y.get("knn") or {}).get("best") or {}).get("accuracy")
    rb, rr = rep(base_dir), rep(reg_dir)
    if not rb or not rr:
        return
    rows = []
    for name, bv, rv in [("linear probe acc.", rb[0], rr[0]), ("kNN acc. (best k)", rb[1], rr[1])]:
        if bv is None or rv is None:
            continue
        rows.append([name, f"{bv:.3f}", f"{rv:.3f}", f"{rv - bv:+.3f}"])
    if not rows:
        return
    df = pd.DataFrame(rows, columns=["metric", base_label, reg_label, "Δ"])
    df.to_csv(out_dir / "table_accuracy.csv", index=False)
    render_table(df, "Downstream accuracy (should stay ~flat)", out_dir / "table_accuracy.png")


# --------------------------------------------------------------------------- #
# takeaway text
# --------------------------------------------------------------------------- #
def write_takeaway(stats_by_metric, layers, base_label, reg_label, out_dir):
    ls = stats_by_metric["lsas"]
    deep = ls[ls["layer"].isin(layers[-2:])]
    mean_deep = float(deep["diff"].mean())
    corr = stats_by_metric.get("corr_pca_xai")
    iou = stats_by_metric.get("soft_iou_pca_xai")
    direction = "increased" if mean_deep > 0.01 else ("decreased" if mean_deep < -0.01 else "was largely unchanged")
    lines = [f"# Takeaway: {reg_label} vs {base_label}", ""]
    lines.append(f"- **Overall alignment (LSAS)** {direction} at deeper layers "
                 f"(mean Δ over {', '.join(layers[-2:])} = {mean_deep:+.3f}).")
    if corr is not None and iou is not None:
        lines.append(f"- **Decomposition:** correlation Δ = {corr['diff'].mean():+.3f} on average, "
                     f"overlap (soft-IoU) Δ = {iou['diff'].mean():+.3f} — the two move in "
                     f"{'opposite' if corr['diff'].mean() * iou['diff'].mean() < 0 else 'the same'} directions.")
    sig = ls[ls["sig_fdr"]]["layer"].tolist()
    lines.append(f"- **Significant layers (FDR):** {', '.join(sig) if sig else 'none'}; "
                 f"n = {int(ls['n'].iloc[0])} images/layer, single seed.")
    lines.append("")
    lines.append("_Caveats: token-gradient saliency only (attention-rollout not yet run); "
                 "single seed; deeper-layer token gradients are noisy by construction._")
    (out_dir / "takeaway.md").write_text("\n".join(lines))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-dir", default="experiments/experiment-c10-vit-lejepa-v4-short-lr1e4-sig005")
    ap.add_argument("--register-dir", default="experiments/experiment-c10-vit-lejepa-v4-short-lr1e4-sig005-reg4")
    ap.add_argument("--baseline-label", default="baseline")
    ap.add_argument("--register-label", default="registers")
    ap.add_argument("--which", default="predicted", choices=["predicted", "true"])
    ap.add_argument("--baseline-csv", default=None)
    ap.add_argument("--register-csv", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    base_csv = Path(args.baseline_csv) if args.baseline_csv else discover(Path(args.baseline_dir) / "metrics", args.which)
    reg_csv = Path(args.register_csv) if args.register_csv else discover(Path(args.register_dir) / "metrics", args.which)
    if not base_csv or not reg_csv:
        raise SystemExit(f"Could not find matching *_{args.which}.csv in both experiments.")

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.register_dir) / "presentation"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"baseline: {base_csv}\nregister: {reg_csv}\nout: {out_dir}", flush=True)

    merged, metrics, layers = load_paired(base_csv, reg_csv)
    stats_by_metric = {m: per_layer_stats(merged, m, layers) for m in metrics}

    # figures
    if "lsas" in stats_by_metric:
        fig_lsas(stats_by_metric["lsas"], layers, args.baseline_label, args.register_label,
                 out_dir / "fig1_lsas_by_layer.png")
    fig_decomposition(stats_by_metric, layers, out_dir / "fig2_decomposition.png")
    base_raw = pd.read_csv(base_csv)
    if fig_controls(base_raw, layers, args.baseline_label, out_dir / "fig3_controls.png"):
        print("  wrote fig3_controls.png (baseline had control columns)")

    # tables
    if "lsas" in stats_by_metric:
        lsas_table(stats_by_metric["lsas"], args.baseline_label, args.register_label, out_dir)
    accuracy_table(args.baseline_dir, args.register_dir, args.baseline_label, args.register_label, out_dir)

    takeaway = write_takeaway(stats_by_metric, layers, args.baseline_label, args.register_label, out_dir)
    print("\n" + takeaway)
    print(f"\nAll assets written to: {out_dir}")
    for p in sorted(out_dir.glob("*")):
        print("  ", p.name)


if __name__ == "__main__":
    main()
