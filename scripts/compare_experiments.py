from __future__ import annotations

"""
Compare two alignment experiments layer-by-layer (e.g. registers vs no-registers,
or LeJEPA vs supervised).

Pairs LSAS rows on (sample_idx, layer) — valid because compared runs share the
same seed and data order, so sample_idx is the same image in each. Paired
differences are far more sensitive than comparing two means.

Handled automatically:
  * matches LSAS CSVs by their <method>_<target> suffix, so e.g. a *_vit_lejepa_*
    file lines up with a *_vit_supervised_* file (different stems, same suffix);
  * picks up any metric column present in both arms, including nmi_pca_xai;
  * paired bootstrap 95% CI, Cohen's d, Wilcoxon p, and Benjamini-Hochberg FDR
    across the layer x metric family;
  * if the center/shuffle control columns are present (compute_baselines=true at
    eval time), adds a control panel: matched vs shuffled vs center.

Examples:
  # registers vs baseline (auto-discovers matching files)
  python scripts/compare_experiments.py \
    --baseline-dir experiments/experiment-c10-vit-lejepa-v4-short-lr1e4-sig005 \
    --register-dir experiments/experiment-c10-vit-lejepa-v4-short-lr1e4-sig005-reg4

  # LeJEPA vs supervised, explicit files
  python scripts/compare_experiments.py \
    --baseline-csv experiments/<sup>/metrics/stage2_vit_supervised_lsas_token_gradient_predicted.csv \
    --register-csv experiments/<lejepa>/metrics/stage2_vit_lejepa_lsas_token_gradient_predicted.csv \
    --baseline-label supervised --register-label lejepa
"""

import argparse
import re
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

CANDIDATE_METRICS = ["lsas", "corr_pca_xai", "soft_iou_pca_xai", "nmi_pca_xai"]
BASELINE_COLS = ["lsas_shuffled", "lsas_pca_center", "lsas_xai_center"]


# --------------------------------------------------------------------------- #
# discovery / matching
# --------------------------------------------------------------------------- #
def _match_key(name: str) -> str:
    """Key a CSV by whatever follows the last 'lsas_' — i.e. <method>_<target>.csv.

    So stage2_vit_lejepa_lsas_token_gradient_predicted.csv and
    stage2_vit_supervised_lsas_token_gradient_predicted.csv share the key
    'token_gradient_predicted.csv' and get matched across configs.
    """
    low = name.lower()
    idx = low.rfind("lsas_")
    return low[idx + len("lsas_"):] if idx >= 0 else low


def find_lsas_csvs(metrics_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in sorted(metrics_dir.glob("*.csv")):
        name = p.name.lower()
        if "lei" in name or name.startswith("representation"):
            continue
        if name.endswith("_predicted.csv") or name.endswith("_true.csv"):
            out[_match_key(name)] = p
    return out


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #
def bootstrap_ci(diffs: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    if diffs.size < 2:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n_boot, diffs.size))
    means = diffs[idx].mean(axis=1)
    return float(diffs.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cohens_d_paired(diffs: np.ndarray) -> float:
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    sd = diffs.std(ddof=1)
    return float(diffs.mean() / sd) if sd > 0 else float("nan")


def benjamini_hochberg(pvals: list[float], alpha: float = 0.05):
    """Return (adjusted_p, reject) lists in the original order; NaNs pass through."""
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    adj = np.full_like(p, np.nan)
    reject = np.zeros(p.shape, dtype=bool)
    if ok.sum() == 0:
        return adj.tolist(), reject.tolist()
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    prev = 1.0
    for rank, j in enumerate(reversed(order)):  # step-up from largest p
        k = m - rank
        val = min(prev, p[j] * m / k)
        adj[j] = val
        prev = val
    reject[order] = adj[order] <= alpha
    return adj.tolist(), reject.tolist()


def active_metrics(base: pd.DataFrame, reg: pd.DataFrame) -> list[str]:
    return [m for m in CANDIDATE_METRICS if m in base.columns and m in reg.columns]


def analyze_metric_file(base_csv, reg_csv, base_label, reg_label, out_dir):
    base = pd.read_csv(base_csv)
    reg = pd.read_csv(reg_csv)
    metrics = active_metrics(base, reg)
    if not metrics:
        print(f"  [skip] no shared metric columns between {base_csv.name} and {reg_csv.name}")
        return None

    keys = ["sample_idx", "layer"]
    tl = ["true_label"] if "true_label" in base.columns and "true_label" in reg.columns else []
    merged = base[keys + metrics + tl].merge(reg[keys + metrics + tl], on=keys, suffixes=("_base", "_reg"))
    n_paired = len(merged)

    paired = n_paired > 0
    if paired and tl:
        mismatch = float((merged["true_label_base"] != merged["true_label_reg"]).mean())
        if mismatch > 0.02:
            print(f"  WARNING [{base_csv.name}]: {mismatch:.1%} label mismatch across arms -> UNPAIRED.")
            paired = False

    layers = sorted(set(base["layer"]) | set(reg["layer"]), key=layer_sort_key)
    rows, pcollect = [], []  # pcollect: (row_index, metric, p) for BH
    for layer in layers:
        row = {"layer": layer}
        b_l, r_l = base[base["layer"] == layer], reg[reg["layer"] == layer]
        for m in metrics:
            row[f"{m}_{base_label}_mean"] = float(b_l[m].mean())
            row[f"{m}_{reg_label}_mean"] = float(r_l[m].mean())
            row[f"{m}_{base_label}_std"] = float(b_l[m].std())
            row[f"{m}_{reg_label}_std"] = float(r_l[m].std())
            if paired:
                ml = merged[merged["layer"] == layer]
                diffs = (ml[f"{m}_reg"] - ml[f"{m}_base"]).to_numpy()
                mean_d, lo, hi = bootstrap_ci(diffs)
                row[f"{m}_diff_mean"], row[f"{m}_diff_ci95_low"], row[f"{m}_diff_ci95_high"] = mean_d, lo, hi
                row[f"{m}_cohens_d"] = cohens_d_paired(diffs)
                row[f"{m}_ci_excludes_0"] = bool(lo > 0 or hi < 0) if not np.isnan(lo) else False
                p = float("nan")
                if _HAVE_SCIPY and np.count_nonzero(diffs) > 0 and diffs.size >= 10:
                    try:
                        p = float(wilcoxon(diffs).pvalue)
                    except Exception:
                        p = float("nan")
                row[f"{m}_wilcoxon_p"] = p
                pcollect.append((len(rows), m, p))
            else:
                row[f"{m}_diff_mean"] = float(r_l[m].mean() - b_l[m].mean())
        rows.append(row)

    summary = pd.DataFrame(rows)

    # Benjamini-Hochberg across the whole layer x metric family for this file.
    if paired and pcollect:
        adj, rej = benjamini_hochberg([p for _, _, p in pcollect])
        for (ri, m, _), pa, rj in zip(pcollect, adj, rej):
            summary.loc[ri, f"{m}_p_fdr"] = pa
            summary.loc[ri, f"{m}_significant_fdr"] = bool(rj)

    stem = base_csv.stem
    summary.to_csv(out_dir / f"{stem}_layer_summary.csv", index=False)
    _plot_metric_file(summary, layers, metrics, base_label, reg_label, paired, out_dir / f"{stem}_curve.png", stem)
    summary.attrs["meta"] = {"paired": paired, "metrics": metrics, "n_paired": n_paired,
                             "n_base": len(base), "n_reg": len(reg)}

    # center/shuffle control panel, if those columns exist in either arm.
    for df, lab in [(base, base_label), (reg, reg_label)]:
        if any(c in df.columns for c in BASELINE_COLS):
            _plot_baseline_panel(df, layers, lab, out_dir / f"{stem}_baseline_{lab}.png")
    return summary


def _plot_metric_file(summary, layers, metrics, base_label, reg_label, paired, out_path, title):
    x = np.arange(len(layers))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    fig, axes = plt.subplots(1, 2 if paired else 1, figsize=(7 * (2 if paired else 1), 4.4), squeeze=False)
    ax = axes[0][0]
    for m, c in zip(metrics, colors):
        ax.errorbar(x - 0.04, summary[f"{m}_{base_label}_mean"], yerr=summary[f"{m}_{base_label}_std"],
                    marker="o", ls="--", capsize=3, color=c, alpha=0.7, label=f"{m}·{base_label}")
        ax.errorbar(x + 0.04, summary[f"{m}_{reg_label}_mean"], yerr=summary[f"{m}_{reg_label}_std"],
                    marker="s", ls="-", capsize=3, color=c, label=f"{m}·{reg_label}")
    ax.set_xticks(x); ax.set_xticklabels(layers, rotation=30, ha="right")
    ax.set_ylabel("metric value"); ax.set_title(f"{title}\nper-layer means (±std)")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    if paired:
        ax2 = axes[0][1]
        for m, c in zip(metrics, colors):
            md = summary[f"{m}_diff_mean"].to_numpy()
            lo = summary[f"{m}_diff_ci95_low"].to_numpy()
            hi = summary[f"{m}_diff_ci95_high"].to_numpy()
            ax2.errorbar(x, md, yerr=[md - lo, hi - md], marker="D", capsize=4, color=c, label=m)
        ax2.axhline(0.0, color="k", lw=1)
        ax2.set_xticks(x); ax2.set_xticklabels(layers, rotation=30, ha="right")
        ax2.set_ylabel(f"paired diff ({reg_label} − {base_label})")
        ax2.set_title("paired difference (95% CI); >0 favors register/2nd arm")
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=160); plt.close(fig)


def _plot_baseline_panel(df, layers, label, out_path):
    """Bar chart per layer: matched LSAS vs shuffled vs center — the metric-validity slide."""
    x = np.arange(len(layers)); w = 0.25
    def per_layer(col):
        return [float(df[df["layer"] == l][col].mean()) if col in df.columns else np.nan for l in layers]
    matched = per_layer("lsas")
    shuffled = per_layer("lsas_shuffled")
    center = per_layer("lsas_pca_center")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x - w, matched, w, label="matched (real)", color="#2ca02c")
    ax.bar(x, shuffled, w, label="shuffled pairs (null)", color="#7f7f7f")
    ax.bar(x + w, center, w, label="PCA vs center prior", color="#d62728", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(layers, rotation=30, ha="right")
    ax.set_ylabel("LSAS"); ax.set_title(f"Alignment vs controls — {label}\n(gap between matched and shuffled = real signal)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(out_path, dpi=160); plt.close(fig)


# --------------------------------------------------------------------------- #
# representation
# --------------------------------------------------------------------------- #
def load_representation(metrics_dir: Path):
    hits = list(metrics_dir.glob("representation_*.yaml"))
    if not hits:
        return None
    d = yaml.safe_load(hits[0].read_text())
    probe = (d.get("classifier_or_probe_accuracy") or {}).get("accuracy")
    best = (d.get("knn") or {}).get("best") or {}
    return {"probe_accuracy": probe, "best_knn_accuracy": best.get("accuracy")}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-dir", default="experiments/experiment-c10-vit-lejepa-v4-short-lr1e4-sig005")
    ap.add_argument("--register-dir", default="experiments/experiment-c10-vit-lejepa-v4-short-lr1e4-sig005-reg4")
    ap.add_argument("--baseline-label", default="no_registers")
    ap.add_argument("--register-label", default="registers_4")
    ap.add_argument("--baseline-csv", default=None, help="explicit CSV (overrides auto-discovery)")
    ap.add_argument("--register-csv", default=None, help="explicit CSV (overrides auto-discovery)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.register_dir) / "comparison_vs_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    def emit(s=""):
        print(s); lines.append(s)

    emit("=" * 80)
    emit(f"Comparison: {args.register_label} vs {args.baseline_label}")
    emit("=" * 80)

    # explicit-file mode (e.g. LeJEPA vs supervised across configs)
    if args.baseline_csv and args.register_csv:
        pairs = [(Path(args.baseline_csv), Path(args.register_csv))]
    else:
        b = find_lsas_csvs(Path(args.baseline_dir) / "metrics")
        r = find_lsas_csvs(Path(args.register_dir) / "metrics")
        shared = sorted(set(b) & set(r))
        if not shared:
            emit("\nNo matching LSAS CSVs (by <method>_<target> suffix) in both experiments.")
            emit(f"  baseline: {sorted(b)}")
            emit(f"  register: {sorted(r)}")
        pairs = [(b[k], r[k]) for k in shared]

    for base_csv, reg_csv in pairs:
        emit(f"\n### {base_csv.name}  vs  {reg_csv.name}")
        summary = analyze_metric_file(base_csv, reg_csv, args.baseline_label, args.register_label, out_dir)
        if summary is None:
            continue
        meta = summary.attrs["meta"]
        emit(f"  paired={meta['paired']}  n_base={meta['n_base']}  n_reg={meta['n_reg']}  "
             f"metrics={meta['metrics']}")
        for _, row in summary.iterrows():
            for m in meta["metrics"]:
                bm, rm = row[f"{m}_{args.baseline_label}_mean"], row[f"{m}_{args.register_label}_mean"]
                if meta["paired"]:
                    lo, hi = row[f"{m}_diff_ci95_low"], row[f"{m}_diff_ci95_high"]
                    d = row.get(f"{m}_cohens_d", float("nan"))
                    star = "*" if row.get(f"{m}_significant_fdr") else ("." if row.get(f"{m}_ci_excludes_0") else " ")
                    emit(f"  {row['layer']:>9} {m:>16}: {bm:.4f} -> {rm:.4f}  Δ={rm-bm:+.4f}  "
                         f"CI=[{lo:+.4f},{hi:+.4f}]  d={d:+.2f} {star}")
                else:
                    emit(f"  {row['layer']:>9} {m:>16}: {bm:.4f} -> {rm:.4f}  Δ={rm-bm:+.4f}")

    # representation (dir mode only)
    if not (args.baseline_csv and args.register_csv):
        rb = load_representation(Path(args.baseline_dir) / "metrics")
        rr = load_representation(Path(args.register_dir) / "metrics")
        if rb and rr:
            emit("\n### representation (probe / kNN accuracy)")
            for key in ["probe_accuracy", "best_knn_accuracy"]:
                bv, rv = rb.get(key), rr.get(key)
                d = (rv - bv) if (bv is not None and rv is not None) else None
                emit(f"  {key:>18}: {bv} -> {rv}" + (f"  Δ={d:+.4f}" if d is not None else ""))

    emit("\n" + "=" * 80)
    emit("'*' = survives Benjamini-Hochberg FDR;  '.' = raw 95% CI excludes 0;  d = Cohen's d.")
    emit(f"Outputs written to: {out_dir}")
    (out_dir / "summary.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
