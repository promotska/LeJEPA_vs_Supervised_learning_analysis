from __future__ import annotations
"""Overall (not per-layer) bar comparison across our own trained models, + LEI.

Each --model is LABEL=experiment_dir. Per model:
  * overall alignment = per-sample mean across layers, then mean +/- 95% CI over
    samples (lsas / corr / soft_iou [/ nmi if present]);
  * probe & kNN accuracy from representation_*.yaml;
  * Layer Emergence Index (AUC + normalized emergence depth) from the LSAS curve.

Use for: comparing fully-trained models overall, and sigreg ablations (pass the
sigreg value as LABEL, --xlabel "SIGReg weight").
"""
import argparse, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, yaml

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
try:
    from src.evaluation.lei import compute_layer_emergence_index
except Exception:
    compute_layer_emergence_index = None

METRICS = ["lsas", "corr_pca_xai", "soft_iou_pca_xai"]
LBL = {"lsas": "LSAS", "corr_pca_xai": "correlation", "soft_iou_pca_xai": "overlap (soft-IoU)", "nmi_pca_xai": "NMI"}
MC = {"lsas": "#0072B2", "corr_pca_xai": "#009E73", "soft_iou_pca_xai": "#D55E00", "nmi_pca_xai": "#9467bd"}

def discover(md, which="predicted"):
    c = [p for p in sorted(md.glob(f"*_{which}.csv"))
         if "lei" not in p.name.lower() and not p.name.startswith("representation")]
    tg = [p for p in c if "token_gradient" in p.name]
    return (tg or c)[0] if c else None

def rep(md):
    h = list(md.glob("representation_*.yaml"))
    if not h: return None, None
    d = yaml.safe_load(h[0].read_text())
    return (d.get("classifier_or_probe_accuracy") or {}).get("accuracy"), \
           ((d.get("knn") or {}).get("best") or {}).get("accuracy")

def mean_ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if x.size < 2: return (float(x.mean()) if x.size else float("nan")), 0.0
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(x.size))

def per_sample_overall(df, metric):
    return df.groupby("sample_idx")[metric].mean().to_numpy()  # avg over layers per sample

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="LABEL=experiment_dir (repeatable)")
    ap.add_argument("--which", default="predicted", choices=["predicted", "true"])
    ap.add_argument("--title", default="Overall model comparison")
    ap.add_argument("--xlabel", default="model")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    models = []
    for spec in args.model:
        label, d = spec.split("=", 1); d = Path(d)
        csv = discover(d / "metrics", args.which)
        if csv is None:
            print(f"[skip] no {args.which} CSV in {d}"); continue
        df = pd.read_csv(csv)
        e = {"label": label, "_metrics": [m for m in METRICS + ["nmi_pca_xai"] if m in df.columns]}
        for m in e["_metrics"]:
            e[m] = mean_ci(per_sample_overall(df, m))
        e["probe"], e["knn"] = rep(d / "metrics")
        e["auc"] = e["nei"] = e["peak_layer"] = None
        if compute_layer_emergence_index is not None:
            try:
                r = compute_layer_emergence_index(csv, metric="lsas")
                e["auc"], e["nei"], e["peak_layer"] = r.get("auc"), r.get("normalized_emergence_index"), r.get("peak_layer")
            except Exception as ex:
                print(f"LEI failed for {label}: {ex}")
        models.append(e)

    labels = [m["label"] for m in models]; x = np.arange(len(models))

    # Fig 1: overall alignment (grouped bars)
    shown = [m for m in METRICS if any(m in e["_metrics"] for e in models)]
    w = 0.8 / max(len(shown), 1)
    fig, ax = plt.subplots(figsize=(1.5 * len(models) + 3, 4.6))
    for k, m in enumerate(shown):
        mu = [e.get(m, (np.nan, 0))[0] for e in models]; ci = [e.get(m, (np.nan, 0))[1] for e in models]
        ax.bar(x + k * w - 0.4 + w / 2, mu, w, yerr=ci, capsize=3, color=MC[m], label=LBL[m])
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("overall value (per-sample mean, 95% CI)"); ax.set_xlabel(args.xlabel); ax.set_title(args.title)
    ax.legend(frameon=False); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(out / "fig_overall_alignment.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # Fig 2: accuracy
    if any(m["probe"] is not None or m["knn"] is not None for m in models):
        w2 = 0.38; fig, ax = plt.subplots(figsize=(1.5 * len(models) + 3, 4.4))
        ax.bar(x - w2 / 2, [m["probe"] if m["probe"] is not None else np.nan for m in models], w2,
               color="#0072B2", label="linear probe acc.")
        ax.bar(x + w2 / 2, [m["knn"] if m["knn"] is not None else np.nan for m in models], w2,
               color="#7f7f7f", label="kNN acc.")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("accuracy"); ax.set_xlabel(args.xlabel); ax.set_title(f"{args.title}: accuracy")
        ax.legend(frameon=False); ax.grid(axis="y", alpha=0.25)
        fig.tight_layout(); fig.savefig(out / "fig_accuracy.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # Fig 3: LEI
    if any(m["auc"] is not None for m in models):
        w3 = 0.38; fig, ax = plt.subplots(figsize=(1.5 * len(models) + 3, 4.4))
        ax.bar(x - w3 / 2, [m["auc"] if m["auc"] is not None else np.nan for m in models], w3,
               color="#2ca02c", label="LEI AUC (higher = more aligned overall)")
        ax.bar(x + w3 / 2, [m["nei"] if m["nei"] is not None else np.nan for m in models], w3,
               color="#d62728", label="emergence depth (lower = earlier)")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("LEI"); ax.set_xlabel(args.xlabel); ax.set_title(f"{args.title}: Layer Emergence Index")
        ax.legend(frameon=False, fontsize=9); ax.grid(axis="y", alpha=0.25)
        fig.tight_layout(); fig.savefig(out / "fig_lei.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    rows = []
    for m in models:
        r = {"model": m["label"]}
        for me in shown:
            r[LBL[me]] = round(m.get(me, (np.nan, 0))[0], 4)
        r["probe acc"], r["kNN acc"] = m["probe"], m["knn"]
        r["LEI AUC"] = round(m["auc"], 4) if m["auc"] is not None else None
        r["emergence depth"], r["peak layer"] = m["nei"], m["peak_layer"]
        rows.append(r)
    pd.DataFrame(rows).to_csv(out / "table_overall.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nWrote fig_overall_alignment.png, fig_accuracy.png, fig_lei.png, table_overall.csv to {out}")

if __name__ == "__main__":
    main()