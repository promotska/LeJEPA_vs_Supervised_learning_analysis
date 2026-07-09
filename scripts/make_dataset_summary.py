from __future__ import annotations

"""
Grouped-by-dataset summary: put all models for one dataset on one figure + table.

Give it the experiment folders for a dataset (any mix of ResNet/ViT, supervised/
LeJEPA). Architecture and training type are inferred from the folder name
(vit/r18, lejepa/sup). Produces:

  fig_sas_by_depth.png   mean SAS vs relative depth, one line per model
                         (color = training, line style/marker = architecture)
  table_summary.png/.csv per model: arch, training, probe acc, kNN acc, mean SAS
  summary.txt            console digest

Ungrounded (default) reads the *_predicted.csv LSAS files and plots `lsas`.
--grounded reads the *_g_lsas.csv files (S50 with masks) and plots `g_lsas`,
adding IoU(PCA,mask) and IoU(saliency,mask) to the table.

Example:
  python scripts/make_dataset_summary.py --dataset-name "ImageNet-100" \
    --experiments \
      experiments/experiment-imagenet100-r18-sup \
      experiments/experiment-imagenet100-r18-lejepa \
      experiments/experiment-imagenet100-vit-sup \
      experiments/experiment-imagenet100-vit-lejepa \
    --out-dir experiments/_summary/in100
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
    def layer_sort_key(layer):
        parts = str(layer).replace("_", ".").split(".")
        nums = [int(p) for p in parts if p.isdigit()]
        return (0, nums[-1], str(layer)) if nums else (1, 0, str(layer))

C_SUP, C_LEJEPA = "#7f7f7f", "#0072B2"    # gray / blue by training
STYLE = {"ResNet": dict(ls="-", marker="o"), "ViT": dict(ls="--", marker="s")}


def infer(exp_dir: Path):
    name = exp_dir.name.lower()
    arch = "ViT" if "vit" in name else "ResNet"
    training = "LeJEPA" if "lejepa" in name else "supervised"
    return arch, training


def discover(metrics_dir: Path, grounded: bool, which: str):
    if grounded:
        cands = [p for p in sorted(metrics_dir.glob("*_g_lsas.csv")) if "lei" not in p.name.lower()]
        return cands[0] if cands else None
    cands = [p for p in sorted(metrics_dir.glob(f"*_{which}.csv"))
             if "lei" not in p.name.lower() and not p.name.startswith("representation")]
    tg = [p for p in cands if "token_gradient" in p.name]
    return (tg or cands)[0] if cands else None


def representation(metrics_dir: Path):
    hits = list(metrics_dir.glob("representation_*.yaml"))
    if not hits:
        return None, None
    d = yaml.safe_load(hits[0].read_text())
    probe = (d.get("classifier_or_probe_accuracy") or {}).get("accuracy")
    knn = ((d.get("knn") or {}).get("best") or {}).get("accuracy")
    return probe, knn


def per_layer_mean(df, metric, layers):
    return np.array([float(df[df["layer"] == l][metric].mean()) for l in layers])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--experiments", nargs="+", required=True)
    ap.add_argument("--which", default="predicted", choices=["predicted", "true"])
    ap.add_argument("--grounded", action="store_true", help="read *_g_lsas.csv and plot g_lsas")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    metric = "g_lsas" if args.grounded else "lsas"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"figure.facecolor": "white", "font.size": 13, "axes.titlesize": 15,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7.6, 4.8))

    rows, lines = [], []
    for e in args.experiments:
        e = Path(e); mdir = e / "metrics"
        arch, training = infer(e)
        csv = discover(mdir, args.grounded, args.which)
        if csv is None:
            print(f"  [skip] no {'g_lsas' if args.grounded else args.which} CSV in {e}")
            continue
        df = pd.read_csv(csv)
        if metric not in df.columns:
            print(f"  [skip] {csv.name} has no '{metric}' column"); continue
        layers = sorted(df["layer"].unique(), key=layer_sort_key)
        y = per_layer_mean(df, metric, layers)
        rel = np.linspace(0, 1, len(layers)) if len(layers) > 1 else np.array([0.5])
        color = C_LEJEPA if training == "LeJEPA" else C_SUP
        ax.plot(rel, y, color=color, lw=2.4, ms=8, label=f"{arch} · {training}", **STYLE[arch])

        row = {"model": f"{arch} {training}", "arch": arch, "training": training,
               f"mean {metric}": float(np.nanmean(y)), f"peak {metric}": float(np.nanmax(y))}
        probe, knn = representation(mdir)
        row["probe acc"] = probe
        row["kNN acc"] = knn
        if args.grounded:
            for c, nm in [("iou_pca_gt", "IoU(PCA,mask)"), ("iou_xai_gt", "IoU(saliency,mask)")]:
                if c in df.columns:
                    row[nm] = float(df[c].mean())
        rows.append(row)

    ax.set_xlabel("relative depth (shallow → deep)")
    ax.set_ylabel(metric.upper().replace("_", "-"))
    ax.set_title(f"{args.dataset_name}: {'grounded ' if args.grounded else ''}PCA–saliency alignment by model")
    ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=10)
    fig.tight_layout(); fig.savefig(out_dir / "fig_sas_by_depth.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # summary table
    df = pd.DataFrame(rows)
    fmt = df.copy()
    for c in fmt.columns:
        if c not in ("model", "arch", "training"):
            fmt[c] = fmt[c].map(lambda v: f"{v:.3f}" if isinstance(v, (int, float)) and v == v else "—")
    show_cols = [c for c in ["model", "probe acc", "kNN acc", f"mean {metric}", f"peak {metric}",
                             "IoU(PCA,mask)", "IoU(saliency,mask)"] if c in fmt.columns]
    fmt = fmt[show_cols]
    fmt.to_csv(out_dir / "table_summary.csv", index=False)

    figt, axt = plt.subplots(figsize=(min(2.1 * len(show_cols), 12), 0.6 * len(fmt) + 1.1))
    axt.axis("off"); axt.set_title(f"{args.dataset_name} — model summary", fontsize=15, pad=12)
    t = axt.table(cellText=fmt.values, colLabels=fmt.columns, cellLoc="center", loc="center")
    t.auto_set_font_size(False); t.set_fontsize(11); t.scale(1, 1.6)
    for j in range(len(fmt.columns)):
        c = t[0, j]; c.set_facecolor("#2c3e50"); c.set_text_props(color="white", weight="bold")
    for i, tr in enumerate(fmt["model"]):
        t[i + 1, 0].set_facecolor("#e8f0fe" if "LeJEPA" in tr else "#f2f2f2")
    figt.tight_layout(); figt.savefig(out_dir / "table_summary.png", dpi=200, bbox_inches="tight")
    plt.close(figt)

    print(fmt.to_string(index=False))
    (out_dir / "summary.txt").write_text(fmt.to_string(index=False))
    print(f"\nWrote fig_sas_by_depth.png + table_summary.(png|csv) to {out_dir}")


if __name__ == "__main__":
    main()
