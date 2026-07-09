from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyYAML is required: pip install pyyaml") from exc


DEFAULT_FINAL_EXPERIMENTS = [
    "experiment-c10-r18-sup",
    "experiment-c10-r18-lejepa-v5-sigreg-002",
    "experiment-c10-vit-sup",
    "experiment-c10-vit-lejepa-v4-short-lr1e4-sig005",
    "experiment-imagenet100-r18-sup",
    "experiment-imagenet100-r18-lejepa",
    "experiment-imagenet100-vit-sup",
    "experiment-imagenet100-vit-lejepa",
    "experiment-imagenets50-r18-sup",
    "experiment-imagenets50-r18-lejepa",
    "experiment-imagenets50-vit-sup",
    "experiment-imagenets50-vit-lejepa",
]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default


def mlp_head_params(in_dim: int, hidden_dim: int, out_dim: int, use_bn: bool = True) -> int:
    # MLPHead in your code: Linear(in, hidden) + optional BN(hidden) + Linear(hidden, out).
    total = in_dim * hidden_dim + hidden_dim
    if use_bn:
        total += 2 * hidden_dim
    total += hidden_dim * out_dim + out_dim
    return int(total)


def resnet18_backbone_params(cifar_stem: bool) -> int:
    """Exact trainable parameter count for your ResNetBackbone(model_name='resnet18') without fc."""
    total = 0
    # conv1 + bn1
    if cifar_stem:
        total += 3 * 3 * 3 * 64
    else:
        total += 7 * 7 * 3 * 64
    total += 2 * 64  # bn1 weight,bias

    def basic_block(in_ch: int, out_ch: int, stride: int) -> int:
        params = 0
        params += 3 * 3 * in_ch * out_ch  # conv1
        params += 2 * out_ch              # bn1
        params += 3 * 3 * out_ch * out_ch # conv2
        params += 2 * out_ch              # bn2
        if stride != 1 or in_ch != out_ch:
            params += 1 * 1 * in_ch * out_ch
            params += 2 * out_ch
        return int(params)

    in_ch = 64
    for out_ch, blocks, stride in [(64, 2, 1), (128, 2, 2), (256, 2, 2), (512, 2, 2)]:
        total += basic_block(in_ch, out_ch, stride)
        in_ch = out_ch
        for _ in range(1, blocks):
            total += basic_block(in_ch, out_ch, 1)
    return int(total)


def vit_backbone_params(
    image_size: int,
    patch_size: int,
    embed_dim: int,
    depth: int,
    mlp_ratio: float = 4.0,
    cls_token: bool = True,
) -> int:
    """Exact count for your custom ViTCifarBackbone and close estimate for public ViT-B/16."""
    num_patches = (image_size // patch_size) ** 2
    tokens = num_patches + (1 if cls_token else 0)
    hidden = int(embed_dim * mlp_ratio)

    total = 0
    # PatchEmbed Conv2d + bias.
    total += patch_size * patch_size * 3 * embed_dim + embed_dim
    # cls token + pos embedding.
    if cls_token:
        total += embed_dim
    total += tokens * embed_dim

    # Transformer blocks. Your code uses nn.MultiheadAttention + two LayerNorms + MLP.
    per_block = 0
    per_block += 3 * embed_dim * embed_dim + 3 * embed_dim  # in_proj_weight + in_proj_bias
    per_block += embed_dim * embed_dim + embed_dim          # out_proj
    per_block += 4 * embed_dim                              # norm1 + norm2 weight,bias
    per_block += embed_dim * hidden + hidden                # mlp fc1
    per_block += hidden * embed_dim + embed_dim             # mlp fc2
    total += depth * per_block

    # Final LayerNorm.
    total += 2 * embed_dim
    return int(total)


def classifier_params(feature_dim: int, num_classes: int) -> int:
    return int(feature_dim * num_classes + num_classes)


def infer_method(exp_name: str, cfg: dict[str, Any], public_kind: str | None = None) -> str:
    if public_kind:
        return public_kind
    text = " ".join(
        str(x).lower()
        for x in [exp_name, cfg.get("method", {}).get("name"), cfg.get("project", {}).get("name")]
        if x is not None
    )
    if "lejepa" in text:
        return "LeJEPA"
    if "supervised" in text or "sup" in text:
        return "supervised"
    return "unknown"


def infer_dataset(cfg: dict[str, Any], exp_name: str = "") -> str:
    data = cfg.get("data", {})
    dataset = str(data.get("dataset", "")).lower()
    root = str(data.get("root", "")).lower()
    text = f"{dataset} {root} {exp_name.lower()}"
    if "cifar10" in text or "cifar-10" in text:
        return "CIFAR-10"
    if "cifar100" in text or "cifar-100" in text:
        return "CIFAR-100"
    if "imagenets50" in text or "imagenet_s50" in text or "imagenet-s-50" in text:
        return "ImageNet-S-50"
    if "imagenet100" in text or "imagenet-100" in text:
        return "ImageNet-100"
    if dataset:
        return str(data.get("dataset", "unknown"))
    return "unknown"


def infer_input_size(cfg: dict[str, Any]) -> int | None:
    model = cfg.get("model", {})
    data = cfg.get("data", {})
    explicit = safe_int(model.get("image_size"), safe_int(data.get("image_size"), None))
    if explicit is not None:
        return explicit
    dataset = str(data.get("dataset", "")).lower()
    if dataset in {"cifar10", "cifar100"}:
        return 32
    return None


def infer_patch_or_stem(cfg: dict[str, Any]) -> str:
    model = cfg.get("model", {})
    arch = str(model.get("architecture", "")).lower()
    if "vit" in arch:
        return f"patch{model.get('patch_size', 16)}"
    if "resnet" in arch:
        if "cifar" in arch or infer_input_size(cfg) == 32:
            return "CIFAR stem: 3x3 stride1, no maxpool"
        return "ImageNet stem: 7x7 stride2 + maxpool"
    return "unknown"


def infer_feature_dim(cfg: dict[str, Any]) -> int | None:
    model = cfg.get("model", {})
    arch = str(model.get("architecture", "")).lower()
    if "vit" in arch:
        return safe_int(model.get("embed_dim"), None)
    if "resnet50" in arch:
        return 2048
    if "resnet" in arch:
        return 512
    return safe_int(model.get("embed_dim"), safe_int(model.get("feature_dim"), None))


def infer_num_layers(cfg: dict[str, Any]) -> str:
    model = cfg.get("model", {})
    arch = str(model.get("architecture", "")).lower()
    if "vit" in arch:
        depth = model.get("depth", 12)
        return f"{depth} transformer blocks"
    if "resnet18" in arch:
        return "ResNet-18: 8 residual blocks / 4 stages"
    if "resnet50" in arch:
        return "ResNet-50: 16 bottleneck blocks / 4 stages"
    return "unknown"


def infer_depth_numeric(cfg: dict[str, Any]) -> int | None:
    model = cfg.get("model", {})
    arch = str(model.get("architecture", "")).lower()
    if "vit" in arch:
        return safe_int(model.get("depth"), 12)
    if "resnet18" in arch:
        return 18
    if "resnet50" in arch:
        return 50
    return None


def resolve_checkpoint_path(exp_dir: Path, cfg: dict[str, Any]) -> str:
    candidates: list[str] = []
    eval_cfg = cfg.get("evaluation", {})
    ckpt_cfg = cfg.get("checkpoints", {})
    for key in ["checkpoint_path"]:
        if eval_cfg.get(key):
            candidates.append(str(eval_cfg[key]))
    for key in ["best_path", "probe_best_path", "backbone_best_path", "last_path", "probe_last_path", "backbone_last_path"]:
        if ckpt_cfg.get(key):
            candidates.append(str(ckpt_cfg[key]))
    if not candidates:
        return ""
    raw = candidates[0]
    path = Path(raw)
    if path.is_absolute():
        return str(path)
    if raw.startswith("outputs/checkpoints/"):
        return str(exp_dir / raw.replace("outputs/checkpoints/", "checkpoints/"))
    return str(exp_dir / raw)


def analytical_counts_from_cfg(cfg: dict[str, Any], method: str) -> dict[str, Any]:
    model = cfg.get("model", {})
    arch = str(model.get("architecture", "")).lower()
    num_classes = safe_int(model.get("num_classes"), 0) or 0
    method_l = method.lower()

    if "resnet18" in arch or arch == "resnet18_cifar":
        cifar_stem = "cifar" in arch or infer_input_size(cfg) == 32
        backbone = resnet18_backbone_params(cifar_stem=cifar_stem)
        cls = classifier_params(512, num_classes)
        if method_l == "lejepa":
            projection_dim = safe_int(model.get("projection_dim"), 256) or 256
            prediction_dim = safe_int(model.get("prediction_dim"), 512) or 512
            projector = mlp_head_params(512, prediction_dim, projection_dim)
            predictor = mlp_head_params(projection_dim, prediction_dim, projection_dim)
            return {
                "num_parameters": backbone + cls,  # evaluated: backbone + linear probe
                "trainable_parameters": backbone + cls,
                "backbone_parameters": backbone,
                "head_parameters": cls,
                "training_total_parameters": backbone + projector + predictor,
                "evaluated_total_parameters": backbone + cls,
                "count_source": "analytical_resnet18_lejepa",
                "count_error": "",
            }
        return {
            "num_parameters": backbone + cls,
            "trainable_parameters": backbone + cls,
            "backbone_parameters": backbone,
            "head_parameters": cls,
            "training_total_parameters": backbone + cls,
            "evaluated_total_parameters": backbone + cls,
            "count_source": "analytical_resnet18_supervised",
            "count_error": "",
        }

    if "vit" in arch:
        image_size = safe_int(model.get("image_size"), 224) or 224
        patch_size = safe_int(model.get("patch_size"), 16) or 16
        embed_dim = safe_int(model.get("embed_dim"), 768) or 768
        depth = safe_int(model.get("depth"), 12) or 12
        mlp_ratio = float(model.get("mlp_ratio", 4.0))
        backbone = vit_backbone_params(image_size, patch_size, embed_dim, depth, mlp_ratio)
        cls = classifier_params(embed_dim, num_classes)
        if method_l == "lejepa":
            projection_dim = safe_int(model.get("projection_dim"), 256) or 256
            prediction_dim = safe_int(model.get("prediction_dim"), 512) or 512
            projector = mlp_head_params(embed_dim, prediction_dim, projection_dim)
            predictor = mlp_head_params(projection_dim, prediction_dim, projection_dim)
            return {
                "num_parameters": backbone + cls,
                "trainable_parameters": backbone + cls,
                "backbone_parameters": backbone,
                "head_parameters": cls,
                "training_total_parameters": backbone + projector + predictor,
                "evaluated_total_parameters": backbone + cls,
                "count_source": "analytical_vit_lejepa",
                "count_error": "",
            }
        return {
            "num_parameters": backbone + cls,
            "trainable_parameters": backbone + cls,
            "backbone_parameters": backbone,
            "head_parameters": cls,
            "training_total_parameters": backbone + cls,
            "evaluated_total_parameters": backbone + cls,
            "count_source": "analytical_vit_supervised",
            "count_error": "",
        }

    return {
        "num_parameters": "",
        "trainable_parameters": "",
        "backbone_parameters": "",
        "head_parameters": "",
        "training_total_parameters": "",
        "evaluated_total_parameters": "",
        "count_source": "failed",
        "count_error": f"unsupported architecture: {arch}",
    }


def public_row_from_config_json(
    json_path: Path,
    experiment: str,
    dataset: str,
    method: str,
    checkpoint_label: str,
    output_hint: str = "",
) -> dict[str, Any]:
    cfg_json = load_json(json_path)
    image_size = safe_int(cfg_json.get("image_size"), 224) or 224
    patch_size = safe_int(cfg_json.get("patch_size"), 16) or 16
    embed_dim = safe_int(cfg_json.get("embed_dim"), safe_int(cfg_json.get("num_features"), 768)) or 768
    depth = safe_int(cfg_json.get("depth"), 12) or 12
    num_heads = safe_int(cfg_json.get("num_heads"), 12) or 12
    mlp_ratio = float(cfg_json.get("mlp_ratio", 4.0))
    num_classes = safe_int(cfg_json.get("num_classes"), 0 if method.lower() == "lejepa" else 1000) or 0
    backbone = vit_backbone_params(image_size, patch_size, embed_dim, depth, mlp_ratio)
    head = classifier_params(embed_dim, num_classes) if num_classes else 0
    total = backbone + head
    architecture = cfg_json.get("architecture") or cfg_json.get("architectures", ["ViT-B/16"])[0]
    if isinstance(architecture, list):
        architecture = architecture[0]
    return {
        "experiment": experiment,
        "dataset": dataset,
        "architecture": str(architecture),
        "method": method,
        "checkpoint": checkpoint_label,
        "checkpoint_exists": json_path.exists(),
        "num_parameters": total,
        "trainable_parameters": 0,
        "backbone_parameters": backbone,
        "head_parameters": head,
        "training_total_parameters": "",
        "evaluated_total_parameters": total,
        "num_layers": f"{depth} transformer blocks",
        "depth_numeric": depth,
        "input_size": image_size,
        "patch_size_or_conv_stem": f"patch{patch_size}",
        "feature_dim": embed_dim,
        "num_heads": num_heads,
        "num_classes": num_classes,
        "flops_optional": "not_computed",
        "flops_method": "not_computed",
        "count_source": f"analytical_vit_from_json:{json_path}",
        "count_error": "",
        "config_path": str(json_path),
        "notes": output_hint,
    }


def pick_config(exp_dir: Path) -> Path | None:
    config_dir = exp_dir / "configs"
    if not config_dir.exists():
        return None
    originals = sorted(config_dir.glob("original*.yaml"))
    if originals:
        return originals[0]
    resolved = sorted(config_dir.glob("resolved_*train*.yaml"))
    if resolved:
        return resolved[0]
    resolved = sorted(config_dir.glob("resolved*.yaml"))
    return resolved[0] if resolved else None


def summarize_experiment(exp_dir: Path) -> dict[str, Any] | None:
    cfg_path = pick_config(exp_dir)
    if cfg_path is None:
        return None
    cfg = load_yaml(cfg_path)
    exp_name = exp_dir.name
    method = infer_method(exp_name, cfg)
    dataset = infer_dataset(cfg, exp_name)
    model_cfg = cfg.get("model", {})
    arch = str(model_cfg.get("architecture", "unknown"))
    counts = analytical_counts_from_cfg(cfg, method)
    checkpoint = resolve_checkpoint_path(exp_dir, cfg)
    checkpoint_exists = bool(checkpoint and Path(checkpoint).exists())

    row = {
        "experiment": exp_name,
        "dataset": dataset,
        "architecture": arch,
        "method": method,
        "checkpoint": checkpoint,
        "checkpoint_exists": checkpoint_exists,
        "num_layers": infer_num_layers(cfg),
        "depth_numeric": infer_depth_numeric(cfg),
        "input_size": infer_input_size(cfg),
        "patch_size_or_conv_stem": infer_patch_or_stem(cfg),
        "feature_dim": infer_feature_dim(cfg),
        "num_heads": model_cfg.get("num_heads", ""),
        "num_classes": model_cfg.get("num_classes", ""),
        "flops_optional": "not_computed",
        "flops_method": "not_computed",
        "config_path": str(cfg_path),
        "notes": "num_parameters is evaluation-model total; backbone_parameters is best for architecture fairness; training_total_parameters includes LeJEPA projector/predictor.",
    }
    row.update(counts)
    return row


def discover_experiments(root: Path, use_default_final_set: bool = True) -> list[Path]:
    if not root.exists():
        return []
    if use_default_final_set:
        selected = [root / name for name in DEFAULT_FINAL_EXPERIMENTS if (root / name).exists()]
        if selected:
            return selected
    return sorted([p for p in root.iterdir() if p.is_dir() and (p / "configs").exists()])


def write_per_experiment(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        cfg_path = Path(str(row.get("config_path", "")))
        if not cfg_path.exists() or cfg_path.suffix != ".yaml":
            continue
        if cfg_path.parent.name != "configs":
            continue
        exp_dir = cfg_path.parents[1]
        metrics_dir = exp_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_csv(metrics_dir / "model_fairness.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize model fairness: params/layers/stem/patch size/FLOPs placeholder.")
    parser.add_argument("--experiments-root", default="experiments", help="Root directory containing experiment-* folders.")
    parser.add_argument("--experiments", nargs="*", default=None, help="Explicit experiment directories. Overrides default discovery.")
    parser.add_argument("--output", default="experiments/model_fairness.csv", help="Output CSV path.")
    parser.add_argument("--all-experiments", action="store_true", help="Use all experiment folders under root, not only the final selected set.")
    parser.add_argument("--write-per-experiment", action="store_true", help="Also write metrics/model_fairness.csv inside each experiment.")
    parser.add_argument("--public-okai-config", default=None, help="Path to OK-AI LeJEPA ViT-B/16 config.json.")
    parser.add_argument("--public-supervised-config", default=None, help="Path to timm supervised ViT-B/16 config.json.")
    parser.add_argument("--public-dataset", default="ImageNet-100", help="Dataset label for public checkpoint rows.")
    args = parser.parse_args()

    root = Path(args.experiments_root)
    if args.experiments:
        exp_dirs = [Path(x) for x in args.experiments]
    else:
        exp_dirs = discover_experiments(root, use_default_final_set=not args.all_experiments)

    rows: list[dict[str, Any]] = []
    for exp_dir in exp_dirs:
        row = summarize_experiment(exp_dir)
        if row is not None:
            rows.append(row)
        else:
            print(f"WARNING: skipped {exp_dir}: no config found")

    if args.public_okai_config:
        rows.append(
            public_row_from_config_json(
                json_path=Path(args.public_okai_config),
                experiment="public-okai-vitb16",
                dataset=args.public_dataset,
                method="LeJEPA",
                checkpoint_label="OK-AI/lejepa-vitb16-pretrain-in1k",
                output_hint="public checkpoint; parameter count estimated from config.json; no classifier head for inference-only row",
            )
        )
    if args.public_supervised_config:
        rows.append(
            public_row_from_config_json(
                json_path=Path(args.public_supervised_config),
                experiment="public-supervised-vitb16",
                dataset=args.public_dataset,
                method="supervised",
                checkpoint_label="timm/vit_base_patch16_224.augreg_in1k",
                output_hint="public checkpoint; parameter count estimated from config.json; includes ImageNet-1K classifier head",
            )
        )
    if not rows:
        raise RuntimeError("No experiment rows were found. Check --experiments-root or --experiments.")

    df = pd.DataFrame(rows)
    columns = [
        "experiment", "dataset", "architecture", "method", "checkpoint", "checkpoint_exists",
        "num_parameters", "trainable_parameters", "backbone_parameters", "head_parameters",
        "training_total_parameters", "evaluated_total_parameters", "num_layers", "depth_numeric",
        "input_size", "patch_size_or_conv_stem", "feature_dim", "num_heads", "num_classes",
        "flops_optional", "flops_method", "count_source", "count_error", "config_path", "notes",
    ]
    df = df[[c for c in columns if c in df.columns] + [c for c in df.columns if c not in columns]]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved model fairness table: {out}")
    display_cols = ["experiment", "dataset", "architecture", "method", "num_parameters", "backbone_parameters", "num_layers", "input_size", "patch_size_or_conv_stem"]
    print(df[display_cols].to_string(index=False))
    if args.write_per_experiment:
        write_per_experiment(rows)
        print("Also wrote metrics/model_fairness.csv inside each experiment directory where possible.")


if __name__ == "__main__":
    main()
