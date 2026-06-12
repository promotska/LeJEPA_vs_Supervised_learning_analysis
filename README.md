# Stage 1 — CIFAR-10 ResNet-18 Supervised vs LeJEPA

This is a compact Stage-1 implementation for the project:

- Dataset: CIFAR-10
- Models:
  - ResNet-18 supervised classifier
  - ResNet-18 LeJEPA-style self-supervised backbone + linear probe
- Evaluation:
  - layer-wise activation extraction
  - PCA semantic heatmaps
  - Grad-CAM saliency maps
  - LSAS = PCA ↔ Grad-CAM alignment

Important limitation: CIFAR-10 does not provide ground-truth object masks. Therefore this stage computes **LSAS**, not fully grounded **G-LSAS**. G-LSAS should be used later with a dataset that has segmentation masks.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/CINECA
pip install -r requirements.txt
```

On CINECA, create the environment according to the cluster's PyTorch/CUDA module rules, then install the Python dependencies.

## Train supervised ResNet-18

```bash
python scripts/train_supervised.py --config configs/cifar10_resnet18_supervised.yaml
```

This saves:

```text
outputs/checkpoints/resnet18_supervised_best.pt
outputs/checkpoints/resnet18_supervised_last.pt
```

## Train LeJEPA-style ResNet-18 + linear probe

```bash
python scripts/train_lejepa.py --config configs/cifar10_resnet18_lejepa.yaml
```

This saves:

```text
outputs/checkpoints/resnet18_lejepa_backbone_best.pt
outputs/checkpoints/resnet18_lejepa_probe_best.pt
```

The LeJEPA implementation here is intentionally minimal for Stage 1: symmetric joint-embedding prediction with a SIGReg-style anti-collapse regularizer. If your professor expects the exact official LeJEPA formulation, replace `src/training/losses.py` while keeping the rest of the pipeline.

## Evaluate LSAS

```bash
python scripts/evaluate_alignment.py \
  --config configs/cifar10_resnet18_supervised.yaml \
  --mode supervised

python scripts/evaluate_alignment.py \
  --config configs/cifar10_resnet18_lejepa.yaml \
  --mode lejepa
```

This saves:

```text
outputs/metrics/stage1_supervised_lsas.csv
outputs/metrics/stage1_lejepa_lsas.csv
outputs/figures/supervised/*.png
outputs/figures/lejepa/*.png
```

## Plot supervised vs LeJEPA layer curve

```bash
python scripts/make_figures.py
```

Output:

```text
outputs/figures/stage1_supervised_vs_lejepa_lsas.png
```

## What to inspect first

1. Open the qualitative figures in `outputs/figures/supervised` and `outputs/figures/lejepa`.
2. Check whether PCA and Grad-CAM maps highlight similar regions.
3. Open the CSV files and compare mean LSAS by layer.
4. Compare the final curve from `scripts/make_figures.py`.
