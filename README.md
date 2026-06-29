# Emerging Interpretability in LeJEPA vs. Supervised Learning

This repository implements a layer-wise interpretability comparison between **supervised learning** and **LeJEPA-SIGReg self-supervised learning** on CNN and ViT backbones.

The central research question is:

> Do LeJEPA-trained models produce layer-wise latent features whose PCA-derived semantic regions align better with XAI saliency maps than supervised models?

The implemented pipeline extracts intermediate layer features, converts them into PCA-based spatial semantic maps, computes XAI saliency maps, and compares them layer by layer using LSAS.

At the current code stage, the project implements **LSAS**, not full **G-LSAS**. Full G-LSAS requires ground-truth object masks, which are not part of CIFAR-10/CIFAR-100/ImageNet-100 classification datasets by default.

---

## 1. What this project compares

The project is designed around two comparison axes.

### 1.1 Training effect

This compares whether self-supervised LeJEPA training produces more spatially meaningful intermediate representations than standard supervised classification.

| Method | Meaning |
|---|---|
| Supervised | Backbone trained directly with cross-entropy classification loss. |
| LeJEPA-SIGReg | Backbone pretrained with JEPA prediction loss plus official SIGReg regularization, then evaluated through a frozen-backbone linear probe. |

### 1.2 Architecture effect

This compares whether the same pattern appears in CNNs and ViTs.

| Architecture | Implemented backbone |
|---|---|
| CNN | ResNet-18 CIFAR stem or ImageNet-style stem. |
| ViT | Small custom ViT for CIFAR and 224x224 ViT-S/16-style model for ImageNet-100. |

The intended full matrix is:

| Dataset | ResNet supervised | ResNet LeJEPA | ViT supervised | ViT LeJEPA |
|---|---:|---:|---:|---:|
| CIFAR-10 | Implemented | Implemented | Implemented | Implemented |
| CIFAR-100 | Loader support exists; configs can be added | Loader support exists; configs can be added | Loader support exists; configs can be added | Loader support exists; configs can be added |
| ImageNet-100 | Requires supervised config | Implemented config | Requires supervised config | Implemented config |

---

## 2. What is implemented

### 2.1 Training

Implemented training modes:

| Target name | Model | Training script | Description |
|---|---|---|---|
| `supervised` | ResNet supervised | `scripts/train_supervised.py` | Cross-entropy training using config-driven model factory. |
| `lejepa` | ResNet LeJEPA | `scripts/train_lejepa.py` | LeJEPA pretraining plus linear probe. |
| `vit_supervised` | ViT supervised | `scripts/train_vit_supervised.py` | Cross-entropy training for ViT. |
| `vit_lejepa` | ViT LeJEPA | `scripts/train_vit_lejepa.py` | ViT LeJEPA pretraining plus linear probe. |

### 2.2 LeJEPA-SIGReg

The LeJEPA implementation uses:

- multi-view global/local crops;
- JEPA-style prediction between projected view embeddings;
- official SIGReg through the external `lejepa` package;
- no stop-gradient;
- no teacher network;
- no EMA teacher;
- frozen-backbone linear probe after pretraining.

The old `SigRegStyleLoss` is still present only for backward compatibility with earlier experiments. It should not be used for final LeJEPA-SIGReg scientific runs.

### 2.3 Evaluation

Implemented evaluation modes:

| Command mode | Meaning | Output |
|---|---|---|
| `repr` | Representation evaluation | Classifier/probe accuracy and kNN accuracy. |
| `eval` | Predicted-class LSAS | PCA map compared with XAI map targeted to predicted class. |
| `eval_true` | True-class LSAS | PCA map compared with XAI map targeted to ground-truth class label. |

Important: `eval_true` does **not** mean ground-truth segmentation mask evaluation. It only changes the class used as the XAI target from predicted label to true class label.

### 2.4 XAI methods

| Architecture | XAI method | Implementation |
|---|---|---|
| ResNet | Grad-CAM | `src/xai/gradcam.py` |
| ViT | Token-gradient saliency | `src/xai/vit_saliency.py` |
| ViT | Attention Rollout | `src/xai/attention_rollout.py` |

For ViTs, the method is controlled by:

```yaml
evaluation:
  vit_xai_method: attention_rollout   # options: token_gradient, attention_rollout
```

Attention Rollout is not class-specific. Therefore, `eval` and `eval_true` are less conceptually different for Attention Rollout than they are for Grad-CAM or token-gradient saliency.

### 2.5 LSAS

Implemented LSAS is:

```text
LSAS_l = 0.5 * Corr(PCA_l, XAI_l) + 0.5 * SoftIoU(PCA_l, XAI_l)
```

It measures agreement between:

- PCA-derived semantic map from intermediate features;
- XAI saliency map from the same layer.

Implemented in:

```text
src/evaluation/sas.py
```

### 2.6 G-LSAS status

The code contains a `g_lsas` function skeleton, but full G-LSAS is not yet an end-to-end implemented experiment because it requires ground-truth object masks.

The intended full G-LSAS formula is:

```text
G-LSAS_l =
  0.35 * Corr(PCA_l, XAI_l)
+ 0.25 * IoU(PCA_l, GT)
+ 0.25 * IoU(XAI_l, GT)
+ 0.15 * Stability(PCA_l, PCA_l_aug)
```

Missing for full G-LSAS:

- mask dataset loader;
- mask resizing/alignment;
- PCA-to-mask IoU;
- XAI-to-mask IoU;
- augmented-view PCA stability;
- G-LSAS aggregation and plotting.

---

## 3. Repository structure

```text
configs/                         YAML experiment configs
jobs/                            SLURM scripts for CINECA
scripts/                         CLI entry points
src/data/                        dataset loading and transforms
src/evaluation/                  LSAS / G-LSAS metric functions
src/features/                    feature hooks and PCA maps
src/networks/                    ResNet, ViT, heads, model factory
src/training/                    training loops, losses, checkpointing
src/visualization/               plots and qualitative map grids
src/xai/                         Grad-CAM, ViT saliency, Attention Rollout
docs/                            method notes and deviations
experiments/                     generated experiment outputs
logs/                            local/SLURM logs
```

---

## 4. Environment setup

### 4.1 Create and activate virtual environment

On CINECA, from the project directory:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the normal project dependencies according to your cluster setup. The code expects at least:

```bash
pip install torch torchvision pandas pyyaml tqdm matplotlib scikit-learn pillow numpy
```

If you use Hugging Face saved CIFAR datasets:

```bash
pip install datasets
```

### 4.2 Install official LeJEPA package

`lejepa` is not published as a normal PyPI package in the usual way. Install it from GitHub:

```bash
pip install "git+https://github.com/rbalestr-lab/lejepa.git"
```

Then test import on a compute node, not only on the login node:

```bash
python - <<'PY'
import torch
import lejepa
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("lejepa:", lejepa.__file__)
PY
```

On CINECA, first import can be slow. The observed startup cost can be around one minute for `torch` plus `lejepa`.

### 4.3 SIGReg wrapper test

The repository includes:

```text
jobs/test_sigreg_wrapper.slurm
```

Run:

```bash
sbatch jobs/test_sigreg_wrapper.slurm
```

Check logs:

```bash
tail -f logs/test-sigreg_<JOBID>.out
tail -f logs/test-sigreg_<JOBID>.err
```

Expected successful output includes:

```text
created OfficialSIGRegLoss ...
loss: tensor(...)
OK
```

---

## 5. Datasets

### 5.1 CIFAR-10 and CIFAR-100

The dataloader supports:

```yaml
data:
  dataset: CIFAR10
  root: datasets
```

and:

```yaml
data:
  dataset: CIFAR100
  root: datasets
```

The code uses `download=False`, so datasets must already be available on disk.

For torchvision CIFAR, expected structure is the usual torchvision cache under:

```text
datasets/
```

The dataloader also supports Hugging Face saved datasets if these directories exist:

```text
datasets/hf_cifar10/
datasets/hf_cifar100/
```

### 5.2 ImageNet-100 / ImageFolder

For ImageNet-100-style experiments, configs use:

```yaml
data:
  dataset: ImageFolder
  root: datasets/imagenet100
  image_size: 224
  resize_size: 256
```

Expected directory structure:

```text
datasets/imagenet100/
  train/
    class_000/
      image1.jpg
      image2.jpg
    class_001/
      ...
  val/
    class_000/
      image1.jpg
    class_001/
      ...
```

If `val/` does not exist, the loader tries `test/` for evaluation.

---

## 6. Configuration files

Current configs:

| Config | Purpose |
|---|---|
| `configs/cifar10_resnet18_supervised.yaml` | CIFAR-10 ResNet-18 supervised baseline. |
| `configs/cifar10_resnet18_lejepa.yaml` | CIFAR-10 ResNet-18 LeJEPA-SIGReg. |
| `configs/cifar10_vit_supervised.yaml` | CIFAR-10 ViT supervised baseline. |
| `configs/cifar10_vit_lejepa.yaml` | CIFAR-10 ViT LeJEPA-SIGReg. |
| `configs/imagenet100_resnet18_lejepa.yaml` | ImageNet-100 ResNet-18 LeJEPA-SIGReg at 224x224. |
| `configs/imagenet100_vit_lejepa.yaml` | ImageNet-100 ViT LeJEPA-SIGReg at 224x224. |

Important fields:

```yaml
project:
  seed: 42

experiment:
  root: experiments
  name: experiment-name
  relocate_checkpoint_path: true

data:
  dataset: CIFAR10
  root: datasets
  batch_size: 128
  val_fraction: 0.1

model:
  architecture: resnet18_cifar
  num_classes: 10

training:
  epochs: 20
  learning_rate: 0.001
  weight_decay: 0.0005
  amp: true

checkpoints:
  best_path: outputs/checkpoints/model_best.pt
  last_path: outputs/checkpoints/model_last.pt

evaluation:
  checkpoint_path: outputs/checkpoints/model_best.pt
  output_csv: outputs/metrics/model_lsas.csv
  figure_dir: outputs/figures/model
```

When `relocate_checkpoint_path: true`, paths under `outputs/` are rewritten into:

```text
experiments/<experiment-name>/checkpoints/
experiments/<experiment-name>/metrics/
experiments/<experiment-name>/figures/
```

This keeps every experiment self-contained.

---

## 7. Main command interface

Use:

```bash
bash jobs/submit_experiment.sh <experiment-name> <mode> <target> [config-path]
```

Arguments:

| Argument | Meaning |
|---|---|
| `<experiment-name>` | Output folder under `experiments/`. |
| `<mode>` | One of `train`, `repr`, `eval`, `eval_true`. |
| `<target>` | One of `supervised`, `lejepa`, `vit_supervised`, `vit_lejepa`, `both`. |
| `[config-path]` | Optional YAML config override. Strongly recommended. |

### 7.1 Modes

| Mode | Meaning |
|---|---|
| `train` | Train the model. For LeJEPA, this means pretrain backbone + train linear probe. |
| `repr` | Evaluate representation quality: classifier/probe accuracy + kNN. |
| `eval` | Compute LSAS with predicted class as XAI target. |
| `eval_true` | Compute LSAS with true class label as XAI target. |

### 7.2 Targets

| Target | Use for |
|---|---|
| `supervised` | ResNet supervised. |
| `lejepa` | ResNet LeJEPA-SIGReg. |
| `vit_supervised` | ViT supervised. |
| `vit_lejepa` | ViT LeJEPA-SIGReg. |
| `both` | ResNet supervised + ResNet LeJEPA only. Does not include ViTs. |

---

## 8. Stage 3: CIFAR-10 full 2x2 experiment

Stage 3 compares:

| Architecture | Supervised | LeJEPA-SIGReg |
|---|---:|---:|
| ResNet-18 | yes | yes |
| ViT | yes | yes |

Each model should have four jobs:

```text
train
repr
eval
eval_true
```

### 8.1 ResNet supervised

Train:

```bash
bash jobs/submit_experiment.sh experiment-c10-r18-sup train supervised configs/cifar10_resnet18_supervised.yaml
```

Evaluate representation:

```bash
bash jobs/submit_experiment.sh experiment-c10-r18-sup repr supervised configs/cifar10_resnet18_supervised.yaml
```

Evaluate predicted-class LSAS:

```bash
bash jobs/submit_experiment.sh experiment-c10-r18-sup eval supervised configs/cifar10_resnet18_supervised.yaml
```

Evaluate true-class LSAS:

```bash
bash jobs/submit_experiment.sh experiment-c10-r18-sup eval_true supervised configs/cifar10_resnet18_supervised.yaml
```

### 8.2 ResNet LeJEPA-SIGReg

Train:

```bash
bash jobs/submit_experiment.sh experiment-c10-r18-lejepa train lejepa configs/cifar10_resnet18_lejepa.yaml
```

Evaluate representation:

```bash
bash jobs/submit_experiment.sh experiment-c10-r18-lejepa repr lejepa configs/cifar10_resnet18_lejepa.yaml
```

Evaluate predicted-class LSAS:

```bash
bash jobs/submit_experiment.sh experiment-c10-r18-lejepa eval lejepa configs/cifar10_resnet18_lejepa.yaml
```

Evaluate true-class LSAS:

```bash
bash jobs/submit_experiment.sh experiment-c10-r18-lejepa eval_true lejepa configs/cifar10_resnet18_lejepa.yaml
```

### 8.3 ViT supervised

Train:

```bash
bash jobs/submit_experiment.sh experiment-c10-vit-sup train vit_supervised configs/cifar10_vit_supervised.yaml
```

Evaluate representation:

```bash
bash jobs/submit_experiment.sh experiment-c10-vit-sup repr vit_supervised configs/cifar10_vit_supervised.yaml
```

Evaluate predicted-class LSAS:

```bash
bash jobs/submit_experiment.sh experiment-c10-vit-sup eval vit_supervised configs/cifar10_vit_supervised.yaml
```

Evaluate true-class LSAS:

```bash
bash jobs/submit_experiment.sh experiment-c10-vit-sup eval_true vit_supervised configs/cifar10_vit_supervised.yaml
```

### 8.4 ViT LeJEPA-SIGReg

Train:

```bash
bash jobs/submit_experiment.sh experiment-c10-vit-lejepa train vit_lejepa configs/cifar10_vit_lejepa.yaml
```

Evaluate representation:

```bash
bash jobs/submit_experiment.sh experiment-c10-vit-lejepa repr vit_lejepa configs/cifar10_vit_lejepa.yaml
```

Evaluate predicted-class LSAS:

```bash
bash jobs/submit_experiment.sh experiment-c10-vit-lejepa eval vit_lejepa configs/cifar10_vit_lejepa.yaml
```

Evaluate true-class LSAS:

```bash
bash jobs/submit_experiment.sh experiment-c10-vit-lejepa eval_true vit_lejepa configs/cifar10_vit_lejepa.yaml
```

---

## 9. Stage 5: ImageNet-100 LeJEPA commands

The current repository includes ImageNet-100 LeJEPA configs, but not supervised ImageNet-100 configs.

### 9.1 ResNet-18 ImageNet-100 LeJEPA

```bash
bash jobs/submit_experiment.sh experiment-in100-r18-lejepa train lejepa configs/imagenet100_resnet18_lejepa.yaml
bash jobs/submit_experiment.sh experiment-in100-r18-lejepa repr lejepa configs/imagenet100_resnet18_lejepa.yaml
bash jobs/submit_experiment.sh experiment-in100-r18-lejepa eval lejepa configs/imagenet100_resnet18_lejepa.yaml
bash jobs/submit_experiment.sh experiment-in100-r18-lejepa eval_true lejepa configs/imagenet100_resnet18_lejepa.yaml
```

### 9.2 ViT ImageNet-100 LeJEPA

```bash
bash jobs/submit_experiment.sh experiment-in100-vit-lejepa train vit_lejepa configs/imagenet100_vit_lejepa.yaml
bash jobs/submit_experiment.sh experiment-in100-vit-lejepa repr vit_lejepa configs/imagenet100_vit_lejepa.yaml
bash jobs/submit_experiment.sh experiment-in100-vit-lejepa eval vit_lejepa configs/imagenet100_vit_lejepa.yaml
bash jobs/submit_experiment.sh experiment-in100-vit-lejepa eval_true vit_lejepa configs/imagenet100_vit_lejepa.yaml
```

For complete Stage 5, add:

```text
configs/imagenet100_resnet18_supervised.yaml
configs/imagenet100_vit_supervised.yaml
```

---

## 10. Direct Python commands

The SLURM wrapper is preferred on CINECA, but scripts can also be run directly.

Train ResNet supervised:

```bash
python -u scripts/train_supervised.py --config configs/cifar10_resnet18_supervised.yaml
```

Train ResNet LeJEPA:

```bash
python -u scripts/train_lejepa.py --config configs/cifar10_resnet18_lejepa.yaml
```

Train ViT supervised:

```bash
python -u scripts/train_vit_supervised.py --config configs/cifar10_vit_supervised.yaml
```

Train ViT LeJEPA:

```bash
python -u scripts/train_vit_lejepa.py --config configs/cifar10_vit_lejepa.yaml
```

Evaluate ResNet LSAS:

```bash
python -u scripts/evaluate_alignment.py \
  --config configs/cifar10_resnet18_supervised.yaml \
  --mode supervised \
  --gradcam-target predicted
```

Evaluate ViT LSAS:

```bash
python -u scripts/evaluate_vit_alignment.py \
  --config configs/cifar10_vit_supervised.yaml \
  --mode vit_supervised \
  --gradcam-target predicted \
  --vit-xai-method attention_rollout
```

Evaluate representation:

```bash
python -u scripts/evaluate_representation.py \
  --config configs/cifar10_resnet18_supervised.yaml \
  --mode supervised
```

---

## 11. Output structure

Each experiment writes into:

```text
experiments/<experiment-name>/
```

Typical structure:

```text
experiments/experiment-c10-r18-sup/
  configs/
    original_*.yaml
    resolved_*.yaml
  checkpoints/
    resnet18_supervised_best.pt
    resnet18_supervised_last.pt
  metrics/
    representation_supervised.yaml
    representation_supervised.csv
    stage1_supervised_lsas_predicted.csv
    stage1_supervised_lsas_predicted_curve.png
    stage1_supervised_lsas_true.csv
    stage1_supervised_lsas_true_curve.png
  figures/
    supervised/
      target_predicted/
      target_true/
  logs/
    train_supervised_<JOBID>.out
    train_supervised_<JOBID>.err
    repr_supervised_<JOBID>.out
    repr_supervised_<JOBID>.err
  manifest.yaml
```

The manifest records:

- git commit;
- dirty/clean repository state;
- Python and CUDA environment;
- resolved config path;
- produced artifacts;
- run status.

---

## 12. Expected evaluation outputs

### 12.1 `repr`

Command:

```bash
bash jobs/submit_experiment.sh experiment-c10-r18-sup repr supervised configs/cifar10_resnet18_supervised.yaml
```

Expected metrics:

```text
experiments/experiment-c10-r18-sup/metrics/representation_supervised.yaml
experiments/experiment-c10-r18-sup/metrics/representation_supervised.csv
```

Contains:

- classifier or linear-probe accuracy;
- kNN accuracy for configured k values;
- feature dimension;
- number of bank/query samples;
- elapsed time.

### 12.2 `eval`

Command:

```bash
bash jobs/submit_experiment.sh experiment-c10-r18-sup eval supervised configs/cifar10_resnet18_supervised.yaml
```

Expected metrics:

```text
experiments/experiment-c10-r18-sup/metrics/stage1_supervised_lsas_predicted.csv
experiments/experiment-c10-r18-sup/metrics/stage1_supervised_lsas_predicted_curve.png
```

This uses predicted-class Grad-CAM for ResNet or predicted-class token saliency for ViT token-gradient.

### 12.3 `eval_true`

Command:

```bash
bash jobs/submit_experiment.sh experiment-c10-r18-sup eval_true supervised configs/cifar10_resnet18_supervised.yaml
```

Expected metrics:

```text
experiments/experiment-c10-r18-sup/metrics/stage1_supervised_lsas_true.csv
experiments/experiment-c10-r18-sup/metrics/stage1_supervised_lsas_true_curve.png
```

This uses the ground-truth class label as the XAI target.

---

## 13. Checking job status and logs

List jobs:

```bash
squeue -u $USER
```

Cancel a job:

```bash
scancel <JOBID>
```

Inspect logs:

```bash
tail -f experiments/<experiment-name>/logs/<log-file>.out
tail -f experiments/<experiment-name>/logs/<log-file>.err
```

For simple test jobs:

```bash
tail -f logs/test-sigreg_<JOBID>.out
tail -f logs/test-sigreg_<JOBID>.err
```

---

## 14. Common mistakes

### 14.1 Using the wrong target with a ViT config

Wrong:

```bash
bash jobs/submit_experiment.sh experiment-c10-vit-sup repr supervised configs/cifar10_vit_supervised.yaml
```

Correct:

```bash
bash jobs/submit_experiment.sh experiment-c10-vit-sup repr vit_supervised configs/cifar10_vit_supervised.yaml
```

Target must match model family:

| Config | Correct target |
|---|---|
| `cifar10_resnet18_supervised.yaml` | `supervised` |
| `cifar10_resnet18_lejepa.yaml` | `lejepa` |
| `cifar10_vit_supervised.yaml` | `vit_supervised` |
| `cifar10_vit_lejepa.yaml` | `vit_lejepa` |

### 14.2 Committing logs

Do not run `git add .` after SLURM jobs unless `.gitignore` is correct. Logs and experiment outputs should normally not be committed.

Recommended `.gitignore` entries:

```text
logs/*.out
logs/*.err
experiments/*/logs/*.out
experiments/*/logs/*.err
*.pt
__pycache__/
*.pyc
```

Use targeted adds:

```bash
git add src/training/losses.py configs/cifar10_resnet18_lejepa.yaml
```

not:

```bash
git add .
```

### 14.3 Testing LeJEPA import on login node

`torch` and `lejepa` imports can be slow on login nodes. Test inside SLURM if possible.

---

## 15. Git workflow on CINECA

Before pulling remote code:

```bash
git status
git fetch origin
git branch -vv
```

If your branch is behind and your working tree is clean:

```bash
git pull --ff-only origin ananas-code
```

If you accidentally committed local logs and want to discard the local commit:

```bash
git reset --hard origin/ananas-code
```

If untracked logs remain:

```bash
rm -f logs/*.out logs/*.err
```

---

## 16. Why this methodology is used

The project is not only checking classification accuracy. Accuracy says whether the final classifier predicts the correct label, but it does not show whether intermediate representations have spatial semantic structure.

The evaluation therefore checks three things:

1. **Representation quality**: accuracy and kNN show whether the learned features are useful for classification.
2. **Layer-wise semantic emergence**: PCA maps show where latent feature directions become spatially organized.
3. **PCA-XAI alignment**: LSAS checks whether unsupervised semantic regions and saliency evidence occupy similar spatial regions.

This allows comparing:

- whether LeJEPA develops semantic structure earlier or more consistently than supervised learning;
- whether CNNs and ViTs show different layer-wise emergence patterns;
- whether PCA maps and XAI maps are aligned or disconnected.

---

## 17. Current limitations

1. **Full G-LSAS is not yet implemented end-to-end.** The current pipeline computes LSAS, not mask-grounded G-LSAS.
2. **CIFAR-10/CIFAR-100 have no ground-truth object masks by default.** For G-LSAS, use a mask dataset such as ImageNet-S, PASCAL VOC, or a COCO subset.
3. **Attention Rollout is not class-targeted.** For ViT Attention Rollout, `eval` and `eval_true` do not have the same meaning as for Grad-CAM.
4. **ImageNet-100 supervised configs are not currently included.** They should be added for a complete Stage 5 2x2 matrix.
5. **CIFAR-100 configs are not currently included.** The loader supports CIFAR-100, but configs must be added.

---

## 18. Recommended next steps

### Stage 3: CIFAR-10 completion

Run the full 2x2 matrix:

```text
ResNet supervised
ResNet LeJEPA-SIGReg
ViT supervised
ViT LeJEPA-SIGReg
```

For each:

```text
train
repr
eval
eval_true
```

### Stage 4: CIFAR-100

Add configs:

```text
configs/cifar100_resnet18_supervised.yaml
configs/cifar100_resnet18_lejepa.yaml
configs/cifar100_vit_supervised.yaml
configs/cifar100_vit_lejepa.yaml
```

Then run the same 2x2 evaluation.

### Stage 5: ImageNet-100

Add supervised configs:

```text
configs/imagenet100_resnet18_supervised.yaml
configs/imagenet100_vit_supervised.yaml
```

Then compare:

```text
ImageNet-100 ResNet supervised
ImageNet-100 ResNet LeJEPA-SIGReg
ImageNet-100 ViT supervised
ImageNet-100 ViT LeJEPA-SIGReg
```

### Stage 6: G-LSAS

Add ground-truth masks and compute:

```text
PCA ↔ XAI
PCA ↔ GT mask
XAI ↔ GT mask
PCA stability under augmentation
```

---

## 19. Quick sanity checks

Compile Python files:

```bash
python -m compileall -q src scripts
```

Check shell scripts:

```bash
bash -n jobs/submit_experiment.sh
for f in jobs/*.slurm; do bash -n "$f" || echo "BAD: $f"; done
```

Check SIGReg wrapper:

```bash
sbatch jobs/test_sigreg_wrapper.slurm
```

Check produced metrics:

```bash
find experiments/experiment-c10-r18-sup/metrics -maxdepth 1 -type f | sort
```

Check LSAS row count:

```bash
python - <<'PY'
import pandas as pd

path = "experiments/experiment-c10-r18-sup/metrics/stage1_supervised_lsas_predicted.csv"
df = pd.read_csv(path)
print("rows:", len(df))
print("layers:", sorted(df["layer"].unique()))
print("mean LSAS per layer:")
print(df.groupby("layer")[["lsas", "corr_pca_xai", "soft_iou_pca_xai"]].mean())
PY
```

---

## 20. Minimal command checklist for CIFAR-10 Stage 3

```bash
# ResNet supervised
bash jobs/submit_experiment.sh experiment-c10-r18-sup train supervised configs/cifar10_resnet18_supervised.yaml
bash jobs/submit_experiment.sh experiment-c10-r18-sup repr supervised configs/cifar10_resnet18_supervised.yaml
bash jobs/submit_experiment.sh experiment-c10-r18-sup eval supervised configs/cifar10_resnet18_supervised.yaml
bash jobs/submit_experiment.sh experiment-c10-r18-sup eval_true supervised configs/cifar10_resnet18_supervised.yaml

# ResNet LeJEPA
bash jobs/submit_experiment.sh experiment-c10-r18-lejepa train lejepa configs/cifar10_resnet18_lejepa.yaml
bash jobs/submit_experiment.sh experiment-c10-r18-lejepa repr lejepa configs/cifar10_resnet18_lejepa.yaml
bash jobs/submit_experiment.sh experiment-c10-r18-lejepa eval lejepa configs/cifar10_resnet18_lejepa.yaml
bash jobs/submit_experiment.sh experiment-c10-r18-lejepa eval_true lejepa configs/cifar10_resnet18_lejepa.yaml

# ViT supervised
bash jobs/submit_experiment.sh experiment-c10-vit-sup train vit_supervised configs/cifar10_vit_supervised.yaml
bash jobs/submit_experiment.sh experiment-c10-vit-sup repr vit_supervised configs/cifar10_vit_supervised.yaml
bash jobs/submit_experiment.sh experiment-c10-vit-sup eval vit_supervised configs/cifar10_vit_supervised.yaml
bash jobs/submit_experiment.sh experiment-c10-vit-sup eval_true vit_supervised configs/cifar10_vit_supervised.yaml

# ViT LeJEPA
bash jobs/submit_experiment.sh experiment-c10-vit-lejepa train vit_lejepa configs/cifar10_vit_lejepa.yaml
bash jobs/submit_experiment.sh experiment-c10-vit-lejepa repr vit_lejepa configs/cifar10_vit_lejepa.yaml
bash jobs/submit_experiment.sh experiment-c10-vit-lejepa eval vit_lejepa configs/cifar10_vit_lejepa.yaml
bash jobs/submit_experiment.sh experiment-c10-vit-lejepa eval_true vit_lejepa configs/cifar10_vit_lejepa.yaml
```
