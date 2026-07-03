# ImageNet-100 + G-LSAS patch

This patch adds a completed mask-grounded evaluation path for the project:

- ImageNet-100 supervised/LeJEPA configs for ResNet-18 and ViT-S/16-style models.
- Generic binary mask datasets:
  - `VOC2012` through `torchvision.datasets.VOCSegmentation`.
  - `folder_masks` for ImageNet-S-like image/mask folders paired by filename stem.
- G-LSAS metric:
  - PCA-XAI correlation.
  - PCA-GT IoU.
  - XAI-GT IoU.
  - PCA stability under photometric augmentation.
- G-LSAS curves and Layer Emergence Index.
- A generic grounded evaluation script for CNN/ResNet and ViT models.
- SLURM job and `submit_experiment.sh` support for `grounded` and `grounded_true` modes.

Important: ImageNet-100 itself has classification labels, not masks. Train ImageNet-100 models, then run G-LSAS on a mask dataset such as ImageNet-S or VOC. In the report, phrase this as “ImageNet-100 trained models with grounded evaluation on ImageNet-S/VOC masks.”

## Apply the patch

From your project root:

```bash
unzip imagenet100_glsas_patch.zip -d /tmp/imagenet100_glsas_patch
cp -r /tmp/imagenet100_glsas_patch/src/* src/
cp -r /tmp/imagenet100_glsas_patch/scripts/* scripts/
cp -r /tmp/imagenet100_glsas_patch/jobs/* jobs/
cp -r /tmp/imagenet100_glsas_patch/configs/* configs/
```

## Expected ImageNet-100 layout

```text
datasets/imagenet100/
  train/
    class_001/*.JPEG
    class_002/*.JPEG
  val/
    class_001/*.JPEG
    class_002/*.JPEG
```

## Expected ImageNet-S-like folder mask layout

```text
datasets/imagenet_s/
  images/val/
    class_001/example.JPEG
  masks/val/
    class_001/example.png
```

Masks are binarized as foreground where `mask > 0`, with `255` ignored by default.

For VOC, edit `grounded_evaluation` in configs:

```yaml
grounded_evaluation:
  dataset: VOC2012
  root: datasets
  split: val
  year: "2012"
```

## Training commands

```bash
bash jobs/submit_experiment.sh experiment-imagenet100-r18-sup train supervised configs/imagenet100_resnet18_supervised_glsas.yaml
bash jobs/submit_experiment.sh experiment-imagenet100-r18-lejepa train lejepa configs/imagenet100_resnet18_lejepa_glsas.yaml
bash jobs/submit_experiment.sh experiment-imagenet100-vit-sup train vit_supervised configs/imagenet100_vit_supervised_glsas.yaml
bash jobs/submit_experiment.sh experiment-imagenet100-vit-lejepa train vit_lejepa configs/imagenet100_vit_lejepa_glsas.yaml
```

## Standard LSAS after training

```bash
bash jobs/submit_experiment.sh experiment-imagenet100-r18-sup eval supervised configs/imagenet100_resnet18_supervised_glsas.yaml
bash jobs/submit_experiment.sh experiment-imagenet100-r18-lejepa eval lejepa configs/imagenet100_resnet18_lejepa_glsas.yaml
bash jobs/submit_experiment.sh experiment-imagenet100-vit-sup eval vit_supervised configs/imagenet100_vit_supervised_glsas.yaml
bash jobs/submit_experiment.sh experiment-imagenet100-vit-lejepa eval vit_lejepa configs/imagenet100_vit_lejepa_glsas.yaml
```

## G-LSAS after training

```bash
bash jobs/submit_experiment.sh experiment-imagenet100-r18-sup grounded supervised configs/imagenet100_resnet18_supervised_glsas.yaml
bash jobs/submit_experiment.sh experiment-imagenet100-r18-lejepa grounded lejepa configs/imagenet100_resnet18_lejepa_glsas.yaml
bash jobs/submit_experiment.sh experiment-imagenet100-vit-sup grounded vit_supervised configs/imagenet100_vit_supervised_glsas.yaml
bash jobs/submit_experiment.sh experiment-imagenet100-vit-lejepa grounded vit_lejepa configs/imagenet100_vit_lejepa_glsas.yaml
```

Outputs are saved to the experiment metrics and figures directories:

- `*_g_lsas.csv`
- `*_g_lsas_curve.png`
- `*_g_lsas_lei.csv`
- `*_g_lsas_lei.yaml`
- qualitative visualizations under `figures/*_g_lsas/`

## Direct non-SLURM command

```bash
python scripts/evaluate_grounded_alignment.py \
  --config configs/imagenet100_resnet18_lejepa_glsas.yaml \
  --mode lejepa \
  --target predicted
```
