# Stage pipeline patch: LSAS-LEI + ImageNet-100 + ImageNet-S-50 G-LSAS

This patch adds the missing pieces for the staged research pipeline:

- experiment-level LEI computation for existing LSAS/G-LSAS CSVs;
- recommended ImageNet-100 configs aligned with the best CIFAR ablations;
- ImageNet-S-50 in-domain training configs;
- SLURM job for LEI computation and `submit_experiment.sh` modes `lei` and `glei`.

## Expected dataset layouts

### ImageNet-100 classification

```text
datasets/imagenet100/
  train/<class_name>/*.JPEG
  val/<class_name>/*.JPEG
```

### ImageNet-S-50 classification + masks

```text
datasets/imagenet_s50/
  classification/
    train/<class_name>/*.JPEG
    val/<class_name>/*.JPEG
  masks/
    val/<class_name>/*.png
```

Training uses only `classification/train` and `classification/val` class folders.
Segmentation masks are used only during G-LSAS evaluation.

## New LEI commands

After `eval` or `eval_true`, compute LSAS-LEI:

```bash
bash jobs/submit_experiment.sh <experiment-name> lei <target> <config-path>
```

After `grounded`, compute G-LSAS-LEI if not already produced by the grounded script:

```bash
bash jobs/submit_experiment.sh <experiment-name> glei <target> <config-path>
```

The grounded evaluation script already computes G-LSAS-LEI automatically; `glei` is useful for recomputing with a different threshold:

```bash
LEI_THRESHOLD=0.25 bash jobs/submit_experiment.sh <experiment-name> glei <target> <config-path>
```

