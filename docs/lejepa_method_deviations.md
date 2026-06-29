# LeJEPA method reproduction notes

This project uses a faithful LeJEPA-method reproduction rather than the old Stage 1 `SigRegStyleLoss` baseline.

## Included in the faithful setup

- JEPA predictive objective over multiple augmented views.
- Official SIGReg regularizer through the external `lejepa` package when `sigreg_implementation: official`.
- No stop-gradient on the target embeddings.
- No teacher-student branch.
- No EMA teacher.
- Official-style global/local multi-crop views.
- Linear probe and kNN evaluation of frozen representations.

## Configurable deviations

Each LeJEPA config contains a `method.deviations` field. This is written into checkpoints and manifests.
Typical deviations are:

- CIFAR experiments use 32x32 images, so global/local crop scales are adapted to small resolution.
- ImageNet-100 is used instead of ImageNet-1K for a controlled course-scale benchmark.
- Smaller backbones may be used when the goal is interpretability comparison rather than reproducing paper-scale benchmark numbers.

## Required dependency

Install the official SIGReg package in the CINECA environment:

```bash
pip install lejepa
```

For debugging only, use:

```yaml
lejepa:
  sigreg_implementation: fallback
```

Do not report fallback runs as official LeJEPA-SIGReg experiments.
