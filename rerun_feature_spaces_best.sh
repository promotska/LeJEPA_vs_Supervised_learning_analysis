#!/usr/bin/env bash
set -euo pipefail

# Auto-generated feature-space reruns with FEATURE_CHECKPOINT=best.
# Review before running.

FEATURE_CHECKPOINT=best bash jobs/submit_experiment.sh "experiment-c10-r18-lejepa" feature_spaces "lejepa" "configs/base/cifar10_resnet18_lejepa.yaml"
FEATURE_CHECKPOINT=best bash jobs/submit_experiment.sh "experiment-c10-r18-lejepa-light-config" feature_spaces "lejepa" "configs/base/cifar10_resnet18_lejepa.yaml"
FEATURE_CHECKPOINT=best bash jobs/submit_experiment.sh "experiment-c10-r18-lejepa-light-v3" feature_spaces "lejepa" "configs/base/cifar10_resnet18_lejepa.yaml"
FEATURE_CHECKPOINT=best bash jobs/submit_experiment.sh "experiment-c10-r18-lejepa-light-v3-sig-005" feature_spaces "lejepa" "configs/base/cifar10_resnet18_lejepa.yaml"
FEATURE_CHECKPOINT=best bash jobs/submit_experiment.sh "experiment-c10-r18-lejepa-v3-sigreg-003" feature_spaces "lejepa" "configs/cifar10_resnet18_lejepa_v3_sigreg_003.yaml"
FEATURE_CHECKPOINT=best bash jobs/submit_experiment.sh "experiment-c10-r18-lejepa-v4-local-crops" feature_spaces "lejepa" "configs/cifar10_resnet18_lejepa_v4_local_crops.yaml"
FEATURE_CHECKPOINT=best bash jobs/submit_experiment.sh "experiment-c10-r18-lejepa-v4-proj-512" feature_spaces "lejepa" "configs/cifar10_resnet18_lejepa_v4_proj_512.yaml"
FEATURE_CHECKPOINT=best bash jobs/submit_experiment.sh "experiment-c10-r18-lejepa-v5-sigreg-002" feature_spaces "lejepa" "configs/cifar10_resnet18_lejepa_v5_sigreg_002.yaml"
FEATURE_CHECKPOINT=best bash jobs/submit_experiment.sh "experiment-c10-vit-lejepa" feature_spaces "vit_lejepa" "configs/base/cifar10_vit_lejepa.yaml"
FEATURE_CHECKPOINT=best bash jobs/submit_experiment.sh "experiment-c10-vit-lejepa-light-config" feature_spaces "vit_lejepa" "configs/base/cifar10_vit_lejepa.yaml"
FEATURE_CHECKPOINT=best bash jobs/submit_experiment.sh "experiment-c10-vit-lejepa-light-v3" feature_spaces "vit_lejepa" "configs/base/cifar10_vit_lejepa.yaml"
FEATURE_CHECKPOINT=best bash jobs/submit_experiment.sh "experiment-c10-vit-lejepa-v4-short-lr1e4-sig005" feature_spaces "vit_lejepa" "configs/cifar10_vit_lejepa_v4_short_lr1e4_sig005.yaml"
FEATURE_CHECKPOINT=best bash jobs/submit_experiment.sh "experiment-c10-vit-lejepa-v4-short-lr2e4" feature_spaces "vit_lejepa" "configs/cifar10_vit_lejepa_v4_short_lr2e4.yaml"
FEATURE_CHECKPOINT=best bash jobs/submit_experiment.sh "experiment-c10-vit-lejepa-v4-short-lr2e4-sig005" feature_spaces "vit_lejepa" "configs/cifar10_vit_lejepa_v4_short_lr2e4_sig005.yaml"
FEATURE_CHECKPOINT=best bash jobs/submit_experiment.sh "experiment-c10-vit-lejepa-v4-short-lr2e4-sig005-proj512" feature_spaces "vit_lejepa" "configs/cifar10_vit_lejepa_v4_short_lr2e4_sig005_proj512.yaml"
FEATURE_CHECKPOINT=best bash jobs/submit_experiment.sh "experiment-c10-vit-lejepa-v5-short-lr15e4-sig005" feature_spaces "vit_lejepa" "configs/cifar10_vit_lejepa_v5_short_lr15e4_sig005.yaml"
