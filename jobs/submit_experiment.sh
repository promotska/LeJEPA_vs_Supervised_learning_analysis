#!/bin/bash
set -euo pipefail

if [ $# -lt 3 ]; then
  echo "Usage:"
  echo "  bash jobs/submit_experiment.sh <experiment-name> <mode> <target> [config-path]"
  echo ""
  echo "Modes:"
  echo "  train"
  echo "  eval"
  echo "  eval_true"
  echo "  repr"
  echo "  feature_spaces"
  echo "  grounded"
  echo "  grounded_true"
  echo ""
  echo "Targets:"
  echo "  supervised        ResNet supervised"
  echo "  lejepa            ResNet LeJEPA"
  echo "  both              ResNet supervised + ResNet LeJEPA"
  echo "  vit_supervised    ViT supervised"
  echo "  vit_lejepa        ViT LeJEPA-SIGReg"
  echo ""
  echo "Examples:"
  echo "  bash jobs/submit_experiment.sh experiment-c10-r18-sup train supervised configs/cifar10_resnet18_supervised.yaml"
  echo "  bash jobs/submit_experiment.sh experiment-c10-r18-sup repr supervised configs/cifar10_resnet18_supervised.yaml"
  echo "  bash jobs/submit_experiment.sh experiment-c10-r18-sup eval supervised configs/cifar10_resnet18_supervised.yaml"
  echo "  bash jobs/submit_experiment.sh experiment-c10-r18-sup eval_true supervised configs/cifar10_resnet18_supervised.yaml"
  echo ""
  echo "  bash jobs/submit_experiment.sh experiment-c10-r18-lejepa train lejepa configs/cifar10_resnet18_lejepa.yaml"
  echo "  bash jobs/submit_experiment.sh experiment-c10-r18-lejepa repr lejepa configs/cifar10_resnet18_lejepa.yaml"
  echo "  bash jobs/submit_experiment.sh experiment-c10-r18-lejepa eval lejepa configs/cifar10_resnet18_lejepa.yaml"
  echo "  bash jobs/submit_experiment.sh experiment-c10-r18-lejepa eval_true lejepa configs/cifar10_resnet18_lejepa.yaml"
  echo "  bash jobs/submit_experiment.sh experiment-c10-r18-lejepa feature_spaces lejepa configs/cifar10_resnet18_lejepa.yaml"
  echo ""
  echo "  bash jobs/submit_experiment.sh experiment-c10-vit-sup train vit_supervised configs/cifar10_vit_supervised.yaml"
  echo "  bash jobs/submit_experiment.sh experiment-c10-vit-sup repr vit_supervised configs/cifar10_vit_supervised.yaml"
  echo "  bash jobs/submit_experiment.sh experiment-c10-vit-sup eval vit_supervised configs/cifar10_vit_supervised.yaml"
  echo "  bash jobs/submit_experiment.sh experiment-c10-vit-sup eval_true vit_supervised configs/cifar10_vit_supervised.yaml"
  echo ""
  echo "  bash jobs/submit_experiment.sh experiment-c10-vit-lejepa train vit_lejepa configs/cifar10_vit_lejepa.yaml"
  echo "  bash jobs/submit_experiment.sh experiment-c10-vit-lejepa repr vit_lejepa configs/cifar10_vit_lejepa.yaml"
  echo "  bash jobs/submit_experiment.sh experiment-c10-vit-lejepa eval vit_lejepa configs/cifar10_vit_lejepa.yaml"
  echo "  bash jobs/submit_experiment.sh experiment-c10-vit-lejepa eval_true vit_lejepa configs/cifar10_vit_lejepa.yaml"
  echo "  bash jobs/submit_experiment.sh experiment-c10-vit-lejepa feature_spaces vit_lejepa configs/cifar10_vit_lejepa.yaml"
  echo ""
  echo "  bash jobs/submit_experiment.sh imagenet100-r18 train lejepa configs/imagenet100_resnet18_lejepa.yaml"
  echo "  bash jobs/submit_experiment.sh imagenet100-vit train vit_lejepa configs/imagenet100_vit_lejepa.yaml"
  exit 1
fi

EXP_NAME="$1"
MODE="$2"
TARGET="$3"
CONFIG_PATH_ARG="${4:-}"
EXP_DIR="experiments/${EXP_NAME}"

EXPORT_BASE="ALL,EXPERIMENT_NAME=${EXP_NAME},MODE=${MODE},TARGET=${TARGET}"

if [ -n "${CONFIG_PATH_ARG}" ]; then
  EXPORT_BASE="${EXPORT_BASE},CONFIG_PATH=${CONFIG_PATH_ARG}"
fi

mkdir -p "${EXP_DIR}/logs"

submit_supervised_train() {
  sbatch \
    --export="${EXPORT_BASE}" \
    --output="${EXP_DIR}/logs/train_supervised_%j.out" \
    --error="${EXP_DIR}/logs/train_supervised_%j.err" \
    jobs/train_supervised_resnet18_cifar10.slurm
}

submit_lejepa_train() {
  sbatch \
    --export="${EXPORT_BASE}" \
    --output="${EXP_DIR}/logs/train_lejepa_%j.out" \
    --error="${EXP_DIR}/logs/train_lejepa_%j.err" \
    jobs/train_lejepa_resnet18_cifar10.slurm
}

submit_supervised_eval_predicted() {
  sbatch \
    --export="${EXPORT_BASE},GRADCAM_TARGET=predicted" \
    --output="${EXP_DIR}/logs/eval_supervised_predicted_%j.out" \
    --error="${EXP_DIR}/logs/eval_supervised_predicted_%j.err" \
    jobs/evaluate_supervised_resnet18_cifar10.slurm
}

submit_lejepa_eval_predicted() {
  sbatch \
    --export="${EXPORT_BASE},GRADCAM_TARGET=predicted" \
    --output="${EXP_DIR}/logs/eval_lejepa_predicted_%j.out" \
    --error="${EXP_DIR}/logs/eval_lejepa_predicted_%j.err" \
    jobs/evaluate_lejepa_resnet18_cifar10.slurm
}

submit_supervised_eval_true() {
  sbatch \
    --export="${EXPORT_BASE},GRADCAM_TARGET=true" \
    --output="${EXP_DIR}/logs/eval_supervised_true_%j.out" \
    --error="${EXP_DIR}/logs/eval_supervised_true_%j.err" \
    jobs/evaluate_supervised_resnet18_cifar10.slurm
}

submit_lejepa_eval_true() {
  sbatch \
    --export="${EXPORT_BASE},GRADCAM_TARGET=true" \
    --output="${EXP_DIR}/logs/eval_lejepa_true_%j.out" \
    --error="${EXP_DIR}/logs/eval_lejepa_true_%j.err" \
    jobs/evaluate_lejepa_resnet18_cifar10.slurm
}

submit_supervised_repr() {
  sbatch \
    --export="${EXPORT_BASE}" \
    --output="${EXP_DIR}/logs/repr_supervised_%j.out" \
    --error="${EXP_DIR}/logs/repr_supervised_%j.err" \
    jobs/evaluate_representation_supervised.slurm
}

submit_lejepa_repr() {
  sbatch \
    --export="${EXPORT_BASE}" \
    --output="${EXP_DIR}/logs/repr_lejepa_%j.out" \
    --error="${EXP_DIR}/logs/repr_lejepa_%j.err" \
    jobs/evaluate_representation_lejepa.slurm
}

submit_lejepa_feature_spaces() {
  sbatch \
    --export="${EXPORT_BASE},FEATURE_SOURCE=all,FEATURE_CHECKPOINT=${FEATURE_CHECKPOINT:-best}" \
    --output="${EXP_DIR}/logs/feature_spaces_lejepa_%j.out" \
    --error="${EXP_DIR}/logs/feature_spaces_lejepa_%j.err" \
    jobs/evaluate_feature_spaces_lejepa.slurm
}

submit_vit_supervised_train() {
  sbatch \
    --export="${EXPORT_BASE}" \
    --output="${EXP_DIR}/logs/train_vit_supervised_%j.out" \
    --error="${EXP_DIR}/logs/train_vit_supervised_%j.err" \
    jobs/train_supervised_vit_cifar10.slurm
}

submit_vit_supervised_eval_predicted() {
  sbatch \
    --export="${EXPORT_BASE},GRADCAM_TARGET=predicted" \
    --output="${EXP_DIR}/logs/eval_vit_supervised_predicted_%j.out" \
    --error="${EXP_DIR}/logs/eval_vit_supervised_predicted_%j.err" \
    jobs/evaluate_supervised_vit_cifar10.slurm
}

submit_vit_supervised_eval_true() {
  sbatch \
    --export="${EXPORT_BASE},GRADCAM_TARGET=true" \
    --output="${EXP_DIR}/logs/eval_vit_supervised_true_%j.out" \
    --error="${EXP_DIR}/logs/eval_vit_supervised_true_%j.err" \
    jobs/evaluate_supervised_vit_cifar10.slurm
}

submit_vit_supervised_repr() {
  sbatch \
    --export="${EXPORT_BASE}" \
    --output="${EXP_DIR}/logs/repr_vit_supervised_%j.out" \
    --error="${EXP_DIR}/logs/repr_vit_supervised_%j.err" \
    jobs/evaluate_representation_vit_supervised.slurm
}

submit_vit_lejepa_train() {
  sbatch \
    --export="${EXPORT_BASE}" \
    --output="${EXP_DIR}/logs/train_vit_lejepa_%j.out" \
    --error="${EXP_DIR}/logs/train_vit_lejepa_%j.err" \
    jobs/train_lejepa_vit_cifar10.slurm
}

submit_vit_lejepa_eval_predicted() {
  sbatch \
    --export="${EXPORT_BASE},GRADCAM_TARGET=predicted" \
    --output="${EXP_DIR}/logs/eval_vit_lejepa_predicted_%j.out" \
    --error="${EXP_DIR}/logs/eval_vit_lejepa_predicted_%j.err" \
    jobs/evaluate_lejepa_vit_cifar10.slurm
}

submit_vit_lejepa_eval_true() {
  sbatch \
    --export="${EXPORT_BASE},GRADCAM_TARGET=true" \
    --output="${EXP_DIR}/logs/eval_vit_lejepa_true_%j.out" \
    --error="${EXP_DIR}/logs/eval_vit_lejepa_true_%j.err" \
    jobs/evaluate_lejepa_vit_cifar10.slurm
}

submit_vit_lejepa_repr() {
  sbatch \
    --export="${EXPORT_BASE}" \
    --output="${EXP_DIR}/logs/repr_vit_lejepa_%j.out" \
    --error="${EXP_DIR}/logs/repr_vit_lejepa_%j.err" \
    jobs/evaluate_representation_vit_lejepa.slurm
}

submit_vit_lejepa_feature_spaces() {
  sbatch \
    --export="${EXPORT_BASE},FEATURE_SOURCE=all,FEATURE_CHECKPOINT=${FEATURE_CHECKPOINT:-best}" \
    --output="${EXP_DIR}/logs/feature_spaces_vit_lejepa_%j.out" \
    --error="${EXP_DIR}/logs/feature_spaces_vit_lejepa_%j.err" \
    jobs/evaluate_feature_spaces_vit_lejepa.slurm
}


submit_grounded_predicted() {
  sbatch \
    --export="${EXPORT_BASE},GROUNDED_TARGET=predicted" \
    --output="${EXP_DIR}/logs/grounded_${TARGET}_predicted_%j.out" \
    --error="${EXP_DIR}/logs/grounded_${TARGET}_predicted_%j.err" \
    jobs/evaluate_grounded_alignment.slurm
}

submit_grounded_true() {
  sbatch \
    --export="${EXPORT_BASE},GROUNDED_TARGET=true" \
    --output="${EXP_DIR}/logs/grounded_${TARGET}_true_%j.out" \
    --error="${EXP_DIR}/logs/grounded_${TARGET}_true_%j.err" \
    jobs/evaluate_grounded_alignment.slurm
}

submit_lei_lsas() {
  sbatch \
    --export="${EXPORT_BASE},LEI_METRIC=lsas,LEI_THRESHOLD=${LEI_THRESHOLD:-0.30}" \
    --output="${EXP_DIR}/logs/lei_lsas_%j.out" \
    --error="${EXP_DIR}/logs/lei_lsas_%j.err" \
    jobs/compute_lei_experiment.slurm
}

submit_lei_glsas() {
  sbatch \
    --export="${EXPORT_BASE},LEI_METRIC=g_lsas,LEI_THRESHOLD=${LEI_THRESHOLD:-0.30}" \
    --output="${EXP_DIR}/logs/lei_glsas_%j.out" \
    --error="${EXP_DIR}/logs/lei_glsas_%j.err" \
    jobs/compute_lei_experiment.slurm
}

case "${MODE}:${TARGET}" in
  train:supervised)
    submit_supervised_train
    ;;
  train:lejepa)
    submit_lejepa_train
    ;;
  train:both)
    submit_supervised_train
    submit_lejepa_train
    ;;
  train:vit_supervised)
    submit_vit_supervised_train
    ;;
  train:vit_lejepa)
    submit_vit_lejepa_train
    ;;

  eval:supervised)
    submit_supervised_eval_predicted
    ;;
  eval:lejepa)
    submit_lejepa_eval_predicted
    ;;
  eval:both)
    submit_supervised_eval_predicted
    submit_lejepa_eval_predicted
    ;;
  eval:vit_supervised)
    submit_vit_supervised_eval_predicted
    ;;
  eval:vit_lejepa)
    submit_vit_lejepa_eval_predicted
    ;;

  eval_true:supervised)
    submit_supervised_eval_true
    ;;
  eval_true:lejepa)
    submit_lejepa_eval_true
    ;;
  eval_true:both)
    submit_supervised_eval_true
    submit_lejepa_eval_true
    ;;
  eval_true:vit_supervised)
    submit_vit_supervised_eval_true
    ;;
  eval_true:vit_lejepa)
    submit_vit_lejepa_eval_true
    ;;

  repr:supervised)
    submit_supervised_repr
    ;;
  repr:lejepa)
    submit_lejepa_repr
    ;;
  repr:both)
    submit_supervised_repr
    submit_lejepa_repr
    ;;
  repr:vit_supervised)
    submit_vit_supervised_repr
    ;;
  repr:vit_lejepa)
    submit_vit_lejepa_repr
    ;;

  feature_spaces:lejepa)
    submit_lejepa_feature_spaces
    ;;
  feature_spaces:vit_lejepa)
    submit_vit_lejepa_feature_spaces
    ;;


  grounded:supervised|grounded:lejepa|grounded:vit_supervised|grounded:vit_lejepa)
    submit_grounded_predicted
    ;;
  grounded_true:supervised|grounded_true:lejepa|grounded_true:vit_supervised|grounded_true:vit_lejepa)
    submit_grounded_true
    ;;


  lei:supervised|lei:lejepa|lei:both|lei:vit_supervised|lei:vit_lejepa)
    submit_lei_lsas
    ;;
  glei:supervised|glei:lejepa|glei:vit_supervised|glei:vit_lejepa)
    submit_lei_glsas
    ;;

  *)
    echo "Invalid combination: mode=${MODE}, target=${TARGET}"
    echo "Valid modes: train, eval, eval_true, repr, feature_spaces, grounded, grounded_true, lei, glei"
    echo "Valid targets: supervised, lejepa, both, vit_supervised, vit_lejepa"
    echo ""
    echo "Note: feature_spaces is valid only for targets: lejepa, vit_lejepa"
    exit 1
    ;;
esac
