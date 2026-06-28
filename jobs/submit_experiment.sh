#!/bin/bash
set -euo pipefail

if [ $# -lt 3 ]; then
  echo "Usage:"
  echo "  bash jobs/submit_experiment.sh <experiment-name> <mode> <target>"
  echo ""
  echo "Modes:"
  echo "  train"
  echo "  eval"
  echo "  eval_true"
  echo "  repr"
  echo ""
  echo "Targets:"
  echo "  supervised        ResNet supervised"
  echo "  lejepa            ResNet LeJEPA"
  echo "  both              ResNet supervised + ResNet LeJEPA"
  echo "  vit_supervised    ViT supervised"
  echo "  vit_lejepa        ViT LeJEPA-SIGReg"
  echo ""
  echo "Examples:"
  echo "  bash jobs/submit_experiment.sh experiment-4 train both"
  echo "  bash jobs/submit_experiment.sh experiment-4 eval both"
  echo "  bash jobs/submit_experiment.sh experiment-4 eval_true lejepa"
  echo "  bash jobs/submit_experiment.sh experiment-4 repr both"
  echo "  bash jobs/submit_experiment.sh experiment-6 train vit_supervised"
  echo "  bash jobs/submit_experiment.sh experiment-6 eval vit_supervised"
  echo "  bash jobs/submit_experiment.sh experiment-6 eval_true vit_supervised"
  echo "  bash jobs/submit_experiment.sh experiment-6 repr vit_supervised"
  echo "  bash jobs/submit_experiment.sh experiment-6 train vit_lejepa"
  echo "  bash jobs/submit_experiment.sh experiment-6 eval vit_lejepa"
  echo "  bash jobs/submit_experiment.sh experiment-6 repr vit_lejepa"
  exit 1
fi

EXP_NAME="$1"
MODE="$2"
TARGET="$3"
EXP_DIR="experiments/${EXP_NAME}"

mkdir -p "${EXP_DIR}/logs"

submit_supervised_train() {
  sbatch \
    --export=ALL,EXPERIMENT_NAME="${EXP_NAME}" \
    --output="${EXP_DIR}/logs/train_supervised_%j.out" \
    --error="${EXP_DIR}/logs/train_supervised_%j.err" \
    jobs/train_supervised_resnet18_cifar10.slurm
}

submit_lejepa_train() {
  sbatch \
    --export=ALL,EXPERIMENT_NAME="${EXP_NAME}" \
    --output="${EXP_DIR}/logs/train_lejepa_%j.out" \
    --error="${EXP_DIR}/logs/train_lejepa_%j.err" \
    jobs/train_lejepa_resnet18_cifar10.slurm
}

submit_supervised_eval_predicted() {
  sbatch \
    --export=ALL,EXPERIMENT_NAME="${EXP_NAME}",GRADCAM_TARGET=predicted \
    --output="${EXP_DIR}/logs/eval_supervised_predicted_%j.out" \
    --error="${EXP_DIR}/logs/eval_supervised_predicted_%j.err" \
    jobs/evaluate_supervised_resnet18_cifar10.slurm
}

submit_lejepa_eval_predicted() {
  sbatch \
    --export=ALL,EXPERIMENT_NAME="${EXP_NAME}",GRADCAM_TARGET=predicted \
    --output="${EXP_DIR}/logs/eval_lejepa_predicted_%j.out" \
    --error="${EXP_DIR}/logs/eval_lejepa_predicted_%j.err" \
    jobs/evaluate_lejepa_resnet18_cifar10.slurm
}

submit_supervised_eval_true() {
  sbatch \
    --export=ALL,EXPERIMENT_NAME="${EXP_NAME}",GRADCAM_TARGET=true \
    --output="${EXP_DIR}/logs/eval_supervised_true_%j.out" \
    --error="${EXP_DIR}/logs/eval_supervised_true_%j.err" \
    jobs/evaluate_supervised_resnet18_cifar10.slurm
}

submit_lejepa_eval_true() {
  sbatch \
    --export=ALL,EXPERIMENT_NAME="${EXP_NAME}",GRADCAM_TARGET=true \
    --output="${EXP_DIR}/logs/eval_lejepa_true_%j.out" \
    --error="${EXP_DIR}/logs/eval_lejepa_true_%j.err" \
    jobs/evaluate_lejepa_resnet18_cifar10.slurm
}

submit_supervised_repr() {
  sbatch \
    --export=ALL,EXPERIMENT_NAME="${EXP_NAME}" \
    --output="${EXP_DIR}/logs/repr_supervised_%j.out" \
    --error="${EXP_DIR}/logs/repr_supervised_%j.err" \
    jobs/evaluate_representation_supervised.slurm
}

submit_lejepa_repr() {
  sbatch \
    --export=ALL,EXPERIMENT_NAME="${EXP_NAME}" \
    --output="${EXP_DIR}/logs/repr_lejepa_%j.out" \
    --error="${EXP_DIR}/logs/repr_lejepa_%j.err" \
    jobs/evaluate_representation_lejepa.slurm
}

submit_vit_supervised_train() {
  sbatch \
    --export=ALL,EXPERIMENT_NAME="${EXP_NAME}" \
    --output="${EXP_DIR}/logs/train_vit_supervised_%j.out" \
    --error="${EXP_DIR}/logs/train_vit_supervised_%j.err" \
    jobs/train_supervised_vit_cifar10.slurm
}

submit_vit_supervised_eval_predicted() {
  sbatch \
    --export=ALL,EXPERIMENT_NAME="${EXP_NAME}",GRADCAM_TARGET=predicted \
    --output="${EXP_DIR}/logs/eval_vit_supervised_predicted_%j.out" \
    --error="${EXP_DIR}/logs/eval_vit_supervised_predicted_%j.err" \
    jobs/evaluate_supervised_vit_cifar10.slurm
}

submit_vit_supervised_eval_true() {
  sbatch \
    --export=ALL,EXPERIMENT_NAME="${EXP_NAME}",GRADCAM_TARGET=true \
    --output="${EXP_DIR}/logs/eval_vit_supervised_true_%j.out" \
    --error="${EXP_DIR}/logs/eval_vit_supervised_true_%j.err" \
    jobs/evaluate_supervised_vit_cifar10.slurm
}

submit_vit_supervised_repr() {
  sbatch \
    --export=ALL,EXPERIMENT_NAME="${EXP_NAME}" \
    --output="${EXP_DIR}/logs/repr_vit_supervised_%j.out" \
    --error="${EXP_DIR}/logs/repr_vit_supervised_%j.err" \
    jobs/evaluate_representation_vit_supervised.slurm
}

submit_vit_lejepa_train() {
  sbatch \
    --export=ALL,EXPERIMENT_NAME="${EXP_NAME}" \
    --output="${EXP_DIR}/logs/train_vit_lejepa_%j.out" \
    --error="${EXP_DIR}/logs/train_vit_lejepa_%j.err" \
    jobs/train_lejepa_vit_cifar10.slurm
}

submit_vit_lejepa_eval_predicted() {
  sbatch \
    --export=ALL,EXPERIMENT_NAME="${EXP_NAME}",GRADCAM_TARGET=predicted \
    --output="${EXP_DIR}/logs/eval_vit_lejepa_predicted_%j.out" \
    --error="${EXP_DIR}/logs/eval_vit_lejepa_predicted_%j.err" \
    jobs/evaluate_lejepa_vit_cifar10.slurm
}

submit_vit_lejepa_eval_true() {
  sbatch \
    --export=ALL,EXPERIMENT_NAME="${EXP_NAME}",GRADCAM_TARGET=true \
    --output="${EXP_DIR}/logs/eval_vit_lejepa_true_%j.out" \
    --error="${EXP_DIR}/logs/eval_vit_lejepa_true_%j.err" \
    jobs/evaluate_lejepa_vit_cifar10.slurm
}

submit_vit_lejepa_repr() {
  sbatch \
    --export=ALL,EXPERIMENT_NAME="${EXP_NAME}" \
    --output="${EXP_DIR}/logs/repr_vit_lejepa_%j.out" \
    --error="${EXP_DIR}/logs/repr_vit_lejepa_%j.err" \
    jobs/evaluate_representation_vit_lejepa.slurm
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

  *)
    echo "Invalid combination: mode=${MODE}, target=${TARGET}"
    echo "Valid modes: train, eval, eval_true, repr"
    echo "Valid targets: supervised, lejepa, both, vit_supervised, vit_lejepa"
    exit 1
    ;;
esac
