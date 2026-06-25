#!/bin/bash
set -euo pipefail

if [ $# -lt 3 ]; then
  echo "Usage:"
  echo "  bash slurm/submit_experiment.sh <experiment-name> <mode> <target>"
  echo ""
  echo "Modes:"
  echo "  train"
  echo "  eval"
  echo ""
  echo "Targets:"
  echo "  supervised"
  echo "  lejepa"
  echo "  both"
  echo ""
  echo "Examples:"
  echo "  bash slurm/submit_experiment.sh experiment-4 train supervised"
  echo "  bash slurm/submit_experiment.sh experiment-4 train lejepa"
  echo "  bash slurm/submit_experiment.sh experiment-4 train both"
  echo "  bash slurm/submit_experiment.sh experiment-4 eval both"
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
    slurm/train_supervised_resnet18_cifar10.slurm
}

submit_lejepa_train() {
  sbatch \
    --export=ALL,EXPERIMENT_NAME="${EXP_NAME}" \
    --output="${EXP_DIR}/logs/train_lejepa_%j.out" \
    --error="${EXP_DIR}/logs/train_lejepa_%j.err" \
    slurm/train_lejepa_resnet18_cifar10.slurm
}

submit_supervised_eval() {
  sbatch \
    --export=ALL,EXPERIMENT_NAME="${EXP_NAME}" \
    --output="${EXP_DIR}/logs/eval_supervised_%j.out" \
    --error="${EXP_DIR}/logs/eval_supervised_%j.err" \
    jobs/evaluate_supervised_resnet18_cifar10.slurm
}

submit_lejepa_eval() {
  sbatch \
    --export=ALL,EXPERIMENT_NAME="${EXP_NAME}" \
    --output="${EXP_DIR}/logs/eval_lejepa_%j.out" \
    --error="${EXP_DIR}/logs/eval_lejepa_%j.err" \
    jobs/evaluate_lejepa_resnet18_cifar10.slurm
}

case "${MODE}:${TARGET}" in
  train:supervised)
    echo "Submitting supervised training for ${EXP_NAME}"
    submit_supervised_train
    ;;
  train:lejepa)
    echo "Submitting LeJEPA training for ${EXP_NAME}"
    submit_lejepa_train
    ;;
  train:both)
    echo "Submitting supervised + LeJEPA training for ${EXP_NAME}"
    submit_supervised_train
    submit_lejepa_train
    ;;
  eval:supervised)
    echo "Submitting supervised evaluation for ${EXP_NAME}"
    submit_supervised_eval
    ;;
  eval:lejepa)
    echo "Submitting LeJEPA evaluation for ${EXP_NAME}"
    submit_lejepa_eval
    ;;
  eval:both)
    echo "Submitting supervised + LeJEPA evaluation for ${EXP_NAME}"
    submit_supervised_eval
    submit_lejepa_eval
    ;;
  *)
    echo "Invalid combination: mode=${MODE}, target=${TARGET}"
    echo "Valid modes: train, eval"
    echo "Valid targets: supervised, lejepa, both"
    exit 1
    ;;
esac
