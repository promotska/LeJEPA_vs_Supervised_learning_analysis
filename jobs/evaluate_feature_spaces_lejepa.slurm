#!/bin/bash
#SBATCH --job-name=cv-feat-r18
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_MNLP26
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR}"

if [ -d "${PROJECT_DIR}/.venv" ]; then
  VENV_DIR="${PROJECT_DIR}/.venv"
elif [ -d "${PROJECT_DIR}/../.venv" ]; then
  VENV_DIR="${PROJECT_DIR}/../.venv"
else
  echo "ERROR: could not find .venv"
  exit 1
fi

cd "${PROJECT_DIR}"

source "${VENV_DIR}/bin/activate"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

: "${EXPERIMENT_NAME:?EXPERIMENT_NAME is required}"

EXP_DIR="experiments/${EXPERIMENT_NAME}"
FEATURE_SOURCE="${FEATURE_SOURCE:-all}"

TRAIN_CONFIG="$(find "${EXP_DIR}/configs" -maxdepth 1 -type f -name "resolved_*train_lejepa*.yaml" | sort | tail -n 1)"

if [ -z "${TRAIN_CONFIG}" ]; then
  echo "ERROR: could not find training resolved config in ${EXP_DIR}/configs"
  echo "Expected pattern: resolved_*train_lejepa*.yaml"
  echo "Available configs:"
  find "${EXP_DIR}/configs" -maxdepth 1 -type f | sort || true
  exit 1
fi

echo "Using training resolved config: ${TRAIN_CONFIG}"

BACKBONE_CHECKPOINT="${EXP_DIR}/checkpoints/resnet18_lejepa_backbone_last.pt"

if [ ! -f "${BACKBONE_CHECKPOINT}" ]; then
  echo "ERROR: missing checkpoint: ${BACKBONE_CHECKPOINT}"
  echo "Available checkpoints:"
  find "${EXP_DIR}/checkpoints" -maxdepth 1 -type f | sort || true
  exit 1
fi

echo "Using backbone checkpoint: ${BACKBONE_CHECKPOINT}"

python -u scripts/evaluate_lejepa_feature_spaces.py \
  --config "${TRAIN_CONFIG}" \
  --checkpoint "${BACKBONE_CHECKPOINT}" \
  --feature-source "${FEATURE_SOURCE}"
