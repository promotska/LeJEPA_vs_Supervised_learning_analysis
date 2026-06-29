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
VENV_DIR="${PROJECT_DIR}/../.venv"

cd "${PROJECT_DIR}"

source "${VENV_DIR}/bin/activate"

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

CONFIG="${CONFIG_PATH:-configs/cifar10_resnet18_lejepa.yaml}"
FEATURE_SOURCE="${FEATURE_SOURCE:-all}"

python -u scripts/evaluate_lejepa_feature_spaces.py \
  --config "${CONFIG}" \
  --feature-source "${FEATURE_SOURCE}"