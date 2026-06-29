#!/bin/bash
#SBATCH --job-name=cv-vit-lejepa-eval
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_MNLP26
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
CONFIG_PATH="${CONFIG_PATH:-configs/cifar10_vit_lejepa.yaml}"
GRADCAM_TARGET="${GRADCAM_TARGET:-predicted}"
VIT_XAI_METHOD="${VIT_XAI_METHOD:-}"

if [ -d "${PROJECT_DIR}/.venv" ]; then
  VENV_DIR="${PROJECT_DIR}/.venv"
elif [ -d "${PROJECT_DIR}/../.venv" ]; then
  VENV_DIR="${PROJECT_DIR}/../.venv"
else
  echo "ERROR: could not find .venv"
  exit 1
fi

cd "${PROJECT_DIR}"
mkdir -p logs
source "${VENV_DIR}/bin/activate"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "Config path: ${CONFIG_PATH}"
nvidia-smi || true

CMD=(python -u scripts/evaluate_vit_alignment.py
  --config "${CONFIG_PATH}"
  --mode vit_lejepa
  --gradcam-target "${GRADCAM_TARGET}")

if [ -n "${VIT_XAI_METHOD}" ]; then
  CMD+=(--vit-xai-method "${VIT_XAI_METHOD}")
fi

"${CMD[@]}"

echo "Job finished at: $(date)"
