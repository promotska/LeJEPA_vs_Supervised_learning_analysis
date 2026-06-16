#!/bin/bash
#SBATCH --job-name=cv-r18-sup
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_MNLP26
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
VENV_DIR="${PROJECT_DIR}/../.venv"

cd "${PROJECT_DIR}"

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Project dir: ${PROJECT_DIR}"
echo "Venv dir: ${VENV_DIR}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

source ".venv/bin/activate"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Started at: $(date)"

python -u scripts/train_supervised.py \
  --config configs/cifar10_resnet18_supervised.yaml

echo "Job finished at: $(date)"
