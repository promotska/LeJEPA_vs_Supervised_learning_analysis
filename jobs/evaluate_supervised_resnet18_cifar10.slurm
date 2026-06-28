#!/bin/bash
#SBATCH --job-name=cv-c10-r18-sup-eval
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

cd "${PROJECT_DIR}"

mkdir -p logs outputs/metrics outputs/figures

source ".venv/bin/activate"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "Started at: $(date)"
echo "Node: $(hostname)"
nvidia-smi || true

echo "Started at: $(date)"

python -u scripts/evaluate_alignment.py \
  --config configs/cifar10_resnet18_supervised.yaml \
  --mode supervised

echo "Finished at: $(date)"