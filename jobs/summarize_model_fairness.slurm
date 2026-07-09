#!/bin/bash
#SBATCH --job-name=cv-fairness
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_MNLP26
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_DIR}"

if [ -d "${PROJECT_DIR}/.venv" ]; then
  source "${PROJECT_DIR}/.venv/bin/activate"
elif [ -d "${PROJECT_DIR}/../.venv" ]; then
  source "${PROJECT_DIR}/../.venv/bin/activate"
fi

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

mkdir -p logs experiments

python -u scripts/summarize_model_fairness.py \
  --experiments-root experiments \
  --output experiments/model_fairness.csv \
  --write-per-experiment \
  --public-okai-config hf_models/okai_lejepa_vitb16/config.json \
  --public-supervised-config hf_models/supervised_vitb16_augreg_in1k/config.json \
  --public-dataset ImageNet-100
