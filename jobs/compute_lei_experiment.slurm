#!/bin/bash
#SBATCH --job-name=cv-lei
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_MNLP26
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
EXP_NAME="${EXPERIMENT_NAME:?EXPERIMENT_NAME is required}"
LEI_METRIC="${LEI_METRIC:-lsas}"
LEI_THRESHOLD="${LEI_THRESHOLD:-0.30}"

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

python -u scripts/compute_lei_for_experiment.py \
  --experiment-dir "experiments/${EXP_NAME}" \
  --metric "${LEI_METRIC}" \
  --threshold "${LEI_THRESHOLD}"
