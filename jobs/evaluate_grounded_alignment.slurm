#!/bin/bash
#SBATCH --job-name=grounded_alignment
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_MNLP26
#SBATCH --qos=normal
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

CONFIG_PATH="${CONFIG_PATH:?CONFIG_PATH is required}"
TARGET="${TARGET:?TARGET is required: supervised, lejepa, vit_supervised, vit_lejepa}"
GROUNDED_TARGET="${GROUNDED_TARGET:-predicted}"

python -u scripts/evaluate_grounded_alignment.py \
  --config "${CONFIG_PATH}" \
  --mode "${TARGET}" \
  --target "${GROUNDED_TARGET}"
