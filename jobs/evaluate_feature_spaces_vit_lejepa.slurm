#!/bin/bash
#SBATCH --job-name=cv-feat-vit
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

TRAIN_CONFIG=""

# 1. If CONFIG_PATH was passed by submit_experiment.sh, respect it.
if [ -n "${CONFIG_PATH:-}" ]; then
  if [ -f "${CONFIG_PATH}" ]; then
    TRAIN_CONFIG="${CONFIG_PATH}"
  elif [ -f "${EXP_DIR}/configs/$(basename "${CONFIG_PATH}")" ]; then
    TRAIN_CONFIG="${EXP_DIR}/configs/$(basename "${CONFIG_PATH}")"
  else
    echo "WARNING: CONFIG_PATH was provided but not found: ${CONFIG_PATH}"
  fi
fi

# 2. Prefer clean ViT LeJEPA training resolved configs.
if [ -z "${TRAIN_CONFIG}" ]; then
  TRAIN_CONFIG="$(find "${EXP_DIR}/configs" -maxdepth 1 -type f -name "resolved_*train_vit_lejepa*.yaml" | sort | tail -n 1)"
fi

# 3. Fallback for older runs that used train_lejepa even with vit_tiny_cifar.
if [ -z "${TRAIN_CONFIG}" ]; then
  TRAIN_CONFIG="$(find "${EXP_DIR}/configs" -maxdepth 1 -type f -name "resolved_*train_lejepa*.yaml" | sort | tail -n 1)"
fi

if [ -z "${TRAIN_CONFIG}" ]; then
  echo "ERROR: could not find LeJEPA training resolved config in ${EXP_DIR}/configs"
  echo "Expected one of:"
  echo "  resolved_*train_vit_lejepa*.yaml"
  echo "  resolved_*train_lejepa*.yaml"
  echo "Available configs:"
  find "${EXP_DIR}/configs" -maxdepth 1 -type f | sort || true
  exit 1
fi

echo "Using training resolved config: ${TRAIN_CONFIG}"

ARCH="$(python - <<PY
import yaml
with open("${TRAIN_CONFIG}", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
print(cfg.get("model", {}).get("architecture", ""))
PY
)"

if [[ "${ARCH}" != vit* ]]; then
  echo "ERROR: selected config is not a ViT config."
  echo "Selected config: ${TRAIN_CONFIG}"
  echo "architecture: ${ARCH}"
  exit 1
fi

echo "Confirmed ViT architecture: ${ARCH}"

echo "Using training resolved config: ${TRAIN_CONFIG}"

BACKBONE_CHECKPOINT="${EXP_DIR}/checkpoints/vit_lejepa_backbone_last.pt"

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
