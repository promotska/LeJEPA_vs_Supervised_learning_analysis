#!/bin/bash
#SBATCH --job-name=test-sigreg
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_MNLP26
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
source .venv/bin/activate

export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Node: $(hostname)"
echo "Started: $(date)"

python - <<'PY'
import time
import torch
from src.training.losses import OfficialSIGRegLoss

t = time.time()
loss_fn = OfficialSIGRegLoss(
    num_slices=1024,
    num_points=17,
)
print("created OfficialSIGRegLoss in seconds:", time.time() - t)

x = torch.randn(128, 256, device="cuda" if torch.cuda.is_available() else "cpu")

t = time.time()
loss = loss_fn(x)
print("loss:", loss)
print("loss shape:", getattr(loss, "shape", None))
print("loss seconds:", time.time() - t)
print("OK")
PY

echo "Finished: $(date)"
