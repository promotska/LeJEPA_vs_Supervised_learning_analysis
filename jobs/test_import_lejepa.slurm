#!/bin/bash
#SBATCH --job-name=test-lejepa
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

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Node: $(hostname)"
echo "Started: $(date)"

python - <<'PY'
import inspect
from lejepa.univariate import EppsPulley
from lejepa.multivariate import SlicingUnivariateTest

print("EppsPulley signature:")
print(inspect.signature(EppsPulley))

print("SlicingUnivariateTest signature:")
print(inspect.signature(SlicingUnivariateTest))
PY

echo "Finished: $(date)"
