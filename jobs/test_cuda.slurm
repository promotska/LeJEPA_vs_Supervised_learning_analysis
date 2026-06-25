#!/bin/bash
#SBATCH --job-name=test-cuda
#SBATCH --account=IscrC_MNLP26
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source .venv/bin/activate

echo "Node:"
hostname

echo "GPU:"
nvidia-smi

python - <<'PY'
import torch

print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda version:", torch.version.cuda)
print("device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    print("CUDA tensor test OK:", y.device)
else:
    raise SystemExit("CUDA is not available")
PY
