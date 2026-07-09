#!/bin/bash
#SBATCH --job-name=debug-okai
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_MNLP26
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=experiments/public-okai-vitb16-in100-infer/logs/debug_okai_%j.out
#SBATCH --error=experiments/public-okai-vitb16-in100-infer/logs/debug_okai_%j.err

set -euo pipefail

PROJECT_DIR="/leonardo_scratch/large/userexternal/apromots/doubleN/cv"
cd "$PROJECT_DIR"

source .venv/bin/activate

export OKAI_MODEL_DIR="${PROJECT_DIR}/hf_models/okai_lejepa_vitb16"
export PYTHONPATH="$OKAI_MODEL_DIR:${PYTHONPATH:-}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python - <<'PY'
import torch
from transformers import AutoModel

model_dir = "/leonardo_scratch/large/userexternal/apromots/doubleN/cv/hf_models/okai_lejepa_vitb16"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

model = AutoModel.from_pretrained(
    model_dir,
    trust_remote_code=True,
    local_files_only=True,
)
model.eval().to(device)

x = torch.randn(1, 3, 224, 224, device=device)

with torch.no_grad():
    try:
        out = model(pixel_values=x)
    except TypeError:
        out = model(x)

print("\nOutput type:", type(out))

if isinstance(out, dict):
    print("Output keys:", list(out.keys()))
    for k, v in out.items():
        if torch.is_tensor(v):
            print(k, tuple(v.shape), v.dtype, v.device)
        else:
            print(k, type(v), v)
else:
    print("Non-dict output:", out)

print("\nQKV candidates:")
for name, module in model.named_modules():
    if name.endswith("qkv") or "qkv" in name.lower():
        print(name, "=>", module)

print("\nAttention-like modules:")
for name, module in model.named_modules():
    lname = name.lower()
    if "attn" in lname or "attention" in lname:
        print(name, "=>", module)

print("\nDONE")
PY
