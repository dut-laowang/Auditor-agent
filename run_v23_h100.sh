#!/usr/bin/env bash
set -euo pipefail

# True clone-and-run entry point for one 96GB H100.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${V23_WORK_DIR:-$REPO/.v23_runtime}"
VENV="${V23_VENV:-$WORK/venv}"
ARCHIVE="$REPO/SFT/auditor_agent_sft_v23_final_package/assets/v23_final_aligned_combined.tar.gz"
ARCHIVE_SHA256="cd0a2f8869927bfe3ddf66cb96d43acd75d90d1055268e5db3546bcaab7b97fb"
DATA="$WORK/data/v23_final_aligned_combined"
export BASE="${BASE:-$WORK}"
export REPO
export V23_DATA_DIR="${V23_DATA_DIR:-$DATA}"
export V23_RUN="${V23_RUN:-$WORK/v23_final_run}"
export V23_EXPERIMENT_RUN="${V23_EXPERIMENT_RUN:-$WORK/v23_final_experiment_run}"
export GPU="${GPU:-0}"
export V23_GPUS="${V23_GPUS:-0,1}"
export V23_GPU_MEMORY_GB="${V23_GPU_MEMORY_GB:-96}"
export V23_GPU_MEMORY_UTILIZATION="${V23_GPU_MEMORY_UTILIZATION:-0.85}"
export QWEN_EVAL_BATCH="${QWEN_EVAL_BATCH:-8}"
export INTERNLM_EVAL_BATCH="${INTERNLM_EVAL_BATCH:-8}"
export MODERN_EVAL_BATCH="${MODERN_EVAL_BATCH:-16}"

command -v python3 >/dev/null || { echo 'python3 is required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'NVIDIA driver/nvidia-smi is required' >&2; exit 2; }
command -v git >/dev/null || { echo 'git is required' >&2; exit 2; }
command -v tar >/dev/null || { echo 'tar is required' >&2; exit 2; }
IFS=',' read -ra requested_gpus <<< "$V23_GPUS"
[[ "${#requested_gpus[@]}" -gt 0 ]] || { echo 'V23_GPUS is empty' >&2; exit 2; }
for requested_gpu in "${requested_gpus[@]}"; do
  requested_gpu="${requested_gpu//[[:space:]]/}"
  gpu_mib="$(nvidia-smi --id="$requested_gpu" --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
  [[ "$gpu_mib" =~ ^[0-9]+$ && "$gpu_mib" -ge 90000 ]] || {
    echo "run_v23_h100.sh requires every selected GPU to have >=90,000 MiB; GPU $requested_gpu reports ${gpu_mib:-unknown} MiB" >&2; exit 2;
  }
done
[[ -f "$ARCHIVE" ]] || { echo "missing Git-tracked V23 data archive: $ARCHIVE" >&2; exit 2; }
actual_sha="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
[[ "$actual_sha" == "$ARCHIVE_SHA256" ]] || { echo "V23 archive hash mismatch" >&2; exit 2; }

mkdir -p "$WORK" "$WORK/data"
if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools
requirements_sha="$(sha256sum "$REPO/requirements-v23-h100.txt" | awk '{print $1}')"
requirements_marker="$VENV/.v23_requirements_${requirements_sha}_torch2.7.0cu128_complete"
if [[ ! -f "$requirements_marker" ]]; then
  python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
  python -m pip install -r "$REPO/requirements-v23-h100.txt"
  python -m pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
  touch "$requirements_marker"
fi
python - <<'PY'
import accelerate,datasets,numpy,peft,safetensors,scipy,sentence_transformers,sklearn,torch,torch_geometric,tqdm,transformers
import pyg_lib,torch_scatter,torch_sparse
expected={'torch':'2.7.0','transformers':'4.53.3','datasets':'3.6.0','peft':'0.16.0','accelerate':'1.8.1','sentence_transformers':'5.0.0','torch_geometric':'2.6.1','sklearn':'1.7.0','numpy':'1.26.4','scipy':'1.15.3','tqdm':'4.67.1','safetensors':'0.5.3'}
actual={'torch':torch.__version__.split('+')[0],'transformers':transformers.__version__,'datasets':datasets.__version__,'peft':peft.__version__,'accelerate':accelerate.__version__,'sentence_transformers':sentence_transformers.__version__,'torch_geometric':torch_geometric.__version__,'sklearn':sklearn.__version__,'numpy':numpy.__version__,'scipy':scipy.__version__,'tqdm':tqdm.__version__,'safetensors':safetensors.__version__}
assert actual==expected,(actual,expected)
assert torch.version.cuda=='12.8',torch.version.cuda
assert torch.cuda.is_available(),'PyTorch cannot see CUDA'
print({'dependency_contract':'PASS',**actual,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)})
PY
if [[ ! -f "$V23_DATA_DIR/COMBINED_MANIFEST.json" ]]; then
  tar -xzf "$ARCHIVE" -C "$WORK/data"
fi

python "$REPO/SFT/auditor_agent_v23_final_experiments/scripts/preflight_v23_experiments.py" \
  --repo "$REPO" --data "$V23_DATA_DIR" --output "$V23_EXPERIMENT_RUN/PREFLIGHT_BOOTSTRAP.json"
bash "$REPO/SFT/auditor_agent_v23_final_experiments/server_scripts/run_v23_all_experiments.sh"

echo "V23 COMPLETE"
echo "Results: $V23_EXPERIMENT_RUN"
echo "Archive: ${V23_EXPERIMENT_RUN%/}_V23_FINAL_EXPERIMENT_RESULTS.tar.gz"
