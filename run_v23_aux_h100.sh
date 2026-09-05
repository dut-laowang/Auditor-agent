#!/usr/bin/env bash
set -euo pipefail

ROOT=/gs/bs/tgh-26IAW/hongbo/project_4_coauthor
MAIN_RELEASE="${V23_MAIN_RELEASE:-$ROOT/releases/v23-full-3f65adb-r2}"
REPO="${V23_REPO:-$MAIN_RELEASE/repo}"
RUNTIME="${V23_RUNTIME:-$ROOT/runtime/v23_3f65adb}"
WORK="$RUNTIME/work"
VENV="$WORK/venv"
export REPO

[[ "$(realpath -e -- "$ROOT")" == "$ROOT" ]]
[[ "$(realpath -e -- "$REPO")" == "$REPO" ]]
[[ "$(realpath -e -- "$RUNTIME")" == "$RUNTIME" ]]
[[ -x "$VENV/bin/python" ]] || { echo "missing validated V23 venv: $VENV" >&2; exit 2; }

export BASE="$WORK"
export V23_DATA_DIR="$WORK/data/v23_final_aligned_combined"
export V23_RUN="$WORK/v23_final_run"
export V23_EXPERIMENT_RUN="$WORK/v23_final_experiment_run"
export TMPDIR="$RUNTIME/tmp"
export HF_HOME="$RUNTIME/cache/huggingface"
export HF_HUB_CACHE="$RUNTIME/cache/huggingface/hub"
export PIP_CACHE_DIR="$RUNTIME/cache/pip"
export TORCH_EXTENSIONS_DIR="$RUNTIME/cache/torch_extensions"
export CUDA_CACHE_PATH="$RUNTIME/cache/cuda"
export TRITON_CACHE_DIR="$RUNTIME/cache/triton"
export XDG_CACHE_HOME="$RUNTIME/cache/xdg"
export NUMBA_CACHE_DIR="$RUNTIME/cache/numba"
export PYTHONNOUSERSITE=1
export PATH="$VENV/bin:$PATH"

CORE="$REPO/SFT/auditor_agent_sft_v23_final_package"
EXP="$REPO/SFT/auditor_agent_v23_final_experiments"
LOGS="$V23_EXPERIMENT_RUN/logs"
mkdir -p "$LOGS"

case "${1:-}" in
  --qwen-heldout)
    export GPU="${2:?GPU id required}"
    exec > >(tee -a "$LOGS/aux_qwen_surface_heldout.log") 2>&1
    METHODS=qwen HELDOUT_SPECS=surface__message HELDOUT_SKIP_BUILD=1 HELDOUT_SKIP_FINALIZE=1 \
      bash "$EXP/server_scripts/run_v23_heldout_suite.sh"
    exit 0
    ;;
  --modern-heldout)
    export GPU="${2:?GPU id required}"
    exec > >(tee -a "$LOGS/aux_modern_heldout.log") 2>&1
    bash "$CORE/server_scripts/run_v23_modernbert_once.sh"
    METHODS=modernbert HELDOUT_SKIP_BUILD=1 bash "$EXP/server_scripts/run_v23_heldout_suite.sh"
    exit 0
    ;;
  --graph-baselines)
    export GPU="${2:?GPU id required}"
    exec > >(tee -a "$LOGS/aux_graph_baselines.log") 2>&1
    GNN_METHODS=gat bash "$EXP/server_scripts/run_v22_official_gsafe_tam_once.sh"
    GNN_METHODS=tam bash "$EXP/server_scripts/run_v22_official_gsafe_tam_once.sh"
    bash "$EXP/server_scripts/run_v22_official_xgguard_once.sh"
    bash "$EXP/server_scripts/run_v23_official_blindguard_once.sh"
    exit 0
    ;;
esac

IFS=',' read -r -a gpu_ids <<< "${V23_AUX_GPUS:-0,1}"
[[ "${#gpu_ids[@]}" == 2 ]] || { echo 'V23_AUX_GPUS must contain exactly two GPU ids' >&2; exit 2; }
for gpu in "${gpu_ids[@]}"; do
  gpu="${gpu//[[:space:]]/}"
  mib=$(nvidia-smi --id="$gpu" --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d ' ')
  [[ "$mib" =~ ^[0-9]+$ && "$mib" -ge 90000 ]] || { echo "GPU $gpu is not a full 96GB H100" >&2; exit 2; }
done

python "$EXP/scripts/preflight_v23_experiments.py" \
  --repo "$REPO" --data "$V23_DATA_DIR" --output "$V23_EXPERIMENT_RUN/AUX_PREFLIGHT.json"
python "$EXP/scripts/build_heldout_splits.py" --data-dir "$V23_DATA_DIR" \
  --output-dir "$V23_EXPERIMENT_RUN/heldout_data" --modernbert-zero-truncation

setsid bash "$0" --qwen-heldout "${gpu_ids[0]}" &
qwen_pid=$!
setsid bash "$0" --modern-heldout "${gpu_ids[1]}" &
modern_pid=$!
setsid bash "$0" --graph-baselines "${gpu_ids[1]}" &
graph_pid=$!

terminate_both() {
  trap - INT TERM EXIT
  kill -TERM -- "-$qwen_pid" "-$modern_pid" "-$graph_pid" 2>/dev/null || true
  wait "$qwen_pid" "$modern_pid" "$graph_pid" 2>/dev/null || true
}
trap 'terminate_both; exit 130' INT TERM

declare -A running=( ["$qwen_pid"]=1 ["$modern_pid"]=1 ["$graph_pid"]=1 )
while ((${#running[@]})); do
  finished=''
  set +e
  wait -n -p finished "${!running[@]}"
  status=$?
  set -e
  [[ -n "$finished" ]] && unset 'running[$finished]'
  if [[ "$status" -ne 0 ]]; then terminate_both; exit "$status"; fi
done

python - "$V23_RUN" "$V23_EXPERIMENT_RUN" <<'PY'
import json,sys
from pathlib import Path
run,exp=map(Path,sys.argv[1:])
required=[
 run/'MODERNBERT_COMPLETE.json',
 exp/'SUPPLEMENT_SUITE_COMPLETE.json',
 exp/'baselines/gsafeguard_official_v23_v1/test/metrics.json',
 exp/'baselines/tam_official_v23_v1/test/metrics.json',
 exp/'baselines/xgguard_official_v23_v2/test/metrics.json',
 exp/'baselines/blindguard_official_v23_v1/test/metrics.json',
 exp/'heldout/surface__message/qwen/metrics.json',
]
missing=[str(p) for p in required if not p.is_file()]
if missing: raise SystemExit(f'auxiliary completion contract failed; missing: {missing}')
out={'status':'PASS','version':'V23-aux-independent-v2','artifacts':[str(p) for p in required],
     'generative_heldout':'surface=message / Qwen3-8B SFT'}
(exp/'V23_AUX_INDEPENDENT_COMPLETE.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
PY
