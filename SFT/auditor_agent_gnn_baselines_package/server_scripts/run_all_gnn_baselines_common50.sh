#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}
REPO=${REPO:-$BASE/Auditor-agent}
PKG=${PKG:-$REPO/SFT/auditor_agent_gnn_baselines_package}
V12_DATA=${V12_DATA:-$REPO/SFT/auditor_agent_sft_v12_package/sft_dataset_graph_grounded_v12}
BALANCED50=${BALANCED50:-$BASE/balanced_common_v8_v9_50/balanced_common_run_ids.txt}

REFS=${REFS:-$BASE/gnn_refs}
GSAFE=${GSAFE:-$REFS/G-safeguard}
BLIND=${BLIND:-$REFS/BlindGuard}
GRAPH_DATA=${GRAPH_DATA:-$BASE/marble_agent_graph_common50}
OUT=${OUT:-$BASE/gnn_vs_sft_common50}

export HF_HOME=${HF_HOME:-$BASE/sft_models/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$BASE/sft_models/hf_cache/transformers}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$BASE/sft_models/hf_cache/transformers}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}

cd "$REPO"
git pull
cd "$BASE"

mkdir -p "$REFS" "$OUT"
if [ ! -d "$GSAFE/.git" ]; then
  git clone https://github.com/wslong20/G-safeguard.git "$GSAFE"
fi
if [ ! -d "$BLIND/.git" ]; then
  git clone https://github.com/MR9812/BlindGuard.git "$BLIND"
fi

python - <<'PY'
import importlib.util
import re
import subprocess
import sys

if importlib.util.find_spec("sentence_transformers") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])

missing = [
    pkg for pkg in ["torch_geometric", "torch_scatter", "torch_sparse"]
    if importlib.util.find_spec(pkg) is None
]
if missing:
    import torch

    torch_base = re.match(r"^(\d+\.\d+\.\d+)", torch.__version__).group(1)
    cuda = torch.version.cuda
    cuda_tag = "cpu" if cuda is None else "cu" + cuda.replace(".", "")
    wheel_url = f"https://data.pyg.org/whl/torch-{torch_base}+{cuda_tag}.html"

    print("Installing missing PyG packages:", missing)
    print("Using PyG wheel index:", wheel_url)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "pyg_lib", "torch_scatter", "torch_sparse",
        "-f", wheel_url,
    ])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch-geometric"])
PY

rm -rf "$GRAPH_DATA" "$OUT/gsafeguard" "$OUT/blindguard"

python "$PKG/server_scripts/build_marble_agent_graph_dataset.py" \
  --sft-data-dir "$V12_DATA" \
  --balanced-ids "$BALANCED50" \
  --output-dir "$GRAPH_DATA"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python "$PKG/server_scripts/run_gsafeguard_adapter.py" \
  --official-ta-dir "$GSAFE/TA" \
  --graph-data-dir "$GRAPH_DATA" \
  --output-dir "$OUT/gsafeguard" \
  --epochs 50 \
  --lr 0.001 \
  --hidden-dim 1024

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python "$PKG/server_scripts/run_blindguard_adapter.py" \
  --official-ta-dir "$BLIND/TA" \
  --graph-data-dir "$GRAPH_DATA" \
  --output-dir "$OUT/blindguard" \
  --epochs 50 \
  --lr 0.001 \
  --hidden-dim 1024 \
  --latent-dim 512

python "$PKG/server_scripts/compare_gnn_vs_sft.py" \
  --v8-metrics "$BASE/qwen3_8b_sft_v8_balanced50_eval_tok1024/metrics.json" \
  --v12-metrics "$BASE/qwen3_8b_sft_v12_balanced50_eval_tok1024/metrics.json" \
  --gsafeguard-metrics "$OUT/gsafeguard/metrics.json" \
  --blindguard-metrics "$OUT/blindguard/metrics.json" \
  --output "$OUT/comparison_table.json"

echo "Done."
echo "Graph data: $GRAPH_DATA"
echo "G-Safeguard-style metrics: $OUT/gsafeguard/metrics.json"
echo "BlindGuard-style metrics: $OUT/blindguard/metrics.json"
echo "Comparison: $OUT/comparison_table.json"
