#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"
REPO="${REPO:-$BASE/Auditor-agent}"
PKG="$REPO/SFT/auditor_agent_gnn_baselines_package"
V19="$REPO/SFT/auditor_agent_sft_v19_qualityfix_package"
DATA="$V19/three_track_datasets/marble_only"
REFS="$BASE/gnn_refs"
GSAFE="$REFS/G-safeguard"
BLIND="$REFS/BlindGuard"
XGGUARD="$REFS/XG-Guard"
RESULTS="$BASE/v19_gnn_marble_validation"
CACHE="$BASE/sft_models/v19_gnn_component_cache_v3"

export HF_HOME="${HF_HOME:-$BASE/sft_models/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TOKENIZERS_PARALLELISM=false
export GNN_RESULTS_DIR="$RESULTS"

cd "$REPO"
mkdir -p "$REFS" "$RESULTS" "$CACHE"
# Remove only the obsolete, incorrectly named TAM artifacts from the previous package.
rm -rf "$RESULTS/blindguard_tam_v2_zero_truncation"
rm -f "$RESULTS/blindguard_tam.log"
python "$V19/scripts/restore_track_data.py" "$DATA"
python "$V19/scripts/audit_v19_integrity.py" \
  --data-dir "$DATA" --output "$RESULTS/data_integrity.json"

if [[ ! -d "$GSAFE/.git" ]]; then
  git clone https://github.com/wslong20/G-safeguard.git "$GSAFE"
fi
if [[ ! -d "$BLIND/.git" ]]; then
  git clone https://github.com/MR9812/BlindGuard.git "$BLIND"
fi
if [[ ! -d "$XGGUARD/.git" ]]; then
  git clone https://github.com/CampanulaBells/XG-Guard.git "$XGGUARD"
fi
git -C "$GSAFE" fetch --prune origin
git -C "$BLIND" fetch --prune origin
git -C "$XGGUARD" fetch --prune origin
git -C "$GSAFE" checkout --detach 890c99f1cbc864e9ff0c85859619a14f42bc9cab
git -C "$BLIND" checkout --detach 1889c20a326ba9ba9a6982744d473626e74f9986
git -C "$XGGUARD" checkout --detach 86e1121512f76800f80d4687e492c7f99f049929

python - <<'PY'
import importlib.util
import json
import os
import re
import subprocess
import sys

if importlib.util.find_spec("torch") is None:
    raise RuntimeError("PyTorch must already be installed in the active CUDA environment")
import torch
if importlib.util.find_spec("sentence_transformers") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])
if importlib.util.find_spec("sklearn") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
if importlib.util.find_spec("einops") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "einops"])
missing_pyg = [name for name in ("torch_geometric", "torch_scatter", "torch_sparse") if importlib.util.find_spec(name) is None]
if missing_pyg:
    torch_base = re.match(r"^(\d+\.\d+\.\d+)", torch.__version__).group(1)
    cuda_tag = "cpu" if torch.version.cuda is None else "cu" + torch.version.cuda.replace(".", "")
    wheel_url = f"https://data.pyg.org/whl/torch-{torch_base}+{cuda_tag}.html"
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "pyg_lib", "torch_scatter", "torch_sparse",
        "-f", wheel_url,
    ])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch-geometric==2.6.1"])
import einops
import sentence_transformers
import sklearn
import torch_geometric
import torch_scatter
import torch_sparse
if not torch.cuda.is_available():
    raise RuntimeError("The scheduler must expose one CUDA GPU")
environment = {
    "status": "PASS",
    "gpu": torch.cuda.get_device_name(0),
    "visible_gpu_count": torch.cuda.device_count(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "torch_geometric": torch_geometric.__version__,
    "sentence_transformers": sentence_transformers.__version__,
    "sklearn": sklearn.__version__,
    "einops": einops.__version__,
}
with open(os.path.join(os.environ["GNN_RESULTS_DIR"], "environment.json"), "w", encoding="utf-8") as handle:
    json.dump(environment, handle, indent=2)
print(environment)
PY

python "$PKG/scripts/selftest_v19_gnn_logic.py"

python "$PKG/server_scripts/v19_component_gnn_multitask.py" train-validation \
  --model-kind gat --official-dir "$GSAFE/TA" --data-dir "$DATA" \
  --cache-dir "$CACHE" --output-dir "$RESULTS/gsafeguard_gat_v2_zero_truncation" \
  --epochs 20 --lr 0.001 --hidden-dim 512 --latent-dim 256 \
  2>&1 | tee "$RESULTS/gsafeguard_gat.log"

python "$PKG/server_scripts/v19_unsupervised_graph_baselines.py" train-validation \
  --model-kind blindguard --official-dir "$BLIND/MA" --data-dir "$DATA" \
  --cache-dir "$CACHE/bilevel" --output-dir "$RESULTS/blindguard_scl_v19" \
  --epochs 50 --batch-size 8 --lr 0.001 --weight-decay 0.0001 \
  --hidden-dim 512 --latent-dim 256 --seed 3701 \
  2>&1 | tee "$RESULTS/blindguard_scl.log"

python "$PKG/server_scripts/v19_unsupervised_graph_baselines.py" train-validation \
  --model-kind xgguard --official-dir "$XGGUARD" --data-dir "$DATA" \
  --cache-dir "$CACHE/bilevel" --output-dir "$RESULTS/xgguard_bilevel_v19" \
  --epochs 50 --batch-size 8 --lr 0.00001 --weight-decay 0.0 \
  --alpha 0.0001 --seed 3701 \
  2>&1 | tee "$RESULTS/xgguard_bilevel.log"

python "$PKG/scripts/summarize_v19_gnn_results.py" \
  --gat "$RESULTS/gsafeguard_gat_v2_zero_truncation/metrics.json" \
  --blindguard "$RESULTS/blindguard_scl_v19/metrics.json" \
  --xgguard "$RESULTS/xgguard_bilevel_v19/metrics.json" \
  --output "$RESULTS/comparison_rows.json"

python "$V19/scripts/write_sha256_manifest.py" "$RESULTS"
tar -czf "$BASE/v19_gnn_marble_validation.tar.gz" -C "$BASE" "$(basename "$RESULTS")"
echo "V19 GNN training and validation complete. Final test remains sealed."
echo "Results: $RESULTS"
echo "Transfer: $BASE/v19_gnn_marble_validation.tar.gz"
