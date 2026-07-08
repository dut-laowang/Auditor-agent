# Run Qwen3-8B MAS Auditor SFT v5

Setup:

```bash
cd /gs/bs/tgh-26IAW/hongbo/project_4_coauthor
test -d Auditor-agent || git clone https://github.com/dut-laowang/Auditor-agent.git
cd Auditor-agent && git pull && cd ..

BASE=/gs/bs/tgh-26IAW/hongbo/project_4_coauthor
REPO=$BASE/Auditor-agent
PKG=$REPO/SFT/auditor_agent_sft_v5_package
DATA=$PKG/sft_dataset_graph_grounded_v5

V5_OUT=$BASE/sft_models/qwen3-8b-mas-auditor-lora-v5-main
V4_ADAPTER=$BASE/sft_models/qwen3-8b-mas-auditor-lora-v4-main
V2_ADAPTER=$BASE/sft_models/qwen3-8b-mas-auditor-lora-v2

HF_CACHE=$BASE/sft_models/hf_cache
export HF_HOME=$HF_CACHE
export TRANSFORMERS_CACHE=$HF_CACHE/transformers
export HF_HUB_CACHE=$HF_CACHE/transformers
export HF_HUB_DISABLE_XET=1
```

Install once:

```bash
python -m pip install -U "transformers>=4.51.0" datasets peft accelerate scikit-learn tqdm
python -m pip uninstall -y kernels
```

Train v5 directly from Qwen3-8B:

```bash
CUDA_VISIBLE_DEVICES=0 python $PKG/server_scripts/train_qwen3_lora_sft.py \
  --model Qwen/Qwen3-8B \
  --data-dir $DATA \
  --output-dir $V5_OUT \
  --max-len 4096 \
  --epochs 2 \
  --lr 2e-4 \
  --batch 1 \
  --grad-accum 16
```

Evaluate v5 on v5 test:

```bash
CUDA_VISIBLE_DEVICES=0 python $PKG/server_scripts/eval_qwen3_fullschema.py \
  --mode sft \
  --model Qwen/Qwen3-8B \
  --adapter $V5_OUT \
  --test-file $DATA/test.jsonl \
  --output-dir $BASE/qwen3_8b_sft_v5_main_eval_tok512 \
  --max-new-tokens 512
```

Evaluate v4 on the same v5 test:

```bash
CUDA_VISIBLE_DEVICES=0 python $PKG/server_scripts/eval_qwen3_fullschema.py \
  --mode sft \
  --model Qwen/Qwen3-8B \
  --adapter $V4_ADAPTER \
  --test-file $DATA/test.jsonl \
  --output-dir $BASE/qwen3_8b_sft_v4_on_v5_test_eval_tok512 \
  --max-new-tokens 512
```

Optional v2 baseline on the same v5 test:

```bash
CUDA_VISIBLE_DEVICES=0 python $PKG/server_scripts/eval_qwen3_fullschema.py \
  --mode sft \
  --model Qwen/Qwen3-8B \
  --adapter $V2_ADAPTER \
  --test-file $DATA/test.jsonl \
  --output-dir $BASE/qwen3_8b_sft_v2_on_v5_test_eval_tok512 \
  --max-new-tokens 512
```

Compare:

```bash
python $PKG/server_scripts/compare_eval_metrics.py \
  --before $BASE/qwen3_8b_sft_v4_on_v5_test_eval_tok512/metrics.json \
  --after $BASE/qwen3_8b_sft_v5_main_eval_tok512/metrics.json \
  --output $BASE/qwen3_8b_sft_v5_main_eval_tok512/compare_v4_on_v5_test.json
```

`metrics.json` reports binary/three-class metrics plus audit-trace quality.
