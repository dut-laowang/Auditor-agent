# Run Qwen3-8B MAS Auditor SFT v3

```bash
cd /gs/bs/tgh-26IAW/hongbo/project_4_coauthor
test -d Auditor-agent || git clone https://github.com/dut-laowang/Auditor-agent.git
cd Auditor-agent && git pull && cd ..

BASE=/gs/bs/tgh-26IAW/hongbo/project_4_coauthor
REPO=$BASE/Auditor-agent
PKG=$REPO/SFT/auditor_agent_sft_v3_package
DATA=$PKG/sft_dataset_graph_grounded_v3
HF_CACHE=$BASE/sft_models/hf_cache
export HF_HOME=$HF_CACHE
export HF_HUB_CACHE=$HF_CACHE/hub
export TRANSFORMERS_CACHE=$HF_CACHE/transformers
export HF_HUB_DISABLE_XET=1
```

Install once:

```bash
python -m pip install -U "transformers>=4.51.0" datasets peft accelerate scikit-learn tqdm
```

Continue SFT from v2 LoRA:

```bash
V2_ADAPTER=$BASE/sft_models/qwen3-8b-mas-auditor-lora-v2
V3_OUT=$BASE/sft_models/qwen3-8b-mas-auditor-lora-v3-from-v2

CUDA_VISIBLE_DEVICES=0 python $PKG/server_scripts/train_qwen3_lora_sft.py \
  --model Qwen/Qwen3-8B \
  --init-adapter $V2_ADAPTER \
  --data-dir $DATA \
  --output-dir $V3_OUT \
  --max-len 4096 \
  --epochs 1 \
  --lr 1e-4 \
  --batch 1 \
  --grad-accum 16
```

Formal v3 SFT from base:

```bash
V3_BASE_OUT=$BASE/sft_models/qwen3-8b-mas-auditor-lora-v3-base

CUDA_VISIBLE_DEVICES=0 python $PKG/server_scripts/train_qwen3_lora_sft.py \
  --model Qwen/Qwen3-8B \
  --data-dir $DATA \
  --output-dir $V3_BASE_OUT \
  --max-len 4096 \
  --epochs 2 \
  --lr 2e-4 \
  --batch 1 \
  --grad-accum 16
```

Evaluate:

```bash
CUDA_VISIBLE_DEVICES=0 python $PKG/server_scripts/eval_qwen3_fullschema.py \
  --mode sft \
  --model Qwen/Qwen3-8B \
  --adapter $V3_OUT \
  --test-file $DATA/test.jsonl \
  --output-dir $BASE/qwen3_8b_sft_v3_from_v2_eval_tok1024 \
  --max-new-tokens 1024

CUDA_VISIBLE_DEVICES=0 python $PKG/server_scripts/eval_qwen3_fullschema.py \
  --mode base \
  --model Qwen/Qwen3-8B \
  --test-file $DATA/test.jsonl \
  --output-dir $BASE/qwen3_8b_base_v3_eval_tok1024 \
  --max-new-tokens 1024
```

`metrics.json` contains verdict F1 and `audit_trace_quality`:

- `valid_json_rate`
- `has_audit_trace_rate`
- `avg_trace_steps`
- `avg_evidence_refs`
- `evidence_ref_validity_rate`

Compare v2/v3 or base/v3 metrics:

```bash
python $PKG/server_scripts/compare_eval_metrics.py \
  --before $BASE/qwen3_8b_sft_v2_eval_tok1024/metrics.json \
  --after $BASE/qwen3_8b_sft_v3_from_v2_eval_tok1024/metrics.json \
  --output $BASE/qwen3_8b_sft_v3_from_v2_eval_tok1024/compare_with_v2.json
```
