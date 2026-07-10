# Run Qwen3-8B MAS Auditor SFT v12

```bash
cd /gs/bs/tgh-26IAW/hongbo/project_4_coauthor

BASE=/gs/bs/tgh-26IAW/hongbo/project_4_coauthor
REPO=$BASE/Auditor-agent
PKG=$REPO/SFT/auditor_agent_sft_v12_package
DATA=$PKG/sft_dataset_graph_grounded_v12
OUT=$BASE/sft_models/qwen3-8b-mas-auditor-lora-v12-main

BALANCED50=$BASE/balanced_common_v8_v9_50
V12_50=$BASE/qwen3_8b_sft_v12_balanced50_eval_tok1024
V12_FULL=$BASE/qwen3_8b_sft_v12_main_eval_tok512

export HF_HOME=$BASE/sft_models/hf_cache
export TRANSFORMERS_CACHE=$BASE/sft_models/hf_cache/transformers
export HF_HUB_CACHE=$BASE/sft_models/hf_cache/transformers
export HF_HUB_DISABLE_XET=1

cd $REPO
git pull
cd $BASE
```

Train:

```bash
rm -rf $OUT $V12_50 $V12_FULL
CUDA_VISIBLE_DEVICES=0 python $PKG/server_scripts/train_qwen3_lora_sft.py \
  --model Qwen/Qwen3-8B \
  --data-dir $DATA \
  --output-dir $OUT \
  --max-len 4096 \
  --epochs 2 \
  --lr 2e-4 \
  --batch 1 \
  --grad-accum 16
```

Balanced 50 eval:

```bash
python $PKG/server_scripts/make_subset_by_ids.py \
  --all-file $DATA/all.jsonl \
  --ids-file $BALANCED50/balanced_common_run_ids.txt \
  --output-file $BALANCED50/v12_balanced_common.jsonl

CUDA_VISIBLE_DEVICES=0 python $PKG/server_scripts/eval_qwen3_fullschema.py \
  --mode sft \
  --model Qwen/Qwen3-8B \
  --adapter $OUT \
  --test-file $BALANCED50/v12_balanced_common.jsonl \
  --output-dir $V12_50 \
  --max-new-tokens 1024
```

Full eval:

```bash
CUDA_VISIBLE_DEVICES=0 python $PKG/server_scripts/eval_qwen3_fullschema.py \
  --mode sft \
  --model Qwen/Qwen3-8B \
  --adapter $OUT \
  --test-file $DATA/test.jsonl \
  --output-dir $V12_FULL \
  --max-new-tokens 512
```
