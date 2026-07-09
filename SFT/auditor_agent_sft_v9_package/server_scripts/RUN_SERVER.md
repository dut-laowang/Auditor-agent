# Run Qwen3-8B MAS Auditor SFT v9

```bash
cd /gs/bs/tgh-26IAW/hongbo/project_4_coauthor

BASE=/gs/bs/tgh-26IAW/hongbo/project_4_coauthor
REPO=$BASE/Auditor-agent
PKG=$REPO/SFT/auditor_agent_sft_v9_package
DATA=$PKG/sft_dataset_graph_grounded_v9
OUT=$BASE/sft_models/qwen3-8b-mas-auditor-lora-v9-main
EVAL=$BASE/qwen3_8b_sft_v9_main_eval_tok512
EVAL200=$BASE/qwen3_8b_sft_v9_main_eval_tok512_200

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
rm -rf $OUT $EVAL $EVAL200
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

Quick 200-sample eval:

```bash
CUDA_VISIBLE_DEVICES=0 python $PKG/server_scripts/eval_qwen3_fullschema.py \
  --mode sft \
  --model Qwen/Qwen3-8B \
  --adapter $OUT \
  --test-file $DATA/test.jsonl \
  --output-dir $EVAL200 \
  --max-new-tokens 512 \
  --limit 200
```

Full eval:

```bash
CUDA_VISIBLE_DEVICES=0 python $PKG/server_scripts/eval_qwen3_fullschema.py \
  --mode sft \
  --model Qwen/Qwen3-8B \
  --adapter $OUT \
  --test-file $DATA/test.jsonl \
  --output-dir $EVAL \
  --max-new-tokens 512
```
