# Run Qwen3-8B MAS Auditor SFT V15-HQ

One-shot training + 50-sample evaluation + full test evaluation:

```bash
cd /gs/bs/tgh-26IAW/hongbo/project_4_coauthor

BASE=/gs/bs/tgh-26IAW/hongbo/project_4_coauthor
REPO=$BASE/Auditor-agent
PKG=$REPO/SFT/auditor_agent_sft_v15_hq_current_package
DATA=$PKG/sft_dataset_graph_grounded_v15_hq
OUT=$BASE/sft_models/qwen3-8b-mas-auditor-lora-v15-hq-current

SUBSET_DIR=$BASE/qwen3_8b_sft_v15_hq_current_subset50
SUBSET_FILE=$SUBSET_DIR/v15_hq_current_test50.jsonl
EVAL50=$BASE/qwen3_8b_sft_v15_hq_current_eval50_tok1024
EVALFULL=$BASE/qwen3_8b_sft_v15_hq_current_full_eval_tok512

export HF_HOME=$BASE/sft_models/hf_cache
export TRANSFORMERS_CACHE=$BASE/sft_models/hf_cache/transformers
export HF_HUB_CACHE=$BASE/sft_models/hf_cache/transformers
export HF_HUB_DISABLE_XET=1

cd $REPO
git pull
cd $BASE

rm -rf $OUT $SUBSET_DIR $EVAL50 $EVALFULL

CUDA_VISIBLE_DEVICES=0 python $PKG/server_scripts/train_qwen3_lora_sft.py \
  --model Qwen/Qwen3-8B \
  --data-dir $DATA \
  --output-dir $OUT \
  --max-len 4096 \
  --epochs 2 \
  --lr 2e-4 \
  --batch 1 \
  --grad-accum 16

python $PKG/server_scripts/make_stratified_subset.py \
  --input-file $DATA/test.jsonl \
  --output-file $SUBSET_FILE \
  --n 50 \
  --seed 42

CUDA_VISIBLE_DEVICES=0 python $PKG/server_scripts/eval_qwen3_fullschema.py \
  --mode sft \
  --model Qwen/Qwen3-8B \
  --adapter $OUT \
  --test-file $SUBSET_FILE \
  --output-dir $EVAL50 \
  --max-new-tokens 1024

CUDA_VISIBLE_DEVICES=0 python $PKG/server_scripts/eval_qwen3_fullschema.py \
  --mode sft \
  --model Qwen/Qwen3-8B \
  --adapter $OUT \
  --test-file $DATA/test.jsonl \
  --output-dir $EVALFULL \
  --max-new-tokens 512
```
