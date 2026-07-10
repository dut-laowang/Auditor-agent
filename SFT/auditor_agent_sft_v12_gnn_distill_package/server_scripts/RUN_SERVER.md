# One-shot server command

Run from the server. It assumes the repo is under:

```text
/gs/bs/tgh-26IAW/hongbo/project_4_coauthor/Auditor-agent
```

It uses existing V12 data in the repo and writes models/results under `project_4_coauthor`.

Short command:

```bash
cd /gs/bs/tgh-26IAW/hongbo/project_4_coauthor/Auditor-agent
git pull
cd /gs/bs/tgh-26IAW/hongbo/project_4_coauthor
CUDA_VISIBLE_DEVICES=0 bash Auditor-agent/SFT/auditor_agent_sft_v12_gnn_distill_package/server_scripts/run_all_v12_gnn_distill.sh
```

Expanded command:

```bash
cd /gs/bs/tgh-26IAW/hongbo/project_4_coauthor

BASE=/gs/bs/tgh-26IAW/hongbo/project_4_coauthor
REPO=$BASE/Auditor-agent
PKG=$REPO/SFT/auditor_agent_sft_v12_gnn_distill_package
V12_DATA=$REPO/SFT/auditor_agent_sft_v12_package/sft_dataset_graph_grounded_v12

GNN_DATA=$BASE/gnn_dataset_v12_distill
GNN_OUT=$BASE/gnn_v12_distill_baseline
DISTILL_DATA=$BASE/sft_data_v12_gnn_distill
SFT_OUT=$BASE/sft_models/qwen3-8b-mas-auditor-lora-v12-gnn-distill

BALANCED50=$BASE/balanced_common_v8_v9_50
DISTILL_50=$BASE/qwen3_8b_sft_v12_gnn_distill_balanced50_eval_tok1024
DISTILL_FULL=$BASE/qwen3_8b_sft_v12_gnn_distill_full_eval_tok512

export HF_HOME=$BASE/sft_models/hf_cache
export TRANSFORMERS_CACHE=$BASE/sft_models/hf_cache/transformers
export HF_HUB_CACHE=$BASE/sft_models/hf_cache/transformers
export HF_HUB_DISABLE_XET=1

cd $REPO
git pull
cd $BASE

rm -rf $GNN_DATA $GNN_OUT $DISTILL_DATA $SFT_OUT $DISTILL_50 $DISTILL_FULL

python $PKG/server_scripts/build_gnn_graph_dataset.py \
  --data-dir $V12_DATA \
  --output-dir $GNN_DATA

CUDA_VISIBLE_DEVICES=0 python $PKG/server_scripts/train_light_gnn_baseline.py \
  --gnn-data-dir $GNN_DATA \
  --output-dir $GNN_OUT \
  --epochs 20 \
  --lr 1e-3 \
  --hidden-dim 96

python $PKG/server_scripts/build_distilled_sft.py \
  --seed-data-dir $V12_DATA \
  --gnn-score-file $GNN_OUT/all_scores.jsonl \
  --output-dir $DISTILL_DATA

CUDA_VISIBLE_DEVICES=0 python $PKG/server_scripts/train_qwen3_lora_sft.py \
  --model Qwen/Qwen3-8B \
  --data-dir $DISTILL_DATA \
  --output-dir $SFT_OUT \
  --max-len 4096 \
  --epochs 2 \
  --lr 2e-4 \
  --batch 1 \
  --grad-accum 16

python $PKG/server_scripts/make_subset_by_ids.py \
  --all-file $DISTILL_DATA/all.jsonl \
  --ids-file $BALANCED50/balanced_common_run_ids.txt \
  --output-file $BALANCED50/v12_gnn_distill_balanced_common.jsonl

CUDA_VISIBLE_DEVICES=0 python $PKG/server_scripts/eval_qwen3_fullschema.py \
  --mode sft \
  --model Qwen/Qwen3-8B \
  --adapter $SFT_OUT \
  --test-file $BALANCED50/v12_gnn_distill_balanced_common.jsonl \
  --output-dir $DISTILL_50 \
  --max-new-tokens 1024

CUDA_VISIBLE_DEVICES=0 python $PKG/server_scripts/eval_qwen3_fullschema.py \
  --mode sft \
  --model Qwen/Qwen3-8B \
  --adapter $SFT_OUT \
  --test-file $DISTILL_DATA/test.jsonl \
  --output-dir $DISTILL_FULL \
  --max-new-tokens 512
```

Compare quickly against existing V12 balanced50:

```bash
python $PKG/server_scripts/compare_eval_metrics.py \
  --before $BASE/qwen3_8b_sft_v12_balanced50_eval_tok1024/metrics.json \
  --after $DISTILL_50/metrics.json \
  --output $DISTILL_50/compare_against_v12_balanced50.json
```
