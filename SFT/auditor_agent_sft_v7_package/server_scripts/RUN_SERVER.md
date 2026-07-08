# Run Qwen3-8B MAS Auditor SFT v7

Pull and set paths:

```bash
cd /gs/bs/tgh-26IAW/hongbo/project_4_coauthor
test -d Auditor-agent || git clone https://github.com/dut-laowang/Auditor-agent.git
cd Auditor-agent && git pull && cd ..

BASE=/gs/bs/tgh-26IAW/hongbo/project_4_coauthor
REPO=$BASE/Auditor-agent
PKG=$REPO/SFT/auditor_agent_sft_v7_package
DATA=$PKG/sft_dataset_graph_grounded_v7

V7_OUT=$BASE/sft_models/qwen3-8b-mas-auditor-lora-v7-main
EVAL_FULL=$BASE/qwen3_8b_sft_v7_main_eval_tok512
EVAL_200=$BASE/qwen3_8b_sft_v7_main_eval_tok512_200

HF_CACHE=$BASE/sft_models/hf_cache
export HF_HOME=$HF_CACHE
export TRANSFORMERS_CACHE=$HF_CACHE/transformers
export HF_HUB_CACHE=$HF_CACHE/transformers
export HF_HUB_DISABLE_XET=1
```

Train v7 from Qwen3-8B:

```bash
CUDA_VISIBLE_DEVICES=0 python $PKG/server_scripts/train_qwen3_lora_sft.py \
  --model Qwen/Qwen3-8B \
  --data-dir $DATA \
  --output-dir $V7_OUT \
  --max-len 4096 \
  --epochs 2 \
  --lr 2e-4 \
  --batch 1 \
  --grad-accum 16
```

Quick 200-sample evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python $PKG/server_scripts/eval_qwen3_fullschema.py \
  --mode sft \
  --model Qwen/Qwen3-8B \
  --adapter $V7_OUT \
  --test-file $DATA/test.jsonl \
  --output-dir $EVAL_200 \
  --max-new-tokens 512 \
  --limit 200
```

Full evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python $PKG/server_scripts/eval_qwen3_fullschema.py \
  --mode sft \
  --model Qwen/Qwen3-8B \
  --adapter $V7_OUT \
  --test-file $DATA/test.jsonl \
  --output-dir $EVAL_FULL \
  --max-new-tokens 512
```

Results:

```bash
cat $EVAL_200/metrics.json
cat $EVAL_FULL/metrics.json
```
