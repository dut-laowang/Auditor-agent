# Server Run Commands

Clone the repo and use the package under `SFT/`:

```bash
cd /gs/bs/tgh-26IAW/hongbo/project_4_coauthor
git clone https://github.com/dut-laowang/Auditor-agent.git
```

Set paths:

```bash
BASE=/gs/bs/tgh-26IAW/hongbo/project_4_coauthor
REPO=$BASE/Auditor-agent
PKG=$REPO/SFT/auditor_agent_sft_v2_package
DATA=$PKG/sft_dataset_graph_grounded_v2
MODEL_OUT=$BASE/sft_models/qwen3-8b-mas-auditor-lora-v2
HF_CACHE=$BASE/sft_models/hf_cache

export HF_HOME=$HF_CACHE
export HF_HUB_CACHE=$HF_CACHE/hub
export TRANSFORMERS_CACHE=$HF_CACHE/transformers
export HF_HUB_DISABLE_XET=1
```

Install dependencies in an isolated environment:

```bash
conda create -n mas_sft python=3.10 -y
conda activate mas_sft
pip install -U torch transformers datasets accelerate peft scikit-learn tqdm
```

Train Qwen3-8B LoRA SFT:

```bash
CUDA_VISIBLE_DEVICES=0 python $PKG/server_scripts/train_qwen3_lora_sft.py \
  --model Qwen/Qwen3-8B \
  --data-dir $DATA \
  --output-dir $MODEL_OUT \
  --max-len 4096 \
  --epochs 2 \
  --lr 2e-4 \
  --batch 1 \
  --grad-accum 16
```

Evaluate SFT with the original full-schema SFT prompt:

```bash
CUDA_VISIBLE_DEVICES=0 python $PKG/server_scripts/eval_qwen3_fullschema.py \
  --mode sft \
  --model Qwen/Qwen3-8B \
  --adapter $MODEL_OUT \
  --test-file $DATA/test.jsonl \
  --output-dir $BASE/qwen3_8b_sft_v2_eval_tok1024 \
  --max-new-tokens 1024
```

Evaluate Base with the same full-schema prompt and generation budget:

```bash
CUDA_VISIBLE_DEVICES=0 python $PKG/server_scripts/eval_qwen3_fullschema.py \
  --mode base \
  --model Qwen/Qwen3-8B \
  --test-file $DATA/test.jsonl \
  --output-dir $BASE/qwen3_8b_base_v2_eval_tok1024 \
  --max-new-tokens 1024
```

Outputs:

```text
$BASE/qwen3_8b_sft_v2_eval_tok1024/metrics.json
$BASE/qwen3_8b_sft_v2_eval_tok1024/predictions.jsonl
$BASE/qwen3_8b_base_v2_eval_tok1024/metrics.json
$BASE/qwen3_8b_base_v2_eval_tok1024/predictions.jsonl
```

The strict comparison keeps the same test set, prompt, parser, and generation budget. The only model-side variable is whether the LoRA adapter is loaded.
