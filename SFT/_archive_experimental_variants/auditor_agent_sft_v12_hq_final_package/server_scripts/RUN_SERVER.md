# V12-HQ Server Run

One-shot command:

```bash
cd /gs/bs/tgh-26IAW/hongbo/project_4_coauthor/Auditor-agent
git pull
cd /gs/bs/tgh-26IAW/hongbo/project_4_coauthor
CUDA_VISIBLE_DEVICES=0 bash Auditor-agent/SFT/auditor_agent_sft_v12_hq_final_package/server_scripts/run_all_v12_hq_final.sh
```

Result folders:

```text
/gs/bs/tgh-26IAW/hongbo/project_4_coauthor/sft_models/qwen3-8b-mas-auditor-lora-v12-hq-final
/gs/bs/tgh-26IAW/hongbo/project_4_coauthor/qwen3_8b_sft_v12_hq_final_balanced50_eval_tok1024
/gs/bs/tgh-26IAW/hongbo/project_4_coauthor/qwen3_8b_sft_v12_hq_final_full_eval_tok512
```

Compare with V12:

```bash
BASE=/gs/bs/tgh-26IAW/hongbo/project_4_coauthor
PKG=$BASE/Auditor-agent/SFT/auditor_agent_sft_v12_hq_final_package

python $PKG/server_scripts/compare_eval_metrics.py \
  --before $BASE/qwen3_8b_sft_v12_balanced50_eval_tok1024/metrics.json \
  --after $BASE/qwen3_8b_sft_v12_hq_final_balanced50_eval_tok1024/metrics.json \
  --output $BASE/qwen3_8b_sft_v12_hq_final_balanced50_eval_tok1024/compare_against_v12_balanced50.json
```
