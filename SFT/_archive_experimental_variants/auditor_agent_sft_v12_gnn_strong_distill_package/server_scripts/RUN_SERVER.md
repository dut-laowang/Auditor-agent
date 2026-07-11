# V12 GNN-Strong-Distill Server Run

Short command:

```bash
cd /gs/bs/tgh-26IAW/hongbo/project_4_coauthor/Auditor-agent
git pull
cd /gs/bs/tgh-26IAW/hongbo/project_4_coauthor
CUDA_VISIBLE_DEVICES=0 bash Auditor-agent/SFT/auditor_agent_sft_v12_gnn_strong_distill_package/server_scripts/run_all_v12_gnn_strong_distill.sh
```

This runs:

```text
1. Build strong utterance graph dataset from V12
2. Install sentence-transformers if missing
3. Encode node/edge text with all-MiniLM-L6-v2
4. Train edge-aware GAT teacher
5. Build GNN-distilled SFT data
6. Train Qwen3-8B LoRA
7. Evaluate balanced50 and full test
```

Result folders:

```text
/gs/bs/tgh-26IAW/hongbo/project_4_coauthor/gnn_v12_strong_distill_baseline
/gs/bs/tgh-26IAW/hongbo/project_4_coauthor/sft_data_v12_gnn_strong_distill
/gs/bs/tgh-26IAW/hongbo/project_4_coauthor/qwen3_8b_sft_v12_gnn_strong_distill_balanced50_eval_tok1024
/gs/bs/tgh-26IAW/hongbo/project_4_coauthor/qwen3_8b_sft_v12_gnn_strong_distill_full_eval_tok512
```

Compare against V12 balanced50:

```bash
BASE=/gs/bs/tgh-26IAW/hongbo/project_4_coauthor
PKG=$BASE/Auditor-agent/SFT/auditor_agent_sft_v12_gnn_strong_distill_package

python $PKG/server_scripts/compare_eval_metrics.py \
  --before $BASE/qwen3_8b_sft_v12_balanced50_eval_tok1024/metrics.json \
  --after $BASE/qwen3_8b_sft_v12_gnn_strong_distill_balanced50_eval_tok1024/metrics.json \
  --output $BASE/qwen3_8b_sft_v12_gnn_strong_distill_balanced50_eval_tok1024/compare_against_v12_balanced50.json
```
