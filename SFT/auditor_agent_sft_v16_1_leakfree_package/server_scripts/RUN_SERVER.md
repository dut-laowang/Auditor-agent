# Run V16.1 Leak-Free

Upload this package under `Auditor-agent/SFT/`, then run:

```bash
cd /gs/bs/tgh-26IAW/hongbo/project_4_coauthor/Auditor-agent
bash SFT/auditor_agent_sft_v16_1_leakfree_package/server_scripts/run_all_v16_1.sh
```

The script performs:

1. Qwen3-8B LoRA SFT;
2. stratified 50-sample evaluation;
3. full grouped-test evaluation;
4. optional evaluation on the exact V12 common50 file;
5. metric comparison when the V12 common50 metrics path is supplied.

Optional environment variables:

```bash
COMMON50=/path/to/v12_common50.jsonl
V12_COMMON50_METRICS=/path/to/v12/common50/metrics.json
GPU=0
```

Metrics include three-class and binary reports, component localization
precision/recall/F1/hit/exact-match, scope accuracy, and audit-reference validity.
