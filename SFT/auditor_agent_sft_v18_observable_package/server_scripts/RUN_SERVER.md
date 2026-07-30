# Run V18 Observable

From the repository root:

```bash
BASE=/gs/bs/tgh-26IAW/hongbo/project_4_coauthor
REPO=$BASE/Auditor-agent
GPU=0 BASE="$BASE" REPO="$REPO" \
  bash SFT/auditor_agent_sft_v18_observable_package/server_scripts/run_all_v18.sh
```

The workflow restores and verifies the losslessly archived training data, runs
Qwen3-8B LoRA SFT, and evaluates stratified 50, 200, and full test sets.
Training resumes from the latest checkpoint, and evaluation resumes from the
validated predictions prefix after an interrupted QRSH session.
