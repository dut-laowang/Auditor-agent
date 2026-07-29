# Run V17 Semantic-Preserving

From the repository root:

```bash
BASE=/root/autodl-tmp/auditor_v17
REPO=$BASE/Auditor-agent
GPU=0 BASE="$BASE" REPO="$REPO" \
  bash SFT/auditor_agent_sft_v17_semantic_package/server_scripts/run_all_v17.sh
```

The workflow performs:

1. Qwen3-8B LoRA SFT for two epochs with effective batch size 16;
2. stratified 50-sample evaluation;
3. stratified 200-sample evaluation;
4. full grouped-test evaluation;
5. optional evaluation on the V12 common50 set.

Outputs are written under `$BASE`, while model and Hugging Face caches stay on
the AutoDL data disk.

The workflow is restart-safe. Training resumes from the newest `checkpoint-*`
under the model output directory, and each evaluation continues from the
validated prefix already stored in `predictions.jsonl`. Re-running the same
command therefore continues an interrupted job instead of discarding progress.
