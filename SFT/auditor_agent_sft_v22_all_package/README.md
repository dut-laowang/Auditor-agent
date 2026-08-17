# V22-ALL unified multi-source audit pipeline

V22-ALL combines the three completed framework/task-source tracks without
changing their observable evidence:

- `marble_mab`: MARBLE x MultiAgentBench
- `autogen_mab`: AutoGen x MultiAgentBench
- `marble_appworld`: MARBLE x AppWorld

The local bundle contains the Qwen-context-eligible training rows and the full
frozen validation rows.  Sealed test files are never read or transported.
Every input row keeps its original `run_id`, messages, metadata, evidence IDs,
and target.  Track membership is stored in a sidecar index rather than injected
into model-visible text.

## Required pre-upload gate

`prepare_v22_all_source.py` performs a full structural/leakage/evidence audit
and deterministic stratified sampling.  It samples 50 rows for every
`track x split x verdict` cell (900 rows total) and writes the sampled rows for
review.  The ZIP is created only when every cell and every full-dataset gate
passes.  On failure it writes `QUALITY_FAILURE_REPORT.json` and exits without
an uploadable archive.

Expected current inputs after the existing Qwen 8,192-token gate:

| Track | Train | Validation | Sealed test (not read) |
| --- | ---: | ---: | ---: |
| MARBLE x MultiAgentBench | 5,404 | 1,756 | 1,522 |
| AutoGen x MultiAgentBench | 1,912 | 792 | 624 |
| MARBLE x AppWorld | 3,122 | 406 | 393 |
| **V22-ALL** | **10,438** | **2,954** | **2,539** |

## Local preparation

```powershell
python SFT/auditor_agent_sft_v22_all_package/scripts/prepare_v22_all_source.py `
  --marble-mab-data D:\path\to\marble\context_filtered_dataset `
  --autogen-mab-data D:\path\to\autogen\context_filtered_dataset `
  --marble-appworld-data D:\path\to\appworld\context_filtered_dataset `
  --output-dir D:\FIRST_COPILOT\plan_e\v22_all_source_bundle
```

The successful output is `v22_all_source_bundle.zip`.

## One-command server run

The quality-gated ZIP is versioned under `source_bundle/`. Pull the repository
and run:

```bash
cd /gs/bs/tgh-26IAW/hongbo/project_4_coauthor/Auditor-agent && \
git pull --ff-only origin main && \
V22_ALL_RUN=/gs/bs/tgh-26IAW/hongbo/project_4_coauthor/v22_all_run \
TEACHER_BATCH=4 MODERN_TRAIN_BATCH=2 MODERN_GRAD_ACCUM=8 \
QWEN_TRAIN_BATCH=2 QWEN_GRAD_ACCUM=8 QWEN_EVAL_BATCH=4 \
bash SFT/auditor_agent_sft_v22_all_package/server_scripts/run_v22_all_unified_server.sh
```

Set `V22_ALL_SOURCE_BUNDLE=/absolute/path/to/v22_all_source_bundle.zip` only
when intentionally overriding the bundled source artifact.

The pipeline trains one joint ModernBERT Inspector, expands only the joint
training targets with Qwen3-32B, trains one joint enhanced Qwen3-8B Audit SFT,
injects only ModernBERT predictions into validation, and evaluates both the
2,954-row union and every source track separately.
