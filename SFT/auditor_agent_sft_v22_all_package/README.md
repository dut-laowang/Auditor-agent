# V22-ALL unified multi-source audit pipeline

V22-ALL combines the three completed framework/task-source tracks without
changing their observable evidence:

- `marble_mab`: MARBLE x MultiAgentBench
- `autogen_mab`: AutoGen x MultiAgentBench
- `marble_appworld`: MARBLE x AppWorld

The local bundle contains the unified source training rows and the full frozen
validation rows. The server applies an exact final-chat Qwen context gate again
after the three teacher fields are merged; it never truncates a training chat.
Sealed test files are never read or transported.
Every input row keeps its original `run_id`, messages, metadata, evidence IDs,
and target.  Track membership is stored in a sidecar index rather than injected
into model-visible text.

## Required pre-upload gate

`prepare_v22_all_source.py` performs a full structural, leakage, evidence, and
cross-field semantic-contract audit plus deterministic stratified sampling. It samples 50 rows for every
`track x split x verdict` cell (900 rows total) and writes the sampled rows for
review. Every sampled row is rechecked for non-empty observable evidence,
verdict/binary/attack consistency, localization validity, candidate/evidence
alignment, and complete audit-trace semantics. The ZIP is created only when every cell and every full-dataset gate
passes.  On failure it writes `QUALITY_FAILURE_REPORT.json` and exits without
an uploadable archive.

Expected current inputs after the existing Qwen 8,192-token gate:

| Track | Train | Validation | Sealed test (not read) |
| --- | ---: | ---: | ---: |
| MARBLE x MultiAgentBench | 5,404 | 1,756 | 1,522 |
| AutoGen x MultiAgentBench | 1,912 | 792 | 624 |
| MARBLE x AppWorld | 3,122 | 406 | 393 |
| **V22-ALL** | **10,438** | **2,954** | **2,539** |

The ModernBERT-only 8,192-token gate trains that inspector on 10,420 rows
(18 overlength AutoGen training documents are ineligible for ModernBERT), but
those 18 rows remain unchanged in the 10,438-row V22/Qwen training split.

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
TEACHER_BATCH=4 TEACHER_MAX_BATCH_TOKENS=34816 MODERN_TRAIN_BATCH=2 MODERN_GRAD_ACCUM=8 \
QWEN_TRAIN_BATCH=1 QWEN_GRAD_ACCUM=16 QWEN_EVAL_BATCH=4 \
bash SFT/auditor_agent_sft_v22_all_package/server_scripts/run_v22_all_unified_server.sh
```

Set `V22_ALL_SOURCE_BUNDLE=/absolute/path/to/v22_all_source_bundle.zip` only
when intentionally overriding the bundled source artifact.

The pipeline trains one joint ModernBERT Inspector, strictly reuses the already
completed 3,122-row AppWorld Qwen3-32B expansion, expands only the two MAB
tracks, merges both teacher outputs in exact joint-training order, trains one
joint enhanced Qwen3-8B Audit SFT, injects only ModernBERT predictions into
validation, and evaluates both the 2,954-row union and every source track
separately. Any reusable-artifact source hash, ID, order, schema, or leakage
contract mismatch stops the run before new teacher inference.

After the real teacher output is merged, the runner tokenizes the complete
system/user/assistant chat with the pinned Qwen3-8B tokenizer. The pinned model
supports 40,960 positions; this pipeline uses a conservative 12,288-token
budget. No row is truncated or dropped: any over-budget train or validation row
stops the run and is listed by ID, track, verdict, and token count in
`qwen_ready_audit_sft/FINAL_QWEN_CONTEXT_GATE.json`. The final
Qwen training manifest must hash-match these exact filtered files on resume.

If the obsolete runner already created an all-row teacher output, the current
runner automatically quarantines only that teacher output and its downstream
expanded/Qwen artifacts outside the active run under
`$BASE/v22_all_stale_pre_appworld_reuse_*`. It keeps
the source, context-filtered data, joint ModernBERT checkpoint, and ModernBERT
evaluation, then continues in the same run directory. Stale artifacts are not
embedded in the final archive. The completion marker and PASS summary are
written only after the union and all three per-track quality gates pass.

## Two bounded-agent 100-row demo

After the plain-Qwen and V22-ALL cascade validation predictions exist, run:

```bash
V22_ALL_RUN=/gs/bs/tgh-26IAW/hongbo/project_4_coauthor/v22_all_run \
GPU=0 QWEN_EVAL_BATCH=4 \
bash SFT/auditor_agent_sft_v22_all_package/server_scripts/run_v22_all_agent_demo_100.sh
```

This command does not retrain a model or open the sealed test. It uses the same
deterministic, track/verdict-stratified 100 validation rows for both variants,
reuses their base predictions, and calls the corresponding local Qwen checkpoint
only for label-blind policy selections. Defaults cap extra inference at 12% for
plain SFT and 8% for the cascade. Results under
`$V22_ALL_RUN/agent_demo_100` contain ordinary initial/final task metrics,
recheck rate, corrections, corruptions, deltas, and per-row agent traces.
