# V22-ALL unified multi-source audit pipeline

V22-ALL now combines the complete 2 x 2 framework/task-source tracks without
changing their observable evidence:

- `marble_mab`: MARBLE x MultiAgentBench
- `autogen_mab`: AutoGen x MultiAgentBench
- `marble_appworld`: MARBLE x AppWorld
- `autogen_appworld`: AutoGen x AppWorld

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
`track x split x verdict` cell and writes the sampled rows for review. Small
minority cells are reviewed exhaustively. The expanded bundle checks 1,184 rows
in this sampled gate in addition to the full-dataset checks.
review. Every sampled row is rechecked for non-empty observable evidence,
verdict/binary/attack consistency, localization validity, candidate/evidence
alignment, and complete audit-trace semantics. The ZIP is created only when every cell and every full-dataset gate
passes.  On failure it writes `QUALITY_FAILURE_REPORT.json` and exits without
an uploadable archive.

Expected current inputs after the existing Qwen 8,192-token gate:

| Track | Train | Validation | Sealed test (not read) |
| --- | ---: | ---: | ---: |
| MARBLE x MultiAgentBench | 5,404 | 1,756 | 1,522 |
| AutoGen x MultiAgentBench | 6,042 | 2,099 | 1,804 |
| MARBLE x AppWorld | 7,641 | 1,025 | 951 |
| AutoGen x AppWorld | 5,117 | 693 | 619 |
| **V22-ALL expanded** | **24,204** | **5,573** | **4,896** |

The Qwen 12,288-token gate is lossless at these final counts. The independent
ModernBERT 8,192-token user-input gate retains 24,153 train, 5,541 validation,
and 4,866 test rows. The longer Qwen-eligible rows remain in the Plain-Qwen
dataset; the bounded Agent treats them as verifier-unavailable rather than
truncating them.

## Local preparation

```powershell
python SFT/auditor_agent_sft_v22_all_package/scripts/prepare_v22_all_source.py `
  --marble-mab-data D:\path\to\marble\context_filtered_dataset `
  --autogen-mab-data D:\path\to\autogen\context_filtered_dataset `
  --marble-appworld-data D:\path\to\appworld\context_filtered_dataset `
  --autogen-appworld-data D:\path\to\autogen_appworld\context_filtered_dataset `
  --output-dir D:\FIRST_COPILOT\plan_e\v22_all_source_bundle
```

The successful output is `v22_all_source_bundle.zip`.

The expanded assembly is produced by `assemble_v22_all_expanded_tracks.py`.
It inherits the frozen split of every existing task group, removes duplicate
run IDs, assigns deterministic globally unique sample UIDs, and fails on task,
run-ID, or exact-input overlap. Source `ambiguous` and `not_exposed` labels,
leaky private controls, insufficient observable-evidence rows, and over-budget
Qwen chats are never forced into training.

## Expanded Plain-Qwen + verifier Agent run

The primary auditor remains Plain Qwen3-8B SFT. ModernBERT is independently
fine-tuned on its eligible subset and is queried only by the learned bounded
router. Localization remains Qwen-owned except for the existing empty-component
compatibility fallback. The frozen labeled test ZIP must be transferred
separately and is opened only after both training stages finish.

```bash
cd /gs/bs/tgh-26IAW/hongbo/project_4_coauthor/Auditor-agent && \
git pull --ff-only origin main && \
V22_ALL_RUN=/gs/bs/tgh-26IAW/hongbo/project_4_coauthor/v22_all_expanded_run \
V22_ALL_SEALED_TEST_BUNDLE=/absolute/path/v22_all_expanded_sealed_test.zip \
GPU=0 QWEN_TRAIN_BATCH=1 QWEN_GRAD_ACCUM=16 QWEN_EVAL_BATCH=4 \
MODERN_TRAIN_BATCH=2 MODERN_GRAD_ACCUM=8 MODERN_EVAL_BATCH=4 \
bash SFT/auditor_agent_sft_v22_all_package/server_scripts/run_v22_all_expanded_plain_agent_once.sh
```

## Legacy three-track cascade runner (not for the expanded source)

The command below documents the frozen earlier cascade experiment only. Its
hard count/hash guards intentionally reject the expanded source. Use the
expanded Plain-Qwen + verifier Agent command above for current V22-ALL.

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

## Plain-Qwen heterogeneous-verifier Agent: frozen test-300

This is the primary bounded-agent experiment. Plain Qwen remains the auditor;
ModernBERT is consulted only for the 15% of rows assigned the highest Qwen-error
probability by a learned logistic router. A fixed hash split uses 70% of the
validation predictions to train the router and the remaining 30% to train/check
the heterogeneous conflict selector; both are frozen before a deterministic,
stratified 300-row sample is drawn from the common eligible sealed-test pool.
All router features are deployment-visible and no test gold field is used.

```bash
V22_ALL_RUN=/gs/bs/tgh-26IAW/hongbo/project_4_coauthor/v22_all_run \
GPU=0 QWEN_EVAL_BATCH=4 \
bash SFT/auditor_agent_sft_v22_all_package/server_scripts/run_v22_plain_hetero_agent_test300_once.sh
```

Only verdict changes trigger report regeneration with the existing local Plain
Qwen checkpoint. Results and the frozen calibration policy are written under
`$V22_ALL_RUN/plain_hetero_agent_test300`.

After the frozen 300-row pilot passes, evaluate all 2,531 rows jointly eligible
for Plain Qwen and ModernBERT without changing the learned policy or budget:

```bash
V22_ALL_RUN=/gs/bs/tgh-26IAW/hongbo/project_4_coauthor/v22_all_run \
GPU=0 QWEN_EVAL_BATCH=4 \
bash SFT/auditor_agent_sft_v22_all_package/server_scripts/run_v22_plain_hetero_agent_full_test_once.sh
```

The full result uses its own `$V22_ALL_RUN/plain_hetero_agent_full_test`
directory and does not overwrite the pilot.
