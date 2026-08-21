# V22 legacy (15,931-row) missing-experiment suite

This package adds publication-facing experiments without modifying the expanded
V22 source or any completed checkpoint. It accepts only the frozen legacy
three-track data: 10,438 train, 2,954 validation, and the separately sealed
2,539 test rows. Unknown results are rendered as `TBD`; the code never invents
numbers.

## What is and is not a fair comparison

- **Main table:** same post-hoc full-trajectory task and same frozen split.
  Plain Qwen, ModernBERT, the bounded Agent, G-Safeguard, BlindGuard, and
  XG-Guard adaptations may appear here. Adapted graph methods must be marked as
  adaptations because their native output is malicious-agent anomaly, not our
  three-way outcome/localization schema.
- **Held-out generalization (Supplement A):** train from scratch after removing
  one topology (`tree`), one attacked surface (`message`), or one scenario
  (`research`). The surface fold retains clean controls, because clean runs do
  not have a meaningful attack surface. Default execution uses ModernBERT as a
  cheap pilot; set `METHODS=modernbert,qwen` for the full SFT version.
- **Single-agent/trajectory transfer (Supplement B):** build a V22-Flat view
  from exactly the same 10,438/2,954 run IDs, labels, events, and event order;
  remove topology, typed candidates, and candidate/evidence grounding; and
  train Qwen3-8B with the same optimization contract as V22-Graph. The older
  AgentDoG official/outcome-adapted protocol remains a separately labelled
  transfer diagnostic because its native unsafe label is not outcome-equivalent.
- **AgentForesight (Supplement C):** its official task is prefix-only online
  `CONTINUE/ALARM` prediction with decisive-step labels and Exact-F1/ASS/FAR.
  Our V22 records do not contain verified earliest-decisive-step labels.
  Therefore the official native evaluation may be reproduced separately, but
  it must not be inserted into the post-hoc Acc./Macro-F1 main table. A direct
  V22 comparison becomes valid only after blind annotation of decisive steps.
- **Counterfactual ablations:** retain the existing frozen V19 validation-only
  protocol. Do not generate or tune counterfactuals on the sealed test.

The ACL 2026 Long paper 1407 is the XG-Guard paper itself, so it is represented
once as `XG-Guard (ACL'26, adapted)`, not as two baselines.

## Expected publication tables

The table renderer always creates four sections:

1. Main — matched full-trajectory test: method, Acc., Macro-F1, AS Recall,
   binary Acc., Loc. F1, N.
2. Supplement A — topology/surface/scenario held-out generalization.
3. Supplement B — single-agent/flattened trajectory transfer and controlled
   Flat-vs-Graph comparison.
4. Supplement C/D — native online auditing and bounded-Agent accuracy/cost
   (verification, defer, coverage, extra calls).
5. Supplement E — validation-only SFT counterfactual ablations and deltas.

Run the renderer before GPU experiments to inspect the exact layout; missing
cells remain `TBD`:

```bash
python SFT/auditor_agent_v22_legacy_experiments/scripts/render_experiment_tables.py \
  --run-dir /gs/bs/tgh-26IAW/hongbo/project_4_coauthor/v22_all_run \
  --supplement-dir /gs/bs/tgh-26IAW/hongbo/project_4_coauthor/v22_legacy_supplement_run \
  --output-dir /gs/bs/tgh-26IAW/hongbo/project_4_coauthor/v22_legacy_supplement_run/tables
```

## Final one-click suite (no new Qwen SFT)

```bash
cd /gs/bs/tgh-26IAW/hongbo/project_4_coauthor/Auditor-agent && \
git pull --ff-only origin main && \
V22_LEGACY_RUN=/gs/bs/tgh-26IAW/hongbo/project_4_coauthor/v22_all_run \
V22_SUPPLEMENT_RUN=/gs/bs/tgh-26IAW/hongbo/project_4_coauthor/v22_legacy_supplement_run \
GPU=0 RUN_BASE_LLMS=1 RUN_EXTERNAL_BASELINES=1 RUN_AGENT_FULL=1 \
bash SFT/auditor_agent_v22_legacy_experiments/server_scripts/run_v22_legacy_all_missing_experiments.sh
```

The runner prints `task i/N`, an overall progress bar, elapsed time, estimated
remaining time, and live output from the active task. It reuses completed
checkpoints and does not retrain Qwen SFT, V18 transfer, or V19 counterfactual
models. The only default retraining is the three smaller ModernBERT held-out
folds; Qwen3-8B/32B Base are inference-only and resumable. Transfer back the
final `V22_LEGACY_FINAL_RESULTS.tar.gz`; it contains
the Markdown/LaTeX five-table bundle, status JSON, progress record, and logs.

`TABLE_STATUS.json` can be `INCOMPLETE` even when the suite itself passes. This
means an external/API row remains `TBD`, never that a value was inferred. In
particular GPT-4.1 requires an explicit API key/budget, while AgentForesight remains
in Table 3 under its native published Exact-F1/ASS/FAR protocol and is not mixed
with V22 post-hoc metrics.

`RUN_AGENTFORESIGHT_NATIVE=1` additionally reproduces the official external
paper-test protocol and requires `AGENTFORESIGHT_MODEL` and
`AGENTFORESIGHT_DATA`. It never changes V22 predictions.
Set `RUN_XGGUARD_TEST=1` only after its validation thresholds are frozen; this
opens the already sealed legacy test once and writes a separate test directory.
