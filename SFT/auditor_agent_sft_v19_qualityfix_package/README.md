# MAS Auditor SFT V19 Quality Fix

V19 is a conservative quality revision of V18. The Qwen3-8B + LoRA model,
observable-input contract, three-way verdict, attack characterization, graph
localization, and JSON audit report remain unchanged.

## Three independent experiment tracks

- `marble_only`: legacy chain/graph/tree plus fixed star;
- `autogen_only`: aligned AutoGen selector-group-chat/swarm data only;
- `mixed`: the union of the two tracks.

All three use the identical `Graph-grounded-Candidate-SFT/v13` recursive schema.
The final report contains three independent result rows plus an unweighted
cross-track average; the average is not a fourth model.

Each track is released as `dataset_jsonl.zip`. The server runner restores and
SHA-256 verifies `train.jsonl`, `validation.jsonl`, and `test.jsonl` before use;
compression changes transport only, not sample bytes.

## Source policy

- retain legacy MARBLE `chain`, `graph`, and `tree` runs;
- replace legacy `star` runs with `data-89/weekend_fresh_star_fixed.zip`;
- add `data-89/autogen_native_minimal_20260806.tar.zst` as cross-framework data.

The two `data-89` archives do **not** fully replace the old bundle: the new
MARBLE archive contains only `star`, while AutoGen contains different native
coordination modes and explicitly excludes MARBLE data.

## Quality fixes

- dual placements project every resolvable source component instead of falling
  back wholesale to `G::run`;
- train/validation/test are grouped by `(scenario, sample_id)` across all
  frameworks and topologies;
- training reads train + validation only;
- final test evaluation requires an explicit `FINAL_ONCE` acknowledgement and
  writes a consumption record;
- evaluation uses deterministic greedy decoding with configurable batching
  (`EVAL_BATCH_SIZE=4` in the one-click runner), records the batch size in new
  metric files, and resumes from the validated prediction prefix;
- evidence references are selected from localized candidate events and final
  observable outcomes;
- validation-only lexical shortcut and counterfactual ablation scripts are
  included;
- deterministic seeds and a saved run manifest are required.

`manual_review_queue_200_v19.json` is intentionally marked `PENDING`. It is a
review queue, not evidence that human quality control has been completed.

## Additional controlled MARBLE baselines

The Qwen3-32B QLoRA and ModernBERT-8192 multi-task baselines, including two
independent single-GPU runners and the sealed-test command, are documented in
[`BASELINES_QWEN32B_MODERNBERT.md`](BASELINES_QWEN32B_MODERNBERT.md).
