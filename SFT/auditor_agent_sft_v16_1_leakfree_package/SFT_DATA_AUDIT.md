# V16.1 Leak-Free SFT Data Audit

## Final status

- Build validation: passed (`validation = {}`)
- Source JSONL parse errors: Single 0, Dual 0
- Train/test task-group overlap: 0
- Dynamic marker/private-value hits in visible messages: 0
- Static forbidden-pattern hits: 0
- Invalid evidence, component, or localization references: 0
- Exact train/test prompt overlap: 0
- Normalized train/test prompt overlap: 0
- Privileged marker/reference/contrast feature hits: 0
- Legacy three-line proxy-rule accuracy: 44.26% test (majority-class level),
  reduced from 100% in the rejected V16 build
- Word/bigram TF-IDF audit baseline: 57.79% three-class accuracy
- No single structural feature with support >= 20 reaches >= 95% label purity

## Source-label preservation

The 8,040 attacked labels are read from the supplied final-label files and are
not re-judged or rewritten:

- `success`: 2,510 -> `attack_success`
- `failure`: 4,112 -> `attack_failed`
- `not_exposed`: 1,268 -> retained in the exclusion audit
- `ambiguous`: 142 -> retained in the exclusion audit
- `invalid`: 8 -> covered by the non-completed exclusion audit

The training set additionally uses eligible controls:

- eligible private controls: 2,494
- completed clean controls: 160

## Final dataset

- all: 9,276
- train: 7,421
- test: 1,855
- `clean_safe`: 2,654
- `attack_failed`: 4,112
- `attack_success`: 2,510

SHA-256:

- `all.jsonl`: `fbeb101d5dfbf05fcf1cd3f4c58412839280fd5dc53ca2fc5f2bd1d9cd5dfe8b`
- `train.jsonl`: `f90d9b9823f4ad3c4c997d2421e597767c0a9899cdc30fd7b70627f722909487`
- `test.jsonl`: `83ae6f4c146b9d9db0b1d2f28d00193e8833acc4198693acb9595caa992c5361`

## Fifty-sample quality control

The deterministic sample is stratified across scenario, single/dual attack
mode, and target verdict. All 50 passed:

- three-message SFT structure: 50/50
- source-label to target mapping: 50/50
- visible run evidence: 50/50
- graph candidates present: 50/50
- assistant target is valid JSON: 50/50

The machine-readable audit, complete strata, and file hashes are in
`sft_dataset_graph_grounded_v16_1_leakfree/quality_audit_v16_1.json`.
The independent target-proxy results are in
`sft_dataset_graph_grounded_v16_1_leakfree/proxy_leakage_audit_v16_1.json`.

## Evaluation scope

The one-shot server workflow produces:

- three-class accuracy/report;
- binary accuracy/report;
- localization component precision, recall, F1, hit rate, exact match, and
  scope accuracy;
- audit-trace JSON/reference-quality metrics;
- optional V12 common-50 compatibility diagnostics.

Localization remains a projection of source attack placement and is not
presented as independently re-judged causal attribution.
