# V17 Semantic-Preserving SFT Data Audit

## Final status

- Build validation passed (`validation = {}`).
- Source JSONL parse errors: 0.
- Exact, normalized-prompt, and task-group train/test overlap: 0.
- Raw source marker and benchmark-identity hits in model-visible messages: 0.
- Redacted entity IDs reused across runs: 0 of 1,335.
- Invalid evidence, component, and localization references: 0.
- Fifty-sample stratified QC: 50/50 passed all structural and source-label checks.

V17 removes benchmark identity and answer-bearing markers while retaining the
ordinary semantics of instructions, communication, tool use, and observable
outcomes. Sensitive values use deterministic-within-run, unlinkable-across-run
`[REDACTED_ENTITY_*]` replacements.

## Source-label preservation

The supplied final labels are not re-judged or rewritten:

- `success`: 2,510 -> `attack_success`
- `failure`: 4,112 -> `attack_failed`
- `not_exposed`: 1,268 -> excluded and recorded
- `ambiguous`: 142 -> excluded and recorded
- `invalid`: 8 -> covered by non-completed exclusions

Eligible controls add 2,494 private-control and 160 clean runs.

## Final dataset

- all: 9,276
- train: 7,421
- test: 1,855
- `clean_safe`: 2,654
- `attack_failed`: 4,112
- `attack_success`: 2,510

SHA-256:

- `all.jsonl`: `6eca075f9a514f39865b277d2c740bc4e6a713841a1acad056efcfb841af84a5`
- `train.jsonl`: `c3ad4c9722d922796203daa769c399648f59c9f157dcc37007d40c9aee3bbc65`
- `test.jsonl`: `4e4cd48370725064b6d6ca873b908bcc86a2858c3f9509d1702747cdf948f213`

## Shortcut diagnostics

- Legacy proxy rule: 44.26% test three-class accuracy.
- Word/bigram TF-IDF proxy: 56.98%.
- Shallow structural proxy: 37.90%.

These diagnostics are not model results; they check whether labels can be
recovered from obvious formatting or metadata shortcuts. Full details are in
`quality_audit_v17.json` and `proxy_audit_v17.json`.

## Evaluation

The server workflow trains Qwen3-8B LoRA and evaluates stratified 50, stratified
200, and the full 1,855 held-out test set. It reports three-class, binary,
localization, scope-specific localization, and evidence-reference metrics.
Localization targets preserve the source attack-placement projection and are
not presented as independently re-judged causal attribution.
