# V19 data integrity contract

Mandatory gates before training:

1. zero run ID, exact-input, normalized-input, and task-group overlap across
   train/validation/test;
2. zero privileged label/marker/judge fields in user messages;
3. every component/evidence reference resolves against the current input;
4. dual-site distributions are reported separately, including the number of
   projected component IDs;
5. lexical proxy and counterfactual ablations use validation only;
6. `test.jsonl` is never passed to training or model selection;
7. all dataset, adapter, metric, prediction, log, and run-manifest hashes are
   written after generation.

Human review is deferred. Disputed, judge-unavailable, and silver examples are
placed in a stratified queue with `review_status=pending_human_review`; no
manual-QC pass claim is permitted until decisions and reviewer provenance exist.
