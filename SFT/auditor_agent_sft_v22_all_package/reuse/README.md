# Reused V22 AppWorld teacher artifact

`appworld_qwen32b_enrichment_v4.jsonl` is the completed 3,122-row Qwen3-32B
teacher output from `v22_qwen32b_teacher_expansion_v4_full.tar.gz`. V22-ALL
does not regenerate these rows. The server runner verifies the original source
hash, teacher-output hash, exact AppWorld run-ID order, schema, prompt version,
and no-validation/no-sealed-test contract before starting inference for the two
remaining tracks.
