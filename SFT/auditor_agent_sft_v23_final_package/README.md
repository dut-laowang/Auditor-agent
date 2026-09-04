# V23 final expanded dataset

The authoritative dataset is `plan_e/data-812/v23_final_aligned` (four tracks)
and `plan_e/data-812/v23_final_aligned_combined` (combined SFT files).

V23 contains 43,844 rows: 30,619 train, 7,018 validation, and 6,207 test.
All 34,673 V22 rows are preserved byte-for-byte. The 9,171 appended rows use
the same row keys, metadata keys, message roles, user schema, target schemas,
and system prompt as V22. No new per-row provenance field is introduced.

One-command Qwen SFT on the server:

```bash
V23_DATA_DIR=/absolute/path/v23_final_aligned_combined \
V23_RUN=/absolute/path/v23_final_run GPU=0 \
bash SFT/auditor_agent_sft_v23_final_package/server_scripts/run_v23_plain_qwen_sft_once.sh
```

The runner fails before GPU allocation on a train/validation count, content
hash, or track-index mismatch. It pins Qwen/Qwen3-8B and its revision, uses the
audited 12,288-token no-truncation contract, supports exact resume validation,
and opens test only after training and validation finish.

Architecture-transfer SFT uses the text-only `internlm/internlm3-8b-instruct`
model (not InternVL). InternLM receives an independent full tokenizer audit
before GPU allocation and otherwise follows the same split, supervision,
optimization defaults, seed, metrics, and test-access order:

```bash
V23_DATA_DIR=/absolute/path/v23_final_aligned_combined \
V23_RUN=/absolute/path/v23_final_run GPU=0 \
bash SFT/auditor_agent_sft_v23_final_package/server_scripts/run_v23_internlm3_sft_once.sh
```

The standalone commands above run one model at a time. The repository-root
`run_v23_h100.sh` entry point instead uses explicit memory reservations on a
96GB H100 and may co-schedule Qwen and InternLM; seeds, data order, model
revisions, and optimization settings remain independent. Set
`V23_GPU_MEMORY_UTILIZATION` lower if the provider reserves unusual amounts of
device memory.
