# V19 MARBLE zero-truncation baselines: Qwen3-32B and ModernBERT-8192

These baselines consume only the frozen `marble_only` V19 files. AutoGen and
`mixed` are intentionally out of scope. The immutable split hashes are:

| split | rows | SHA-256 |
|---|---:|---|
| train | 4,565 | `d49ec56577a80fab3be360ae9cb1b90e2d751ce686ffa0f0e2064b2d05d0a932` |
| validation | 1,791 | `2882c5adfe2b9c3e7820f7cf56338dec4bf59e091031763ce8ce46d6aa70609e` |
| test | 1,491 | `bee77d962f66f5481e88d89b49b83b3ea9a449e48d776b669ebadd731417167f` |

Both one-click runners restore the ZIP and verify these hashes before training.
Neither runner reads or evaluates `test.jsonl`; final test remains a separate,
explicit one-time action.

The runners also execute three fail-fast gates before model loading: dependency
and CUDA version checks, a deterministic parser/metric self-test, and a complete
data-contract audit covering JSON roles, visible-field leakage markers, target
schema, verdict/binary consistency, candidate coverage, duplicate IDs, split
overlap, row counts, and byte hashes.

After model loading but before optimizer creation, each trainer selects the
longest/worst-memory training rows and performs a real forward and backward
pass. It requires a finite loss plus present, finite, non-zero trainable
gradients, then clears gradients/cache and resets every seed before the formal
run. This catches architecture, LoRA target, tensor-shape, gradient-flow, and
batch-memory failures without updating model weights.

## Environment

Use the existing V19 environment, with recent `transformers`, `peft`,
`datasets`, `scikit-learn`, `accelerate`, and `bitsandbytes`. Qwen3-32B uses
NF4 QLoRA on one GPU. ModernBERT requires a Transformers release that includes
`ModernBertModel` (4.48 or newer).

All downloads are forced below the existing project tree:

```text
/gs/bs/tgh-26IAW/hongbo/project_4_coauthor/sft_models/hf_cache
```

No model or cache is intentionally written under `$HOME`.

## Two independent single-GPU commands

Run Qwen3-32B on one allocated GPU:

```bash
cd /gs/bs/tgh-26IAW/hongbo/project_4_coauthor/Auditor-agent
bash SFT/auditor_agent_sft_v19_qualityfix_package/server_scripts/run_qwen3_32b_marble_v19.sh
```

Run ModernBERT-8192 on another allocated GPU:

```bash
cd /gs/bs/tgh-26IAW/hongbo/project_4_coauthor/Auditor-agent
bash SFT/auditor_agent_sft_v19_qualityfix_package/server_scripts/run_modernbert8192_marble_v19.sh
```

Each command is intended for a separate single-GPU job. The scheduler or job
launcher must expose only the assigned card; the scripts never rewrite
`CUDA_VISIBLE_DEVICES` or select a physical GPU index.

Qwen train and evaluation micro-batches default to 2. If the specific GPU still
runs out of memory, set `TRAIN_BATCH=1 GRAD_ACCUM=16`; this preserves the
effective batch of 16 but must be recorded as a resource fallback. Do not change
the data, seed, epochs, maximum length, prompt template, or LoRA targets.

## Controlled variables

| item | V19 Qwen3-8B | Qwen3-32B | ModernBERT |
|---|---|---|---|
| V19 MARBLE bytes/splits | identical | identical | identical |
| visible fields | system + user | system + user | user JSON only |
| target source | assistant JSON | assistant JSON | assistant JSON parsed into labels |
| maximum length | 8,192 in the corrected zero-truncation runner | 8,192 | 8,192 |
| optimization | LoRA | same LoRA targets/rank, NF4 base | full encoder multitask |
| verdict | generated | generated | 3-class head |
| scope | generated | generated | 6-class head |
| components | generated IDs | generated IDs | dynamic per-candidate head |
| evidence trace | generated | generated | not predicted |

ModernBERT does not receive the SFT output instruction because it is not a
generative model. Its document input is exactly `messages[1].content`; candidate
texts are the unmodified `graph_candidates` objects already present in that
same visible JSON. Metadata is used only after prediction for `run_id` bookkeeping
and never enters the encoder.

All three paths use the same 8,192-token ceiling. This correction is required
because a real V19 MARBLE training sequence was measured at 6,527 Qwen tokens,
which proves that the former 6,144 ceiling was not zero-truncation. The runners enforce a
zero-truncation preflight over the complete split before loading the model. If
even one document, candidate, prompt, or assistant-supervised training sequence
would exceed the ceiling, the run terminates before optimization or generation;
no head-tail retention or silent tokenizer truncation is permitted.

The upstream weights are pinned to immutable Hugging Face commits:

- `Qwen/Qwen3-32B@9216db5781bf21249d130ec9da846c4624c16137`;
- `answerdotai/ModernBERT-base@8949b909ec900327062f0ebf497f51aef5e6f0c8`.

Training and evaluation directories carry machine-checked contracts. Qwen
resume binds the dataset, model revision, quantization, adapter hashes, and
decoding settings. ModernBERT binds the validation-selected component threshold
to the exact checkpoint SHA-256 and refuses an ad-hoc final-test threshold.

The component threshold is selected once on validation by micro-F1 and saved as
`component_threshold.json`. Final test refuses to run without that frozen file.

## Final test (only after validation choices are frozen)

```bash
BASELINE=qwen32b bash SFT/auditor_agent_sft_v19_qualityfix_package/server_scripts/run_final_baseline_test_once_v19.sh
BASELINE=modernbert bash SFT/auditor_agent_sft_v19_qualityfix_package/server_scripts/run_final_baseline_test_once_v19.sh
```

Each output directory receives `SEALED_TEST_CONSUMED.json` and refuses a second
fresh run. Do not invoke these commands during debugging.

## Resource note

Qwen3-32B BF16 weights alone require roughly 64 GB before training states,
activations, and LoRA overhead. The default single-GPU path therefore uses
4-bit NF4 QLoRA and `prepare_model_for_kbit_training`. This changes weight
storage precision relative to the 8B BF16 LoRA run, but preserves the V19 data,
loss mask, chat template, LoRA structure, optimizer schedule, effective batch,
seed, and evaluation contract. A `--quantization none` option exists for a
larger-memory setup, but it is not the one-click default.
