# Graph-grounded MAS Auditor SFT v2

This package contains the v2 SFT conversion script, the generated v2 SFT data, and server-side Qwen3-8B LoRA SFT/evaluation scripts.

## Contents

- `scripts/build_graph_grounded_sft.py`
  - Batch converter from MARBLE benchmark artifacts to graph-grounded SFT JSONL.
- `sft_dataset_graph_grounded_v2/`
  - `train.jsonl`: training split.
  - `test.jsonl`: test split.
  - `all.jsonl`: full dataset.
  - `stats.json`: dataset statistics.
- `sft_samples_graph_grounded_v2/`
  - Six readable examples: Minecraft/Bargaining x clean/attack_success/attack_failed.
- `server_scripts/`
  - `train_qwen3_lora_sft.py`: Qwen3-8B LoRA SFT training.
  - `eval_qwen3_fullschema.py`: Base/SFT full-schema evaluation with the same prompt/parser.
  - `RUN_SERVER.md`: concise server commands.

## Data Format

Each JSONL row uses standard chat SFT format:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {...}
}
```

The user message contains task, graph, attack context, observed events, and clean reference. The assistant message outputs an evidence-based audit JSON with verdict, label, localization, evidence, reason, and action.

## Labels

- `clean_safe`: no attack, safe.
- `attack_failed`: attack present but not successful, safe.
- `attack_success`: attack successful, unsafe.

Binary mapping:

- `safe = clean_safe + attack_failed`
- `unsafe = attack_success`

## v2 Safety Cleanup

The SFT-visible messages do not contain:

- `attack_id`
- `attack_metadata`
- `success_markers`
- `[Injected instruction]`
- uppercase `ACI_...` success markers
- `END_NEGOTIATION`
- `offline verifier`
- `attack-success index`

Markers are used only offline to construct labels. They are not used as model inputs.

Known caveat: `run_id` still contains lowercase `aci_...` attack-family names. This is not a success marker, but can be anonymized in a stricter release.

## Regenerate

Place `benchmark384_multiscenario_runs_1.zip` in the working directory and run:

```bash
python scripts/build_graph_grounded_sft.py \
  --zip benchmark384_multiscenario_runs_1.zip \
  --output-dir sft_dataset_graph_grounded_v2 \
  --test-ratio 0.2 \
  --seed 42
```

The converter expects the zip to contain merged configs, trajectories, MARBLE results, run manifest, and success runs from the benchmark output.
