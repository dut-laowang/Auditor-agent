# V20 AutoGen Core Experiment

This track uses only `autogen_native_complete_with_configs_20260813.tar.zst`.
It applies the V19/V20 observable-input schema, label mapping, G/N/E/T
projection, seed-42 task-grouped split, model revisions, and hyperparameters.
The V2 observable adapter reads actual `delivered_content` from original AutoGen
trajectory logs, excludes privileged attack instrumentation, restores substantive
final outcomes, and rejects attacked rows whose gold source lacks visible semantic
evidence. The final SFT schema remains identical to V20 MARBLE.

Run after activating the existing CUDA environment:

```bash
bash SFT/auditor_agent_sft_v20_autogen_package/server_scripts/run_v20_autogen_core_validation.sh
```

The resumable runner evaluates TF-IDF, Qwen3-8B SFT, ModernBERT-8192,
G-Safeguard, and TAM encoder on validation. The lexical-shortcut ablation is
intentionally omitted, and the sealed test split is never consumed.
