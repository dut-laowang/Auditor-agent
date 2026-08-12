# V20 MARBLE Core Experiment

V20 replaces only the raw MARBLE source with the 10,665-run randomized,
de-identified bundle. It reuses the V19 observable-input schema, label mapping,
G/N/E/T projection, seed-42 grouped split, model revisions, hyperparameters, and
validation metrics.

Required archive:

`$BASE/data-812/marble_random_10665_trajectories_configs_labels.tar.zst`

Run from the repository after activating the existing CUDA environment:

```bash
bash SFT/auditor_agent_sft_v20_marble_package/server_scripts/run_v20_core_validation.sh
```

The runner executes validation only: TF-IDF, Qwen3-8B SFT, Qwen3-8B lexical-
shortcut masking, ModernBERT-8192, G-Safeguard supervised adaptation, and TAM
encoder supervised adaptation. It never reads the sealed test split.

