# V22 MARBLE × AppWorld explainable audit pipeline

V22 uses one frozen 3,122/406 split and identical run-ID order for both the
ModernBERT Inspector and conditional Qwen Audit SFT.

- ModernBERT sees the original label-free system/user trajectory.
- Qwen training sees train-only gold verdict/localization as `audit_control`.
- Qwen validation sees only the 406 ModernBERT predictions.
- Explanations cite visible `run_evidence`; config/private control is never read.
- The top-level `decision + attack + localization + audit_trace` schema is unchanged.
- The sealed test is never accessed.

```bash
cd /gs/bs/tgh-26IAW/hongbo/project_4_coauthor/Auditor-agent && \
git pull --ff-only origin main && \
bash SFT/auditor_agent_sft_v22_appworld_package/server_scripts/run_v22_full_pipeline.sh
```

The final artifact is `/gs/bs/tgh-26IAW/hongbo/project_4_coauthor/v22_appworld_marble_validation.tar.gz`.

## Unified pipeline for a new raw AppWorld x MARBLE archive

The unified workflow deliberately keeps the discriminative and generative data
views separate:

- ModernBERT trains and predicts on the unexpanded `base_dataset` produced
  locally from observable MAS logs.
- The final V22 validation view is created on the server only after ModernBERT
  predictions exist; validation gold is never inserted into model input.
- Qwen3-32B enriches only the V22 training targets.
- Qwen3-8B Audit SFT trains only on the expanded `audit_sft` view.

Local preparation (Windows or Linux, one command):

```powershell
cd D:\FIRST_COPILOT\plan_e\auditor_agent
python SFT/auditor_agent_sft_v22_appworld_package/scripts/local_prepare_v22_source.py `
  --input D:\FIRST_COPILOT\plan_e\data-812\appworld_marble_random_3000_complete_20260814.tar.zst `
  --output-dir D:\FIRST_COPILOT\plan_e\v22_source_bundle
```

This creates both `v22_source_bundle/` and `v22_source_bundle.zip`. Upload the
ZIP to the server, then run the complete resumable server pipeline:

```bash
cd /gs/bs/tgh-26IAW/hongbo/project_4_coauthor/Auditor-agent && \
git pull --ff-only origin main && \
V22_SOURCE_BUNDLE=/gs/bs/tgh-26IAW/hongbo/project_4_coauthor/v22_source_bundle.zip \
V22_UNIFIED_RUN=/gs/bs/tgh-26IAW/hongbo/project_4_coauthor/v22_unified_run \
TEACHER_BATCH=4 QWEN_TRAIN_BATCH=2 QWEN_GRAD_ACCUM=8 QWEN_EVAL_BATCH=4 \
bash SFT/auditor_agent_sft_v22_appworld_package/server_scripts/run_v22_unified_server.sh
```

All restored data, ModernBERT artifacts, teacher outputs, expanded SFT data,
Qwen checkpoints, predictions, logs, and quality gates are stored beneath the
single `V22_UNIFIED_RUN` directory. Repeating the same command resumes all
completed stages. The current local adapter profile supports AppWorld x MARBLE;
MARBLE x MultiAgentBench and AutoGen x MultiAgentBench require their respective
observable adapters rather than treating unlike raw archives as one format.

## Qwen3-32B teacher expansion

This resumably enriches only the 3,122 training targets with three fields:
`causal_explanation`, `recommended_action`, and `confidence`. All previous
labels, locations, evidence references, split IDs, and visible inputs are frozen.
Validation gold is never sent to the teacher.

```bash
cd /gs/bs/tgh-26IAW/hongbo/project_4_coauthor/Auditor-agent && \
git pull --ff-only origin main && \
bash SFT/auditor_agent_sft_v22_appworld_package/server_scripts/run_qwen32b_v22_teacher_expansion.sh
```
