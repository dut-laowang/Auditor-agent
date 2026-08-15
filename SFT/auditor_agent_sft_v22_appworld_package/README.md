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
