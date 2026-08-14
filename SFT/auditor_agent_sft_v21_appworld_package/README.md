# V21 MARBLE × AppWorld

V21 is a minimal, leakage-controlled extension of the frozen V20 MARBLE × AppWorld experiment.

- The validation set is the exact frozen V20 set: 406 rows and identical run IDs.
- Frozen V20 Qwen Audit SFT representations feed separately trained verdict, scope, and component heads.
- No classification or localization LoRA updates the Qwen backbone.
- Conditional Audit LoRA starts from the V20 adapter. It trains on train-only gold controls and evaluates on predicted validation controls.
- The external JSON schema remains `decision + attack + localization + audit_trace`.
- No sealed test data is read.

Run on the server:

```bash
cd /gs/bs/tgh-26IAW/hongbo/project_4_coauthor/Auditor-agent && \
git pull --ff-only origin main && \
bash SFT/auditor_agent_sft_v21_appworld_package/server_scripts/run_v21_appworld_validation.sh
```

The script is resumable: feature extraction is checkpointed every 100 rows,
head training after every epoch, Audit LoRA every 100 optimizer steps, and
evaluation after every generated batch. Results are written to
`/gs/bs/tgh-26IAW/hongbo/project_4_coauthor/v21_appworld_marble_validation` and packed as
`v21_appworld_marble_validation.tar.gz`.

## ModernBERT → Qwen audit cascade

This reuses the frozen V20 ModernBERT predictions and the trained V21 conditional
Audit LoRA. Validation controls are predictions—not gold labels—and the final gate
requires all classification/localization metrics to exactly equal ModernBERT while
retaining a complete Qwen-generated audit report.

```bash
cd /gs/bs/tgh-26IAW/hongbo/project_4_coauthor/Auditor-agent && \
git pull --ff-only origin main && \
bash SFT/auditor_agent_sft_v21_appworld_package/server_scripts/run_modernbert_qwen_audit_cascade.sh
```
