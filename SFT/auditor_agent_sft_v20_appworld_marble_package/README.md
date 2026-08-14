# V20 AppWorld x MARBLE core validation

This package converts `appworld_marble_random_3000_complete_20260814.tar.zst`
to the same Graph-grounded-Candidate-SFT/v13 schema and core validation protocol
used by V20 MARBLE and AutoGen V2.

Observable-field policy:

- MARBLE `agent_act.clean_content` for actual agent task input;
- `agent_output.result` for model output;
- AppWorld `delivered_content` for message and tool-result events;
- no attack instrumentation, label, marker, oracle, or attacked-content field is
  copied into model input;
- split unit is `(scenario, sample_id)` and sealed test is never evaluated by the
  one-click validation script.

Server command:

```bash
cd /gs/bs/tgh-26IAW/hongbo/project_4_coauthor/Auditor-agent && \
git pull --ff-only origin main && \
bash SFT/auditor_agent_sft_v20_appworld_marble_package/server_scripts/run_v20_appworld_marble_core_validation.sh
```
# Joint discriminative-generative SFT comparison

The optional joint experiment preserves the frozen AppWorld x MARBLE V20
dataset, prompt, LoRA settings, decoder, evaluator, and full JSON output.  It
adds only an assistant-boundary three-way verdict loss and a higher token loss
weight for the existing `localization` field.  Gold targets are read only from
the assistant message after the model-visible prompt has been constructed.
During validation, the LM head first generates the complete JSON report. The
auxiliary verdict head then replaces only `decision.verdict` and its derived
`binary_label`; all localization and evidence fields remain the model's
original generation. The evaluator records head accuracy, pre-merge agreement,
and strict full-schema validity in addition to the same end-to-end metrics.

```bash
cd /gs/bs/tgh-26IAW/hongbo/project_4_coauthor/Auditor-agent && \
git pull --ff-only origin main && \
bash SFT/auditor_agent_sft_v20_appworld_marble_package/server_scripts/run_v20_appworld_marble_joint_sft.sh
```

This reuses the prior filtered dataset and baseline metrics, writes to new
model/result directories, supports checkpoint resume, and does not access the
sealed test split.
