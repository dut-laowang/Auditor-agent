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
