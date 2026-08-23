# V22 closed-LLM evaluation

The repository includes the exact 2,539 label-blind requests used by the V22
DeepSeek baseline. It contains observable MAS traces and prompts, but no gold
verdict, scope, or component labels. The runner automatically extracts the
bundled asset and verifies its row count and SHA-256 before making any API call.

## One batch command: GPT-4.1 + Claude Sonnet 4.6

```bash
git clone https://github.com/dut-laowang/Auditor-agent.git
cd Auditor-agent
bash SFT/auditor_agent_v22_legacy_experiments/server_scripts/run_v22_closed_llms_batch_once.sh
```

The command securely prompts for exactly two values: the OpenAI API Key and the
Anthropic API Key. It then runs both 2,539-row evaluations concurrently, shows
live progress, supports resume by rerunning the same command, and finally emits
`V22_CLOSED_LLMS_RAW_RESULTS.tar.gz` for return. Keys are not written to outputs.

## OpenAI GPT-4.1

```bash
git clone https://github.com/dut-laowang/Auditor-agent.git
cd Auditor-agent
export OPENAI_API_KEY='...'
export PROVIDER=openai
export MODEL=gpt-4.1-2025-04-14
export API_WORKERS=16
bash SFT/auditor_agent_v22_legacy_experiments/server_scripts/run_v22_closed_llm_colleague_once.sh
```

## Anthropic Claude Sonnet 4.6

```bash
git clone https://github.com/dut-laowang/Auditor-agent.git
cd Auditor-agent
export ANTHROPIC_API_KEY='...'
export PROVIDER=anthropic
export MODEL=claude-sonnet-4-6
export API_WORKERS=8
bash SFT/auditor_agent_v22_legacy_experiments/server_scripts/run_v22_closed_llm_colleague_once.sh
```

Install the selected provider SDK plus `tqdm` and `scikit-learn` in the active
Python environment before running. Both commands are resumable. Progress reports completed rows, failures, input and
output tokens, throughput, and ETA. Return only the generated
`V22_<provider>_<model>_RAW_RESULTS.tar.gz`; private scoring is performed by the
benchmark owner against the matching gold SHA-256.
