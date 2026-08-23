# V22 closed-LLM colleague evaluation

The separately supplied `V22_CLOSED_LLM_COLLEAGUE_INPUTS.tar.gz` contains 2,539
label-blind requests. It contains no gold verdict, scope, or component labels.
Do not publish this input archive in a public repository.

## OpenAI GPT-4.1

```bash
tar -xzf V22_CLOSED_LLM_COLLEAGUE_INPUTS.tar.gz
export OPENAI_API_KEY='...'
export V22_REQUESTS="$PWD/V22_CLOSED_LLM_REQUESTS.jsonl"
export PROVIDER=openai
export MODEL=gpt-4.1-2025-04-14
export API_WORKERS=16
bash SFT/auditor_agent_v22_legacy_experiments/server_scripts/run_v22_closed_llm_colleague_once.sh
```

## Anthropic Claude Sonnet 4.6

```bash
tar -xzf V22_CLOSED_LLM_COLLEAGUE_INPUTS.tar.gz
export ANTHROPIC_API_KEY='...'
export V22_REQUESTS="$PWD/V22_CLOSED_LLM_REQUESTS.jsonl"
export PROVIDER=anthropic
export MODEL=claude-sonnet-4-6
export API_WORKERS=8
bash SFT/auditor_agent_v22_legacy_experiments/server_scripts/run_v22_closed_llm_colleague_once.sh
```

Both commands are resumable. Progress reports completed rows, failures, input and
output tokens, throughput, and ETA. Return only the generated
`V22_<provider>_<model>_RAW_RESULTS.tar.gz`; private scoring is performed by the
benchmark owner against the matching gold SHA-256.
