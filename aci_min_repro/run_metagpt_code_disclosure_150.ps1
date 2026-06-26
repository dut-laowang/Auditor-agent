$ErrorActionPreference = "Stop"

python "$PSScriptRoot\run_metagpt_code_disclosure_150.py" `
  --model-config "$PSScriptRoot\model_config.yaml" `
  --judge-config "$PSScriptRoot\judge_config.yaml" `
  --output-dir "$PSScriptRoot\outputs\metagpt_code_disclosure_150\raw_runs" `
  --task-limit 30
