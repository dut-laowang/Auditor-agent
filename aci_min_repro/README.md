# ACIArena MetaGPT Code Disclosure 150-Run Reproduction

This folder contains thin wrappers around the official ACIArena repository at:

```text
../mas_safety_doc/_src_aciarena
```

The wrappers do **not** reimplement attacks or prompts. They import ACIArena's
own MAS, attack classes, task classes, and logger, then save one JSON file per
raw trajectory.

## Target Setting

This reproduces the official MetaGPT script setting:

```text
MetaGPT
+ code tasks
+ disclosure suite
+ qa_engineer as malicious agent
+ all disclosure/code attack classes
```

In the current public repository:

```text
30 code tasks x 5 disclosure/code attacks = 150 raw trajectories
```

The 5 attack classes are loaded from ACIArena's registry, not hard-coded in this
wrapper.

## Files To Edit

Before running on the server, create local config files from examples:

```bash
cp model_config.example.yaml model_config.yaml
cp judge_config.example.yaml judge_config.yaml
```

Then edit both files:

```text
plan_e/aci_min_repro/model_config.yaml
plan_e/aci_min_repro/judge_config.yaml
```

Example OpenAI-compatible config:

```yaml
provider: openai
api_key: YOUR_API_KEY
base_url: https://api.openai.com/v1
model_name: gpt-4o-mini
temperature: 0.0
max_tokens: 1024
```

For a local vLLM server, keep `provider: openai` and set `base_url` and
`model_name` to your local OpenAI-compatible endpoint.

## Environment

From the ACIArena root:

```bash
cd plan_e/mas_safety_doc/_src_aciarena
conda create -n aciarena python=3.10 -y
conda activate aciarena
pip install -e .
```

Then return to this repo root:

```bash
cd ../../..
```

## Check The Plan Without Calling LLM

```bash
python plan_e/aci_min_repro/run_metagpt_code_disclosure_150.py --dry-run
```

Expected output should include:

```text
MAS: metagpt
Suite/domain: disclosure/code
Malicious agents: ['qa_engineer']
Tasks: 30
Expected raw trajectories: 150
```

## Run The 150 Trajectories

Linux/macOS:

```bash
bash plan_e/aci_min_repro/run_metagpt_code_disclosure_150.sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File plan_e/aci_min_repro/run_metagpt_code_disclosure_150.ps1
```

Direct Python equivalent:

```bash
python plan_e/aci_min_repro/run_metagpt_code_disclosure_150.py
```

## Output

Raw logs are written to:

```text
plan_e/aci_min_repro/outputs/metagpt_code_disclosure_150/raw_runs
```

Each JSON contains:

```text
meta_data
turns: sender / receiver / message / tool
result: attack_class / attack_surface / attack_payload / attack_success / utility_success
```

The run also writes:

```text
manifest.json
```

with `expected_runs`, `actual_runs`, attack class names, and all output paths.

Check the run:

```bash
python plan_e/aci_min_repro/check_outputs.py \
  --input-dir plan_e/aci_min_repro/outputs/metagpt_code_disclosure_150/raw_runs \
  --expected 150
```

## Convert To Prefix Records

After raw logs are collected:

```bash
python plan_e/aci_min_repro/to_audit_prefix.py \
  --input-dir plan_e/aci_min_repro/outputs/metagpt_code_disclosure_150/raw_runs \
  --output-jsonl plan_e/aci_min_repro/outputs/metagpt_code_disclosure_150/audit_prefix_samples.jsonl
```

## Important Caveats

- `150` means raw run attempts: 30 tasks x 5 attacks. It does **not** mean all
  150 attacks succeed.
- This wrapper stores per-trajectory logs. ACIArena's original benchmark mainly
  reports aggregate ASR.
- The public ACIArena repo currently contains `maspi_code.json`; this wrapper
  falls back to that file if `aciarena_code.json` is absent.
- Higher-level labels such as `instruction_laundering` and
  `privilege_boundary_crossed` are not produced by ACIArena directly. They need
  later rule/LLM/human annotation.
