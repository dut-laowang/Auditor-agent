# Server Commands

Assumed base path:

```bash
/gs/bs/tgh-26IAW/hongbo/project_4_coauthor
```

## 1. Clone

```bash
cd /gs/bs/tgh-26IAW/hongbo/project_4_coauthor
git clone https://github.com/dut-laowang/Auditor-agent.git
git clone https://github.com/ulab-uiuc/MARBLE.git
```

## 2. Environment

```bash
conda create -n auditor-marble python=3.10 -y
conda activate auditor-marble
cd /gs/bs/tgh-26IAW/hongbo/project_4_coauthor/Auditor-agent/auditor_agent_marbel
pip install -e .
cd /gs/bs/tgh-26IAW/hongbo/project_4_coauthor/MARBLE
pip install -e .
```

## 3. API Config

For an OpenAI-compatible yun endpoint:

```bash
export OPENAI_API_KEY="sk-xxx"
export OPENAI_API_BASE="https://yunai.chat/v1"
export OPENAI_BASE_URL="https://yunai.chat/v1"
```

Model and base URL defaults are also recorded in:

```bash
/gs/bs/tgh-26IAW/hongbo/project_4_coauthor/Auditor-agent/auditor_agent_marbel/configs/pilot_run.yaml
```

## 4. Dry Run

```bash
cd /gs/bs/tgh-26IAW/hongbo/project_4_coauthor/Auditor-agent/auditor_agent_marbel
python -m auditor_agent_marbel.runner.run_attack \
  --marble-root /gs/bs/tgh-26IAW/hongbo/project_4_coauthor/MARBLE \
  --run-config configs/pilot_run.yaml \
  --attack-spec data/attack_specs/pilot_attack_specs.jsonl \
  --output-dir outputs/pilot_dry_run \
  --dry-run
```

## 5. Clean Baselines

```bash
python -m auditor_agent_marbel.runner.run_attack \
  --marble-root /gs/bs/tgh-26IAW/hongbo/project_4_coauthor/MARBLE \
  --run-config configs/pilot_run.yaml \
  --output-dir outputs/pilot_clean \
  --clean-only
```

## 6. Attacked Runs

```bash
python -m auditor_agent_marbel.runner.run_attack \
  --marble-root /gs/bs/tgh-26IAW/hongbo/project_4_coauthor/MARBLE \
  --run-config configs/pilot_run.yaml \
  --attack-spec data/attack_specs/pilot_attack_specs.jsonl \
  --output-dir outputs/pilot_attacked
```
