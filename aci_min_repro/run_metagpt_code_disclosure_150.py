#!/usr/bin/env python3
"""One-command ACIArena MetaGPT code disclosure run.

Target setting mirrors ACIArena's official `scripts/metagpt.sh` default:

    MetaGPT + code tasks + disclosure suite + qa_engineer as malicious agent

With the current public repository data, this is:

    30 code tasks x 5 disclosure/code attack classes = 150 raw trajectories

This file does not reimplement attacks. It delegates to run_single_attack.py,
which imports ACIArena's own attacks, MAS implementation, tasks, and logger.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    plan_e_dir = script_dir.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--aciarena-root", type=Path, default=plan_e_dir / "mas_safety_doc" / "_src_aciarena")
    parser.add_argument("--model-config", type=Path, default=script_dir / "model_config.yaml")
    parser.add_argument("--judge-config", type=Path, default=script_dir / "judge_config.yaml")
    parser.add_argument("--output-dir", type=Path, default=script_dir / "outputs" / "metagpt_code_disclosure_150" / "raw_runs")
    parser.add_argument("--task-limit", type=int, default=30)
    parser.add_argument("--defense", default="none")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(Path(__file__).with_name("run_single_attack.py")),
        "--aciarena-root",
        str(args.aciarena_root),
        "--mas",
        "metagpt",
        "--suite",
        "disclosure",
        "--task-domain",
        "code",
        "--all-attacks",
        "--malicious-agents",
        "qa_engineer",
        "--task-limit",
        str(args.task_limit),
        "--defense",
        args.defense,
        "--model-config",
        str(args.model_config),
        "--judge-config",
        str(args.judge_config),
        "--output-dir",
        str(args.output_dir),
    ]
    if args.dry_run:
        cmd.append("--dry-run")

    print("Running:")
    print(" ".join(cmd))
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
