#!/usr/bin/env python3
"""Check ACIArena raw trajectory outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("plan_e/aci_min_repro/outputs/metagpt_code_disclosure_150/raw_runs"))
    parser.add_argument("--expected", type=int, default=150)
    args = parser.parse_args()

    files = sorted(p for p in args.input_dir.glob("*.json") if p.name != "manifest.json")
    ok_turns = 0
    attack_success_true = 0
    attack_success_false = 0
    attack_success_none = 0
    missing_result = []

    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("turns"):
            ok_turns += 1
        result = data.get("result")
        if not isinstance(result, dict):
            missing_result.append(path.name)
            continue
        value = result.get("attack_success")
        if value is True or value == 1.0:
            attack_success_true += 1
        elif value is False or value == 0.0:
            attack_success_false += 1
        else:
            attack_success_none += 1

    print(f"raw_json_files: {len(files)}")
    print(f"expected: {args.expected}")
    print(f"files_with_turns: {ok_turns}")
    print(f"attack_success_true: {attack_success_true}")
    print(f"attack_success_false: {attack_success_false}")
    print(f"attack_success_unknown_or_error: {attack_success_none}")
    if missing_result:
        print(f"missing_result_files: {len(missing_result)}")
        for name in missing_result[:20]:
            print(f"  - {name}")

    if len(files) != args.expected:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
