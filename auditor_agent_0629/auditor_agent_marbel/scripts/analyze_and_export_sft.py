from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def rate(success: int, total: int) -> str:
    return f"{success}/{total} = {success / total:.2%}" if total else "0/0 = 0.00%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize judged ACI-on-MARBLE outputs and export success indexes.")
    parser.add_argument("--judge-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(args.judge_dir / "judged_runs.jsonl")
    completed = [row for row in rows if row.get("status") == "completed"]
    success = [row for row in completed if row.get("success")]

    by_objective_total: Counter[str] = Counter(row["objective"] for row in completed)
    by_objective_success: Counter[str] = Counter(row["objective"] for row in success)
    by_surface_total: Counter[str] = Counter(row["surface"] for row in completed)
    by_surface_success: Counter[str] = Counter(row["surface"] for row in success)
    by_scenario_total: Counter[str] = Counter(row["scenario"] for row in completed)
    by_scenario_success: Counter[str] = Counter(row["scenario"] for row in success)
    by_topology_total: Counter[str] = Counter(row["topology"] for row in completed)
    by_topology_success: Counter[str] = Counter(row["topology"] for row in success)
    by_scenario_objective_total: Counter[str] = Counter(f"{row['scenario']}:{row['objective']}" for row in completed)
    by_scenario_objective_success: Counter[str] = Counter(f"{row['scenario']}:{row['objective']}" for row in success)
    by_surface_objective_total: Counter[str] = Counter(f"{row['surface']}:{row['objective']}" for row in completed)
    by_surface_objective_success: Counter[str] = Counter(f"{row['surface']}:{row['objective']}" for row in success)

    summary: dict[str, Any] = {
        "completed_attacked": len(completed),
        "success": len(success),
        "asr": rate(len(success), len(completed)),
        "by_objective": {key: rate(by_objective_success[key], total) for key, total in sorted(by_objective_total.items())},
        "by_surface": {key: rate(by_surface_success[key], total) for key, total in sorted(by_surface_total.items())},
        "by_scenario": {key: rate(by_scenario_success[key], total) for key, total in sorted(by_scenario_total.items())},
        "by_topology": {key: rate(by_topology_success[key], total) for key, total in sorted(by_topology_total.items())},
        "by_scenario_objective": {
            key: rate(by_scenario_objective_success[key], total)
            for key, total in sorted(by_scenario_objective_total.items())
        },
        "by_surface_objective": {
            key: rate(by_surface_objective_success[key], total)
            for key, total in sorted(by_surface_objective_total.items())
        },
    }
    (args.output_dir / "asr_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with (args.output_dir / "success_runs.jsonl").open("w", encoding="utf-8") as handle:
        for row in success:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    reps: dict[str, dict[str, Any]] = {}
    for row in success:
        reps.setdefault(row["objective"], row)
    (args.output_dir / "representative_successes.json").write_text(
        json.dumps(reps, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
