from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from auditor_agent_marbel.core.jsonl import read_jsonl


def normalize_scenario(raw: str) -> str:
    value = raw.strip().lower()
    aliases = {"db": "database"}
    return aliases.get(value, value)


def load_samples(marble_root: Path, dataset: str, sample_ids: Iterable[int]) -> List[Dict[str, Any]]:
    path = marble_root / dataset
    if not path.exists():
        raise FileNotFoundError(f"MARBLE dataset not found: {path}")

    wanted = {int(sample_id) for sample_id in sample_ids}
    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(read_jsonl(path), start=1):
        task_id = row.get("task_id", idx)
        try:
            task_id_int = int(task_id)
        except (TypeError, ValueError):
            task_id_int = idx
        if task_id_int in wanted:
            row = dict(row)
            row["task_id"] = task_id_int
            rows.append(row)

    found = {int(row["task_id"]) for row in rows}
    missing = wanted - found
    if missing:
        raise ValueError(f"Missing sample ids {sorted(missing)} in {path}")
    return sorted(rows, key=lambda row: int(row["task_id"]))

