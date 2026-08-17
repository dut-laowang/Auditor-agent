from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--track-index", required=True, type=Path)
    parser.add_argument("--track", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--id-field", default="run_id")
    args = parser.parse_args()
    wanted_order = [row["run_id"] for row in read(args.track_index) if row["track"] == args.track]
    rows = read(args.input)
    by_id = {}
    for row in rows:
        run_id = row.get(args.id_field)
        if run_id is None and isinstance(row.get("metadata"), dict):
            run_id = row["metadata"].get("run_id")
        if run_id in by_id:
            raise RuntimeError(f"Duplicate run_id in input: {run_id}")
        by_id[run_id] = row
    missing = [run_id for run_id in wanted_order if run_id not in by_id]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} track rows, first={missing[:5]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for run_id in wanted_order:
            handle.write(json.dumps(by_id[run_id], ensure_ascii=False) + "\n")
    print(json.dumps({"track": args.track, "rows": len(wanted_order), "output": str(args.output)}))


if __name__ == "__main__":
    main()
