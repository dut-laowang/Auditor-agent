import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-file", required=True, type=Path)
    parser.add_argument("--ids-file", required=True, type=Path)
    parser.add_argument("--output-file", required=True, type=Path)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    ids = [line.strip() for line in args.ids_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    wanted = set(ids)
    rows = {}
    with args.all_file.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            run_id = row.get("metadata", {}).get("run_id")
            if run_id in wanted:
                rows[run_id] = row

    missing = [run_id for run_id in ids if run_id not in rows]
    if missing and not args.allow_missing:
        raise SystemExit(f"Missing {len(missing)} run_ids, first missing: {missing[:5]}")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output_file.open("w", encoding="utf-8") as out:
        for run_id in ids:
            if run_id not in rows:
                continue
            out.write(json.dumps(rows[run_id], ensure_ascii=False) + "\n")
            written += 1

    print(json.dumps({
        "requested": len(ids),
        "written": written,
        "missing": missing,
        "output_file": str(args.output_file),
    }, indent=2))


if __name__ == "__main__":
    main()
