import argparse
import json
import random
from pathlib import Path


def load_by_id(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            run_id = row.get("metadata", {}).get("run_id")
            if run_id:
                rows[run_id] = row
    return rows


def bucket_key(row: dict) -> tuple[str, str, str, str]:
    meta = row.get("metadata", {})
    return (
        str(meta.get("scenario", "unknown")),
        str(meta.get("surface", "unknown")),
        str(meta.get("objective", "unknown")),
        str(meta.get("topology", "unknown")),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v8-all", required=True, type=Path)
    parser.add_argument("--v9-all", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260709)
    args = parser.parse_args()

    v8 = load_by_id(args.v8_all)
    v9 = load_by_id(args.v9_all)
    common_ids = sorted(set(v8) & set(v9))
    if len(common_ids) < args.count:
        raise SystemExit(f"Only {len(common_ids)} common run_ids, cannot sample {args.count}.")

    by_bucket: dict[tuple[str, str, str, str], list[str]] = {}
    for run_id in common_ids:
        by_bucket.setdefault(bucket_key(v9[run_id]), []).append(run_id)

    rng = random.Random(args.seed)
    for ids in by_bucket.values():
        rng.shuffle(ids)

    chosen: list[str] = []
    seen = set()
    buckets = sorted(by_bucket.items(), key=lambda item: (len(item[1]), item[0]))
    while len(chosen) < args.count:
        progressed = False
        for _, ids in buckets:
            while ids and ids[0] in seen:
                ids.pop(0)
            if ids and len(chosen) < args.count:
                run_id = ids.pop(0)
                chosen.append(run_id)
                seen.add(run_id)
                progressed = True
        if not progressed:
            break

    chosen = chosen[: args.count]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "common_run_ids.txt").write_text("\n".join(chosen) + "\n", encoding="utf-8")

    with (args.output_dir / "v8_common.jsonl").open("w", encoding="utf-8") as out:
        for run_id in chosen:
            out.write(json.dumps(v8[run_id], ensure_ascii=False) + "\n")
    with (args.output_dir / "v9_common.jsonl").open("w", encoding="utf-8") as out:
        for run_id in chosen:
            out.write(json.dumps(v9[run_id], ensure_ascii=False) + "\n")

    summary = {
        "count": len(chosen),
        "available_common_ids": len(common_ids),
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
