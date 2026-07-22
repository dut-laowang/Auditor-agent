import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def key_for(row):
    meta = row.get("metadata", {})
    return (
        meta.get("verdict", "unknown"),
        meta.get("scenario", "unknown"),
        meta.get("surface", "unknown"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True, type=Path)
    parser.add_argument("--output-file", required=True, type=Path)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    groups = defaultdict(list)
    for row in rows:
        groups[key_for(row)].append(row)

    rng = random.Random(args.seed)
    chosen = []
    keys = list(groups)
    for key in keys:
        rng.shuffle(groups[key])
        if groups[key]:
            chosen.append(groups[key].pop())

    remaining = [row for group in groups.values() for row in group]
    rng.shuffle(remaining)
    chosen.extend(remaining[: max(0, args.n - len(chosen))])
    chosen = chosen[: args.n]
    rng.shuffle(chosen)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as handle:
        for row in chosen:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({"input": len(rows), "output": len(chosen), "seed": args.seed}, indent=2))


if __name__ == "__main__":
    main()
