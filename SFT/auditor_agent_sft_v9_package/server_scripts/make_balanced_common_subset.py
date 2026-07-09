import argparse
import json
import random
from collections import Counter, defaultdict, deque
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


def meta_value(row: dict, key: str) -> str:
    return str(row.get("metadata", {}).get(key, "unknown"))


def secondary_key(row: dict) -> tuple[str, str]:
    meta = row.get("metadata", {})
    return (
        str(meta.get("objective", "unknown")),
        str(meta.get("topology", "unknown")),
    )


def interleave_by_secondary(rows: list[str], reference: dict[str, dict], rng: random.Random) -> deque[str]:
    by_scenario: dict[str, dict[tuple[str, str], list[str]]] = defaultdict(lambda: defaultdict(list))
    for run_id in rows:
        scenario = meta_value(reference[run_id], "scenario")
        by_scenario[scenario][secondary_key(reference[run_id])].append(run_id)

    scenario_queues: dict[str, deque[str]] = {}
    for scenario, buckets in by_scenario.items():
        for ids in buckets.values():
            rng.shuffle(ids)
        scenario_ordered = []
        keys = sorted(buckets)
        while True:
            progressed = False
            for key in keys:
                if buckets[key]:
                    scenario_ordered.append(buckets[key].pop())
                    progressed = True
            if not progressed:
                break
        scenario_queues[scenario] = deque(scenario_ordered)

    ordered = []
    scenarios = sorted(scenario_queues)
    while True:
        progressed = False
        for scenario in scenarios:
            if scenario_queues[scenario]:
                ordered.append(scenario_queues[scenario].popleft())
                progressed = True
        if not progressed:
            break
    return deque(ordered)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v8-all", required=True, type=Path)
    parser.add_argument("--v9-all", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--balance-key", default="surface", help="Metadata key used for primary balancing.")
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--trials", type=int, default=20000)
    args = parser.parse_args()

    v8 = load_by_id(args.v8_all)
    v9 = load_by_id(args.v9_all)
    common_ids = sorted(set(v8) & set(v9))
    if len(common_ids) < args.count:
        raise SystemExit(f"Only {len(common_ids)} common run_ids, cannot sample {args.count}.")

    rng = random.Random(args.seed)
    dimensions = ["surface", "objective", "scenario", "topology"]
    values = {key: sorted({meta_value(v9[run_id], key) for run_id in common_ids}) for key in dimensions}

    def score(sample: list[str]) -> float:
        total = 0.0
        for key in dimensions:
            counts = Counter(meta_value(v9[run_id], key) for run_id in sample)
            ideal = len(sample) / max(len(values[key]), 1)
            for value in values[key]:
                total += (counts.get(value, 0) - ideal) ** 2 / max(ideal, 1.0)
        return total

    best: list[str] | None = None
    best_score = float("inf")
    for _ in range(args.trials):
        sample = rng.sample(common_ids, args.count)
        current = score(sample)
        if current < best_score:
            best = sample
            best_score = current

    if best is None:
        raise SystemExit("Failed to sample a balanced subset.")
    chosen = sorted(best)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "balanced_common_run_ids.txt").write_text("\n".join(chosen) + "\n", encoding="utf-8")

    with (args.output_dir / "v8_balanced_common.jsonl").open("w", encoding="utf-8") as out:
        for run_id in chosen:
            out.write(json.dumps(v8[run_id], ensure_ascii=False) + "\n")
    with (args.output_dir / "v9_balanced_common.jsonl").open("w", encoding="utf-8") as out:
        for run_id in chosen:
            out.write(json.dumps(v9[run_id], ensure_ascii=False) + "\n")

    def distribution(key: str) -> dict[str, int]:
        return dict(sorted(Counter(meta_value(v9[run_id], key) for run_id in chosen).items()))

    summary = {
        "count": len(chosen),
        "available_common_ids": len(common_ids),
        "balance_key": args.balance_key,
        "trials": args.trials,
        "balance_score": best_score,
        "output_dir": str(args.output_dir),
        "by_surface": distribution("surface"),
        "by_objective": distribution("objective"),
        "by_scenario": distribution("scenario"),
        "by_topology": distribution("topology"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
