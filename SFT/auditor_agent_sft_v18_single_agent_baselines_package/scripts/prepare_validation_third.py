from __future__ import annotations

import argparse
import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any


SEED = 20260730
FRACTION = 1 / 3
TEST_N = 200


def read(path: Path, member: str | None = None) -> list[dict[str, Any]]:
    if member:
        with zipfile.ZipFile(path) as archive:
            text = archive.read(member).decode("utf-8-sig")
    else:
        text = path.read_text(encoding="utf-8-sig")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def stratum(row: dict[str, Any]) -> tuple[str, ...]:
    meta = row["metadata"]
    return tuple(
        str(meta.get(key, "unknown"))
        for key in ("verdict", "scenario", "attack_mode", "surface")
    )


def proportional_sample(
    rows: list[dict[str, Any]], target: int, seed: int
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[stratum(row)].append(row)
    rng = random.Random(seed)
    for group in groups.values():
        group.sort(key=lambda row: row["metadata"]["run_id"])
        rng.shuffle(group)

    exact = {key: len(group) * target / len(rows) for key, group in groups.items()}
    quota = {key: int(value) for key, value in exact.items()}
    remaining = target - sum(quota.values())
    order = sorted(groups, key=lambda key: (-(exact[key] - quota[key]), key))
    for key in order[:remaining]:
        quota[key] += 1

    selected_ids = {
        row["metadata"]["run_id"]
        for key, group in groups.items()
        for row in group[: quota[key]]
    }
    # Preserve original order so Flat and Graph are byte-aligned by row index.
    return [row for row in rows if row["metadata"]["run_id"] in selected_ids]


def write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-data-dir", required=True, type=Path)
    parser.add_argument("--flat-data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    graph_train = read(args.graph_data_dir / "train.jsonl.zip", "train.jsonl")
    flat_train = read(args.flat_data_dir / "train.jsonl.zip", "train.jsonl")
    graph_test = read(args.graph_data_dir / "test.jsonl")
    flat_test = read(args.flat_data_dir / "test.jsonl")

    graph_train_ids = [row["metadata"]["run_id"] for row in graph_train]
    flat_train_ids = [row["metadata"]["run_id"] for row in flat_train]
    graph_test_ids = [row["metadata"]["run_id"] for row in graph_test]
    flat_test_ids = [row["metadata"]["run_id"] for row in flat_test]
    if graph_train_ids != flat_train_ids or graph_test_ids != flat_test_ids:
        raise ValueError("Flat/Graph source row alignment failed")

    target = round(len(graph_train) * FRACTION)
    graph_train_third = proportional_sample(graph_train, target, SEED)
    selected_train_ids = {
        row["metadata"]["run_id"] for row in graph_train_third
    }
    flat_train_third = [
        row for row in flat_train if row["metadata"]["run_id"] in selected_train_ids
    ]

    graph_test_200 = proportional_sample(graph_test, TEST_N, SEED + 1)
    selected_test_ids = {row["metadata"]["run_id"] for row in graph_test_200}
    flat_test_200 = [
        row for row in flat_test if row["metadata"]["run_id"] in selected_test_ids
    ]

    write(args.output_dir / "graph/train.jsonl", graph_train_third)
    write(args.output_dir / "graph/test.jsonl", graph_test_200)
    write(args.output_dir / "flat/train.jsonl", flat_train_third)
    write(args.output_dir / "flat/test.jsonl", flat_test_200)

    manifest = {
        "protocol": "validation_third_frozen_before_training",
        "seed": SEED,
        "train_source_rows": len(graph_train),
        "train_target_fraction": FRACTION,
        "train_rows": len(graph_train_third),
        "test_source_rows": len(graph_test),
        "test_rows": len(graph_test_200),
        "stratification": ["verdict", "scenario", "attack_mode", "surface"],
        "train_run_ids": [
            row["metadata"]["run_id"] for row in graph_train_third
        ],
        "test_run_ids": [row["metadata"]["run_id"] for row in graph_test_200],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in manifest.items()
                if key not in {"train_run_ids", "test_run_ids"}
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
