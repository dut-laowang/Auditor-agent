from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import Counter
from pathlib import Path


FORBIDDEN = re.compile(
    r"ACI_[A-Z0-9_]+|\baci_[a-z0-9_]+\b|\bEND_NEGOTIATION\b|"
    r"success_markers?|attack_metadata|attack_id|payload_id|marker_check|"
    r"\[Injected[^\]]*\]|offline verifier|attack-success index|"
    r"labeled as attack-success",
    re.I,
)


def read(path: Path, member: str | None = None) -> list[dict]:
    if member:
        with zipfile.ZipFile(path) as archive:
            text = archive.read(member).decode("utf-8-sig")
    else:
        text = path.read_text(encoding="utf-8-sig")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def visible(row: dict) -> dict:
    return json.loads(row["messages"][1]["content"])


def evidence_equal(flat: dict, graph: dict) -> bool:
    flat_user, graph_user = visible(flat), visible(graph)
    return (
        flat_user["task"] == graph_user["task"]
        and flat_user["trajectory"]["coverage"] == graph_user["run_evidence"]["coverage"]
        and flat_user["trajectory"]["events"] == graph_user["run_evidence"]["observed"]
        and flat_user["trajectory"]["final_output"]
        == graph_user["run_evidence"]["final_output"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    flat_train = read(args.validation_dir / "flat/train.jsonl.zip", "train.jsonl")
    graph_train = read(args.validation_dir / "graph/train.jsonl.zip", "train.jsonl")
    flat_test = read(args.validation_dir / "flat/test.jsonl")
    graph_test = read(args.validation_dir / "graph/test.jsonl")
    manifest = json.loads(
        (args.validation_dir / "manifest.json").read_text(encoding="utf-8-sig")
    )

    violations: list[str] = []
    for split, rows in (("train", flat_train), ("test", flat_test)):
        for index, row in enumerate(rows):
            text = row["messages"][1]["content"]
            obj = visible(row)
            if FORBIDDEN.search(text):
                violations.append(f"{split}:{index}:forbidden-visible-text")
            if any(key in obj for key in ("graph", "topology", "graph_candidates")):
                violations.append(f"{split}:{index}:flat-contains-graph")

    def ids(rows: list[dict]) -> list[str]:
        return [row["metadata"]["run_id"] for row in rows]

    train_ids, test_ids = ids(flat_train), ids(flat_test)
    report = {
        "protocol": manifest["protocol"],
        "seed": manifest["seed"],
        "train_rows": len(flat_train),
        "test_rows": len(flat_test),
        "actual_train_fraction": len(flat_train) / manifest["train_source_rows"],
        "flat_graph_train_run_order_equal": ids(flat_train) == ids(graph_train),
        "flat_graph_test_run_order_equal": ids(flat_test) == ids(graph_test),
        "flat_graph_train_targets_equal": all(
            flat["messages"][2]["content"] == graph["messages"][2]["content"]
            for flat, graph in zip(flat_train, graph_train)
        ),
        "flat_graph_test_targets_equal": all(
            flat["messages"][2]["content"] == graph["messages"][2]["content"]
            for flat, graph in zip(flat_test, graph_test)
        ),
        "flat_graph_train_evidence_equal": all(
            evidence_equal(flat, graph)
            for flat, graph in zip(flat_train, graph_train)
        ),
        "flat_graph_test_evidence_equal": all(
            evidence_equal(flat, graph) for flat, graph in zip(flat_test, graph_test)
        ),
        "train_test_run_overlap": len(set(train_ids) & set(test_ids)),
        "manifest_train_ids_equal": train_ids == manifest["train_run_ids"],
        "manifest_test_ids_equal": test_ids == manifest["test_run_ids"],
        "train_verdict_distribution": Counter(
            row["metadata"]["verdict"] for row in flat_train
        ),
        "test_verdict_distribution": Counter(
            row["metadata"]["verdict"] for row in flat_test
        ),
        "train_attack_mode_distribution": Counter(
            row["metadata"]["attack_mode"] for row in flat_train
        ),
        "test_attack_mode_distribution": Counter(
            row["metadata"]["attack_mode"] for row in flat_test
        ),
        "violations": violations,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    required_true = (
        "flat_graph_train_run_order_equal",
        "flat_graph_test_run_order_equal",
        "flat_graph_train_targets_equal",
        "flat_graph_test_targets_equal",
        "flat_graph_train_evidence_equal",
        "flat_graph_test_evidence_equal",
        "manifest_train_ids_equal",
        "manifest_test_ids_equal",
    )
    if violations or report["train_test_run_overlap"] or not all(
        report[key] for key in required_true
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
