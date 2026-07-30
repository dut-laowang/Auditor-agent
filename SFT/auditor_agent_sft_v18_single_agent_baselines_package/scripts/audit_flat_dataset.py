from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path


FORBIDDEN = re.compile(
    r"ACI_[A-Z0-9_]+|\baci_[a-z0-9_]+\b|\bEND_NEGOTIATION\b|"
    r"success_markers?|attack_metadata|attack_id|payload_id|marker_check|"
    r"\[Injected[^\]]*\]|offline verifier|attack-success index|"
    r"labeled as attack-success",
    re.I,
)


def read(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def normalized(text: str) -> str:
    obj = json.loads(text)
    obj.pop("sample_uid", None)
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def evidence_equal(flat: dict, graph: dict) -> bool:
    flat_user = json.loads(flat["messages"][1]["content"])
    graph_user = json.loads(graph["messages"][1]["content"])
    return (
        flat_user["task"] == graph_user["task"]
        and flat_user["trajectory"]["coverage"] == graph_user["run_evidence"]["coverage"]
        and flat_user["trajectory"]["events"] == graph_user["run_evidence"]["observed"]
        and flat_user["trajectory"]["final_output"]
        == graph_user["run_evidence"]["final_output"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--graph-data-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    flat_train_path = args.data_dir / "train.jsonl"
    if flat_train_path.exists():
        train = read(flat_train_path)
    else:
        with zipfile.ZipFile(args.data_dir / "train.jsonl.zip") as archive:
            train = [
                json.loads(line)
                for line in archive.read("train.jsonl").decode("utf-8-sig").splitlines()
                if line.strip()
            ]
    test = read(args.data_dir / "test.jsonl")
    graph_train_path = args.graph_data_dir / "train.jsonl"
    if graph_train_path.exists():
        graph_train = read(graph_train_path)
    else:
        with zipfile.ZipFile(args.graph_data_dir / "train.jsonl.zip") as archive:
            graph_train = [
                json.loads(line)
                for line in archive.read("train.jsonl").decode("utf-8-sig").splitlines()
                if line.strip()
            ]
    graph_test = read(args.graph_data_dir / "test.jsonl")
    violations: list[str] = []
    for split, items in (("train", train), ("test", test)):
        for index, row in enumerate(items):
            visible = row["messages"][1]["content"]
            obj = json.loads(visible)
            if FORBIDDEN.search(visible):
                violations.append(f"{split}:{index}:forbidden_text")
            for key in ("graph", "topology", "graph_candidates"):
                if key in obj:
                    violations.append(f"{split}:{index}:forbidden_key:{key}")
            if set(obj) != {
                "schema",
                "sample_uid",
                "task",
                "audit_request",
                "trajectory",
                "candidate_component_ids",
            }:
                violations.append(f"{split}:{index}:unexpected_schema")
            candidates = set(obj["candidate_component_ids"])
            agents = sorted(
                value.removeprefix("N::")
                for value in candidates
                if value.startswith("N::")
            )
            expected_edges = {
                f"E::{source}->{target}"
                for source in agents
                for target in agents
                if source != target
            }
            actual_edges = {
                value for value in candidates if value.startswith("E::")
            }
            if actual_edges != expected_edges:
                violations.append(f"{split}:{index}:topology_dependent_edge_vocabulary")

    train_ids = {row["metadata"]["run_id"] for row in train}
    test_ids = {row["metadata"]["run_id"] for row in test}
    train_groups = {
        (row["metadata"]["scenario"], row["metadata"]["sample_id"]) for row in train
    }
    test_groups = {
        (row["metadata"]["scenario"], row["metadata"]["sample_id"]) for row in test
    }
    train_prompts = {normalized(row["messages"][1]["content"]) for row in train}
    test_prompts = {normalized(row["messages"][1]["content"]) for row in test}
    report = {
        "train_rows": len(train),
        "test_rows": len(test),
        "run_id_overlap": len(train_ids & test_ids),
        "task_group_overlap": len(train_groups & test_groups),
        "normalized_prompt_overlap": len(train_prompts & test_prompts),
        "graph_flat_train_run_order_equal": [
            row["metadata"]["run_id"] for row in train
        ]
        == [row["metadata"]["run_id"] for row in graph_train],
        "graph_flat_test_run_order_equal": [
            row["metadata"]["run_id"] for row in test
        ]
        == [row["metadata"]["run_id"] for row in graph_test],
        "graph_flat_train_targets_equal": all(
            flat["messages"][2]["content"] == graph["messages"][2]["content"]
            for flat, graph in zip(train, graph_train)
        ),
        "graph_flat_test_targets_equal": all(
            flat["messages"][2]["content"] == graph["messages"][2]["content"]
            for flat, graph in zip(test, graph_test)
        ),
        "graph_flat_train_observable_evidence_equal": all(
            evidence_equal(flat, graph) for flat, graph in zip(train, graph_train)
        ),
        "graph_flat_test_observable_evidence_equal": all(
            evidence_equal(flat, graph) for flat, graph in zip(test, graph_test)
        ),
        "violations": violations,
    }
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    if violations or any(
        report[key]
        for key in ("run_id_overlap", "task_group_overlap", "normalized_prompt_overlap")
    ):
        raise SystemExit(1)
    for key in (
        "graph_flat_train_run_order_equal",
        "graph_flat_test_run_order_equal",
        "graph_flat_train_targets_equal",
        "graph_flat_test_targets_equal",
        "graph_flat_train_observable_evidence_equal",
        "graph_flat_test_observable_evidence_equal",
    ):
        if not report[key]:
            raise SystemExit(f"Alignment failure: {key}")


if __name__ == "__main__":
    main()
