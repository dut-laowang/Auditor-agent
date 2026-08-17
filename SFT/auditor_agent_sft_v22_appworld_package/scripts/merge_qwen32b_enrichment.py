from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path


FIELDS = {"causal_explanation", "recommended_action", "confidence"}


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v22-data", required=True, type=Path)
    parser.add_argument("--teacher-output", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-train-rows", type=int, default=3122)
    parser.add_argument("--expected-validation-rows", type=int, default=406)
    args = parser.parse_args()
    train = read(args.v22_data / "train.jsonl")
    validation = read(args.v22_data / "validation.jsonl")
    teacher = read(args.teacher_output)
    if (
        len(train) != args.expected_train_rows
        or len(teacher) != args.expected_train_rows
        or len(validation) != args.expected_validation_rows
    ):
        raise RuntimeError("Expanded V22 split size mismatch")
    instruction = (
        " The decision object must also contain causal_explanation, recommended_action, and confidence. "
        "Ground causal_explanation in existing evidence_refs, provide one defensive recommended_action, "
        "and set confidence to high, medium, or low without exposing secrets."
    )
    enriched = []
    for row, generated in zip(train, teacher):
        if row["metadata"]["run_id"] != generated["run_id"]:
            raise RuntimeError("Teacher/train run-id order mismatch")
        extra = generated["enrichment"]
        if set(extra) != FIELDS or extra["confidence"] not in {"high", "medium", "low"}:
            raise RuntimeError(f"Invalid teacher enrichment: {generated['run_id']}")
        if re.search(r'(?i)"(?:password|access_token|api_key|secret)"\s*:', json.dumps(extra)):
            raise RuntimeError(f"Secret-like teacher output: {generated['run_id']}")
        item = copy.deepcopy(row)
        original = json.loads(row["messages"][2]["content"])
        report = copy.deepcopy(original)
        report["decision"].update(extra)
        frozen_before = copy.deepcopy(report); frozen_before["decision"] = {
            key: value for key, value in report["decision"].items() if key not in FIELDS
        }
        if digest(frozen_before) != digest(original):
            raise RuntimeError(f"Frozen report fields changed: {generated['run_id']}")
        item["messages"][0]["content"] += instruction
        item["messages"][2]["content"] = json.dumps(report, ensure_ascii=False)
        enriched.append(item)
    validation_out = copy.deepcopy(validation)
    for row in validation_out:
        row["messages"][0]["content"] += instruction
    train_output = args.output_dir / "train.jsonl"
    validation_output = args.output_dir / "validation.jsonl"
    write(train_output, enriched)
    write(validation_output, validation_out)
    contract = {
        "version": "V22-enriched-audit-v2",
        "train_rows": args.expected_train_rows, "validation_rows": args.expected_validation_rows,
        "source_train_sha256": sha256(args.v22_data / "train.jsonl"),
        "source_validation_sha256": sha256(args.v22_data / "validation.jsonl"),
        "teacher_output_sha256": sha256(args.teacher_output),
        "expanded_train_sha256": sha256(train_output),
        "expanded_validation_sha256": sha256(validation_output),
        "train_ids_preserved": True, "validation_ids_preserved": True,
        "frozen_report_fields_preserved": True,
        "added_fields": sorted(FIELDS),
        "validation_teacher_enrichment": False,
        "validation_gold_exposed_to_teacher": False,
        "sealed_test_accessed": False,
    }
    (args.output_dir / "EXPANSION_CONTRACT.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main()
