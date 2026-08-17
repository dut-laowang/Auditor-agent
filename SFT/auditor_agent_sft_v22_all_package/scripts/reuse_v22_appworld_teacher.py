from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FIELDS = {"causal_explanation", "recommended_action", "confidence"}
APPWORLD_TRACK = "marble_appworld"
APPWORLD_ROWS = 3122


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_id(row: dict) -> str:
    return str(row.get("run_id") or row.get("metadata", {}).get("run_id") or "")


def validate_teacher(rows: list[dict], label: str) -> None:
    seen = set()
    for row in rows:
        identifier = run_id(row)
        if not identifier or identifier in seen:
            raise RuntimeError(f"{label}: missing/duplicate run_id")
        seen.add(identifier)
        if row.get("prompt_version") != "v22-three-field-audit-grade-v4":
            raise RuntimeError(f"{label}: prompt version mismatch: {identifier}")
        extra = row.get("enrichment", {})
        if set(extra) != FIELDS or extra.get("confidence") not in {"high", "medium", "low"}:
            raise RuntimeError(f"{label}: invalid enrichment schema: {identifier}")
        if not all(isinstance(extra[key], str) and extra[key].strip() for key in FIELDS):
            raise RuntimeError(f"{label}: empty enrichment field: {identifier}")


def prepare(args: argparse.Namespace) -> None:
    train = read(args.v22_train)
    index = read(args.track_index)
    membership = {row["run_id"]: row["track"] for row in index}
    if len(membership) != len(index):
        raise RuntimeError("Duplicate run_id in V22-ALL train track index")
    unknown = [run_id(row) for row in train if run_id(row) not in membership]
    if unknown:
        raise RuntimeError(f"V22 train rows missing from track index: {unknown[:5]}")
    appworld = [row for row in train if membership[run_id(row)] == APPWORLD_TRACK]
    new_rows = [row for row in train if membership[run_id(row)] != APPWORLD_TRACK]
    if len(appworld) != APPWORLD_ROWS:
        raise RuntimeError(f"Reusable AppWorld subset changed: {len(appworld)} != {APPWORLD_ROWS}")
    write(args.appworld_train, appworld)
    write(args.new_train, new_rows)

    prior_contract = json.loads(args.prior_contract.read_text(encoding="utf-8"))
    prior = read(args.prior_teacher)
    if prior_contract.get("rows") != APPWORLD_ROWS or prior_contract.get("source_rows") != APPWORLD_ROWS:
        raise RuntimeError("Prior AppWorld teacher row contract mismatch")
    if prior_contract.get("train_sha256") != sha256(args.appworld_train):
        raise RuntimeError("Prior AppWorld teacher source hash does not match the V22-ALL AppWorld subset")
    if prior_contract.get("output_sha256") != sha256(args.prior_teacher):
        raise RuntimeError("Prior AppWorld teacher output hash mismatch")
    if prior_contract.get("validation_gold_accessed") is not False or prior_contract.get("sealed_test_accessed") is not False:
        raise RuntimeError("Prior AppWorld teacher leakage contract is not clean")
    validate_teacher(prior, "prior AppWorld teacher")
    if [run_id(row) for row in prior] != [run_id(row) for row in appworld]:
        raise RuntimeError("Prior AppWorld teacher IDs/order do not exactly match the V22-ALL AppWorld subset")
    result = {
        "status": "PASS",
        "v22_all_train_rows": len(train),
        "reused_appworld_rows": len(appworld),
        "new_teacher_rows": len(new_rows),
        "appworld_train_sha256": sha256(args.appworld_train),
        "prior_teacher_sha256": sha256(args.prior_teacher),
        "new_train_sha256": sha256(args.new_train),
        "appworld_reexpanded": False,
        "sealed_test_accessed": False,
    }
    args.output_contract.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def merge(args: argparse.Namespace) -> None:
    train = read(args.v22_train)
    prior = read(args.prior_teacher)
    new = read(args.new_teacher)
    validate_teacher(prior, "prior AppWorld teacher")
    validate_teacher(new, "new two-track teacher")
    all_teacher = prior + new
    by_id = {run_id(row): row for row in all_teacher}
    if len(by_id) != len(all_teacher):
        raise RuntimeError("Duplicate run_id across reused and new teacher outputs")
    train_ids = [run_id(row) for row in train]
    if set(by_id) != set(train_ids):
        raise RuntimeError(
            f"Teacher/train ID mismatch: teacher={len(by_id)}, train={len(train_ids)}, "
            f"missing={len(set(train_ids) - set(by_id))}, extra={len(set(by_id) - set(train_ids))}"
        )
    combined = [by_id[identifier] for identifier in train_ids]
    write(args.output, combined)
    result = {
        "status": "PASS",
        "rows": len(combined),
        "reused_appworld_rows": len(prior),
        "new_teacher_rows": len(new),
        "output_sha256": sha256(args.output),
        "exact_v22_all_train_order": True,
        "appworld_reexpanded": False,
        "sealed_test_accessed": False,
    }
    args.output_contract.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Reuse the completed V22 AppWorld teacher expansion without rerunning it.")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--v22-train", required=True, type=Path)
    prepare_parser.add_argument("--track-index", required=True, type=Path)
    prepare_parser.add_argument("--prior-teacher", required=True, type=Path)
    prepare_parser.add_argument("--prior-contract", required=True, type=Path)
    prepare_parser.add_argument("--appworld-train", required=True, type=Path)
    prepare_parser.add_argument("--new-train", required=True, type=Path)
    prepare_parser.add_argument("--output-contract", required=True, type=Path)
    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--v22-train", required=True, type=Path)
    merge_parser.add_argument("--prior-teacher", required=True, type=Path)
    merge_parser.add_argument("--new-teacher", required=True, type=Path)
    merge_parser.add_argument("--output", required=True, type=Path)
    merge_parser.add_argument("--output-contract", required=True, type=Path)
    args = parser.parse_args()
    args.output_contract.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "prepare":
        prepare(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
