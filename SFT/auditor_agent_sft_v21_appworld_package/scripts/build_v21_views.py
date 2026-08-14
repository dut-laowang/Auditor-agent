from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


VERDICTS = {"clean_safe", "attack_failed", "attack_success"}
SCOPES = {"none", "global", "node", "edge", "tool", "multi"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def target(row: dict) -> dict:
    answer = json.loads(row["messages"][2]["content"])
    return {
        "verdict": answer["decision"]["verdict"],
        "scope": answer["localization"]["scope"],
        "component_ids": [str(value) for value in answer["localization"]["component_ids"]],
    }


def candidates(row: dict) -> list[dict]:
    user = json.loads(row["messages"][1]["content"])
    return [item for item in user.get("graph_candidates", []) if isinstance(item, dict) and item.get("id")]


def visible_head_row(row: dict) -> dict:
    return {
        "run_id": row["metadata"]["run_id"],
        "messages": copy.deepcopy(row["messages"][:2]),
        "graph_candidates": candidates(row),
        "target": target(row),
    }


def conditional_row(row: dict, control: dict) -> dict:
    output = copy.deepcopy(row)
    user = json.loads(output["messages"][1]["content"])
    user["audit_control"] = control
    output["messages"][1]["content"] = json.dumps(user, ensure_ascii=False)
    output["messages"][0]["content"] += (
        "\nThe audit_control object is an upstream structured decision. Use it as the "
        "decision and localization basis, then produce the same complete JSON audit schema."
    )
    return output


def validate(rows: list[dict], split: str) -> None:
    seen = set()
    for index, row in enumerate(rows):
        roles = [message.get("role") for message in row.get("messages", [])]
        if roles != ["system", "user", "assistant"]:
            raise ValueError(f"Bad role contract at {split}:{index}")
        run_id = row.get("metadata", {}).get("run_id")
        if not run_id or run_id in seen:
            raise ValueError(f"Missing/duplicate run_id at {split}:{index}")
        seen.add(run_id)
        gold = target(row)
        if gold["verdict"] not in VERDICTS or gold["scope"] not in SCOPES:
            raise ValueError(f"Bad target at {split}:{index}: {gold}")
        candidate_ids = {str(item["id"]) for item in candidates(row)}
        if not set(gold["component_ids"]).issubset(candidate_ids):
            raise ValueError(f"Gold component outside visible candidates at {split}:{index}")
        visible = json.dumps(row["messages"][:2], ensure_ascii=False)
        if '"audit_control"' in visible:
            raise ValueError(f"Source V20 unexpectedly contains audit_control at {split}:{index}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v20-data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    expected = {
        "train": (3122, "20372c1d2dad08be7d43465d0d4887491ec82b2d9eca8fff61d16a4708124145"),
        "validation": (406, "5dd89c9950337ee277dedb203f0468ae754154c2c7af5d76eafc00514459805c"),
    }
    manifest = {"version": "V21-AppWorld-MARBLE-multihead-conditional-v1", "splits": {}}
    split_ids = {}
    for split, (count, digest) in expected.items():
        source = args.v20_data / f"{split}.jsonl"
        if sha256(source) != digest:
            raise RuntimeError(f"Frozen V20 {split} hash mismatch")
        rows = read_jsonl(source)
        if len(rows) != count:
            raise RuntimeError(f"Frozen V20 {split} row mismatch")
        validate(rows, split)
        split_ids[split] = {row["metadata"]["run_id"] for row in rows}
        head_rows = [visible_head_row(row) for row in rows]
        write_jsonl(args.output_dir / "discriminative" / f"{split}.jsonl", head_rows)
        if split == "train":
            conditional = [conditional_row(row, target(row)) for row in rows]
            write_jsonl(args.output_dir / "conditional_gold" / "train.jsonl", conditional)
        manifest["splits"][split] = {
            "rows": len(rows),
            "source_sha256": digest,
            "run_id_sha256": hashlib.sha256(
                "\n".join(row["metadata"]["run_id"] for row in rows).encode()
            ).hexdigest(),
        }
    if split_ids["train"] & split_ids["validation"]:
        raise RuntimeError("Train/validation run_id leakage")
    manifest["leakage_contract"] = {
        "discriminative_model_visible_messages": "system+user only; no audit_control",
        "discriminative_gold_location": "target field only",
        "conditional_train_control": "gold train targets only",
        "conditional_validation_control": "must be generated by frozen heads",
        "sealed_test_accessed": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
