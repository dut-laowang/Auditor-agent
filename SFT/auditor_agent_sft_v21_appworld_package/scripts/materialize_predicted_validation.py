from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v20-validation", required=True, type=Path)
    parser.add_argument("--controls", required=True, type=Path)
    parser.add_argument("--gold-train", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    validation = read(args.v20_validation)
    controls = read(args.controls)
    by_id = {row["run_id"]: row for row in controls}
    ids = [row["metadata"]["run_id"] for row in validation]
    if len(validation) != 406 or set(ids) != set(by_id) or len(by_id) != 406:
        raise RuntimeError("Predicted controls do not exactly cover frozen validation IDs")
    output = []
    for row in validation:
        item = copy.deepcopy(row)
        control = by_id[item["metadata"]["run_id"]]
        user = json.loads(item["messages"][1]["content"])
        user["audit_control"] = {
            "verdict": control["pred"],
            "scope": control["pred_scope"],
            "component_ids": control["pred_components"],
        }
        item["messages"][1]["content"] = json.dumps(user, ensure_ascii=False)
        item["messages"][0]["content"] += (
            "\nThe audit_control object is an upstream structured decision. Use it as the "
            "decision and localization basis, then produce the same complete JSON audit schema."
        )
        output.append(item)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Reuse the immutable gold-control training file byte-for-byte.
    (args.output_dir / "train.jsonl").write_bytes(args.gold_train.read_bytes())
    with (args.output_dir / "validation.jsonl").open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    contract = {
        "rows": 406,
        "validation_source_sha256": hashlib.sha256(args.v20_validation.read_bytes()).hexdigest(),
        "controls_sha256": hashlib.sha256(args.controls.read_bytes()).hexdigest(),
        "control_source": "frozen Qwen discriminative heads; validation gold excluded",
        "sealed_test_accessed": False,
    }
    (args.output_dir / "contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main()
