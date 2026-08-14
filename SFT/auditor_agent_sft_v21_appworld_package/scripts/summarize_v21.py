from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def row(name: str, metrics: dict) -> dict:
    return {
        "method": name,
        "n": metrics["n"],
        "accuracy": metrics["three_class_accuracy"],
        "macro_f1": metrics["three_class_report"]["macro avg"]["f1-score"],
        "attack_success_recall": metrics["three_class_report"]["attack_success"]["recall"],
        "localization_f1": metrics["localization"]["component_micro_f1"],
        "component_hit_rate": metrics["localization"]["component_hit_rate"],
        "component_exact_match": metrics["localization"]["component_exact_match"],
        "scope_accuracy": metrics["localization"]["scope_accuracy"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v20-qwen", required=True, type=Path)
    parser.add_argument("--v20-modernbert", required=True, type=Path)
    parser.add_argument("--v21", required=True, type=Path)
    parser.add_argument("--head-metrics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    table = [
        row("V20 Qwen3-8B SFT", load(args.v20_qwen)),
        row("V20 ModernBERT", load(args.v20_modernbert)),
        row("V21 Qwen frozen heads + conditional Audit SFT", load(args.v21)),
    ]
    payload = {
        "dataset": "MARBLE x AppWorld",
        "dataset_role": "validation",
        "validation_rows": 406,
        "table": table,
        "head_training": load(args.head_metrics),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
