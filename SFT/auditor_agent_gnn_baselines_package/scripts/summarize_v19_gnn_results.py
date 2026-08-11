import argparse
import json
from pathlib import Path


def row(path: Path):
    metrics = json.loads(path.read_text(encoding="utf-8"))
    report = metrics["three_class_report"]
    localization = metrics["localization"]
    return {
        "method": metrics["method"],
        "n": metrics["n"],
        "three_class_accuracy": metrics["three_class_accuracy"],
        "three_class_macro_f1": report["macro avg"]["f1-score"],
        "attack_success_recall": report["attack_success"]["recall"],
        "localization_micro_f1": localization["component_micro_f1"],
        "component_hit_rate": localization["component_hit_rate"],
        "component_exact_match": localization["component_exact_match"],
        "scope_accuracy": localization["scope_accuracy"],
        "data_sha256": metrics["data_sha256"],
        "dataset_role": metrics["dataset_role"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gat", required=True, type=Path)
    parser.add_argument("--tam", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = {
        "comparison_contract": "V19 MARBLE validation; identical 1,791 rows and native G/N/E/T candidate space",
        "rows": [row(args.gat), row(args.tam)],
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
