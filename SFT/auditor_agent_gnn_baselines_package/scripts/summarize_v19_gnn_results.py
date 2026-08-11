import argparse
import json
from pathlib import Path

VALIDATION_SHA256 = "2882c5adfe2b9c3e7820f7cf56338dec4bf59e091031763ce8ce46d6aa70609e"
REFERENCE_V19_8B_ACCURACY = 0.7560


def row(path: Path):
    metrics = json.loads(path.read_text(encoding="utf-8"))
    if metrics.get("n") != 1791 or metrics.get("dataset_role") != "validation":
        raise RuntimeError(f"Not the frozen 1,791-row V19 validation result: {path}")
    if metrics.get("data_sha256") != VALIDATION_SHA256:
        raise RuntimeError(f"V19 validation hash mismatch: {path}")
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


def table_row(result):
    percent = lambda value: round(100 * float(value), 2)
    return {
        "method": result["method"] + " (V19-adapted)",
        "condition": "Clean",
        "three_way_accuracy": percent(result["three_class_accuracy"]),
        "delta_accuracy_vs_v19_qwen3_8b": percent(
            result["three_class_accuracy"] - REFERENCE_V19_8B_ACCURACY
        ),
        "macro_f1": percent(result["three_class_macro_f1"]),
        "attack_success_recall": percent(result["attack_success_recall"]),
        "localization_f1": percent(result["localization_micro_f1"]),
        "hit": percent(result["component_hit_rate"]),
        "exact": percent(result["component_exact_match"]),
        "scope_accuracy": percent(result["scope_accuracy"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gat", required=True, type=Path)
    parser.add_argument("--tam", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result_rows = [row(args.gat), row(args.tam)]
    payload = {
        "comparison_contract": "V19 MARBLE validation; identical 1,791 rows and native G/N/E/T candidate space",
        "reference_accuracy_for_delta": {
            "method": "V19 SFT (Qwen3-8B)",
            "accuracy": REFERENCE_V19_8B_ACCURACY,
        },
        "rows": result_rows,
        "main_table_rows": [table_row(result) for result in result_rows],
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    columns = list(payload["main_table_rows"][0])
    tsv = args.output.with_name("main_table_rows.tsv")
    tsv.write_text(
        "\t".join(columns)
        + "\n"
        + "\n".join(
            "\t".join(str(result[column]) for column in columns)
            for result in payload["main_table_rows"]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
