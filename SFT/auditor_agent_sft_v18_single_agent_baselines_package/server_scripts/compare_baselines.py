from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def compact(metrics: dict) -> dict:
    result = {
        "n": metrics["n"],
        "binary_accuracy": metrics.get(
            "binary_accuracy", metrics.get("binary_accuracy_against_v18_outcome_gold")
        ),
        "parse_success_rate": metrics["parse_success_rate"],
    }
    if "three_class_accuracy" in metrics:
        result.update(
            {
                "three_class_accuracy": metrics["three_class_accuracy"],
                "three_class_macro_f1": metrics["three_class_report"]["macro avg"][
                    "f1-score"
                ],
                "unsafe_precision": metrics["binary_report"]["unsafe"]["precision"],
                "unsafe_recall": metrics["binary_report"]["unsafe"]["recall"],
                "localization_f1": metrics["localization"]["component_micro_f1"],
                "scope_accuracy": metrics["localization"]["scope_accuracy"],
            }
        )
    else:
        report = metrics["binary_report_against_v18_outcome_gold"]
        result.update(
            {
                "unsafe_precision": report["unsafe"]["precision"],
                "unsafe_recall": report["unsafe"]["recall"],
                "warning": metrics["gold_semantics"],
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flat", required=True, type=Path)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--agentdog-official", required=True, type=Path)
    parser.add_argument("--agentdog-adapted", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summary = {
        "protocol": {
            "agentdog_official": (
                "Official action-safety output scored against V18 outcome gold only "
                "as a labeled transfer diagnostic; semantics are not equivalent."
            ),
            "agentdog_outcome_adapted": "Same frozen model, preregistered outcome prompt.",
            "v18_flat": "Same V18 data/model/hyperparameters; explicit graph removed.",
            "v18_graph": "Existing V18-Graph/Ours.",
        },
        "results": {
            "agentdog_official": compact(load(args.agentdog_official)),
            "agentdog_outcome_adapted": compact(load(args.agentdog_adapted)),
            "v18_flat": compact(load(args.flat)),
            "v18_graph": compact(load(args.graph)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
