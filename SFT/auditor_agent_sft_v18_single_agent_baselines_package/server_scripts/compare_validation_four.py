from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def trained(metrics: dict) -> dict:
    return {
        "n": metrics["n"],
        "parse_success_rate": metrics["parse_success_rate"],
        "binary_accuracy": metrics["binary_accuracy"],
        "unsafe_precision": metrics["binary_report"]["unsafe"]["precision"],
        "unsafe_recall": metrics["binary_report"]["unsafe"]["recall"],
        "three_class_accuracy": metrics["three_class_accuracy"],
        "three_class_macro_f1": metrics["three_class_report"]["macro avg"][
            "f1-score"
        ],
        "localization_f1": metrics["localization"]["component_micro_f1"],
        "localization_exact": metrics["localization"]["component_exact_match"],
        "scope_accuracy": metrics["localization"]["scope_accuracy"],
    }


def frozen(metrics: dict) -> dict:
    report = metrics["binary_report_against_v18_outcome_gold"]
    return {
        "n": metrics["n"],
        "parse_success_rate": metrics["parse_success_rate"],
        "binary_accuracy_against_v18_outcome_gold": metrics[
            "binary_accuracy_against_v18_outcome_gold"
        ],
        "unsafe_precision": report["unsafe"]["precision"],
        "unsafe_recall": report["unsafe"]["recall"],
        "semantic_warning": metrics["gold_semantics"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agentdog-official", required=True, type=Path)
    parser.add_argument("--agentdog-adapted", required=True, type=Path)
    parser.add_argument("--agentdog-finetuned", required=True, type=Path)
    parser.add_argument("--qwen-flat", required=True, type=Path)
    parser.add_argument("--qwen-graph", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = {
        "experiment": "V18 one-third validation; frozen split before training",
        "results": {
            "agentdog_frozen_official": frozen(load(args.agentdog_official)),
            "agentdog_frozen_outcome_adapted": frozen(load(args.agentdog_adapted)),
            "agentdog_v18_flat_sft": trained(load(args.agentdog_finetuned)),
            "qwen3_v18_flat_sft": trained(load(args.qwen_flat)),
            "qwen3_v18_graph_sft_ours": trained(load(args.qwen_graph)),
        },
        "valid_comparisons": {
            "zero_shot_transfer": "AgentDoG frozen official/adapted",
            "domain_adaptation": "AgentDoG frozen adapted vs AgentDoG V18-Flat SFT",
            "external_method": "AgentDoG V18-Flat SFT vs Qwen3 V18-Graph SFT",
            "controlled_graph_ablation": "Qwen3 V18-Flat SFT vs Qwen3 V18-Graph SFT",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
