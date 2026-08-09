from __future__ import annotations

import argparse
import json
from pathlib import Path


def select(metrics: dict) -> dict[str, float]:
    return {
        "three_class_accuracy": metrics["three_class_accuracy"],
        "three_class_macro_f1": metrics["three_class_report"]["macro avg"]["f1-score"],
        "binary_accuracy": metrics["binary_accuracy"],
        "localization_micro_f1": metrics["localization"]["component_micro_f1"],
        "localization_exact_match": metrics["localization"]["component_exact_match"],
        "scope_accuracy": metrics["localization"]["scope_accuracy"],
        "surface_accuracy_attacked": metrics["attack_characterization"]["surface_accuracy_attacked"],
        "objective_accuracy_attacked": metrics["attack_characterization"]["objective_accuracy_attacked"],
        "valid_json_rate": metrics["audit_trace_quality"]["valid_json_rate"],
        "evidence_ref_validity_rate": metrics["audit_trace_quality"]["evidence_ref_validity_rate"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    clean = select(json.loads((args.result_root / "clean" / "metrics.json").read_text()))
    variants = {}
    for path in sorted(args.result_root.glob("*/metrics.json")):
        name = path.parent.name
        if name == "clean":
            continue
        values = select(json.loads(path.read_text()))
        variants[name] = {
            "metrics": values,
            "delta_from_clean": {key: values[key] - clean[key] for key in clean},
        }
    output = {
        "evaluation_role": "validation_only",
        "test_accessed": False,
        "clean": clean,
        "counterfactuals": variants,
        "reading_rule": (
            "Large degradation after masking a modality indicates dependence on that modality. "
            "Strong performance after cross-label text rotation is evidence of structural reliance; "
            "collapse indicates lexical/text reliance."
        ),
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
