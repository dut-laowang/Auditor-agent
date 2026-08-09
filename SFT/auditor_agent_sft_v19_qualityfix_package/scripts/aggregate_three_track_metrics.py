from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def select(metrics: dict) -> dict[str, float]:
    return {
        "three_class_accuracy": metrics["three_class_accuracy"],
        "three_class_macro_f1": metrics["three_class_report"]["macro avg"]["f1-score"],
        "binary_accuracy": metrics["binary_accuracy"],
        "binary_macro_f1": metrics["binary_report"]["macro avg"]["f1-score"],
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
    parser.add_argument("--marble", required=True, type=Path)
    parser.add_argument("--autogen", required=True, type=Path)
    parser.add_argument("--mixed", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    tracks = {
        "marble_only": select(json.loads(args.marble.read_text())),
        "autogen_only": select(json.loads(args.autogen.read_text())),
        "mixed": select(json.loads(args.mixed.read_text())),
    }
    average, std = {}, {}
    for key in next(iter(tracks.values())):
        values = [track[key] for track in tracks.values()]
        mean = sum(values) / len(values)
        average[key] = mean
        std[key] = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    output = {
        "tracks": tracks,
        "macro_average_across_three_tracks": average,
        "population_std_across_three_tracks": std,
        "aggregation_note": (
            "Unweighted macro-average across three independently trained/evaluated tracks. "
            "It is a summary, not a fourth model and not a pooled-example metric."
        ),
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
