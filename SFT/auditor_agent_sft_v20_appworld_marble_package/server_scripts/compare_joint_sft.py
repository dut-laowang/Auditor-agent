import argparse
import json
import os


def metric(doc, *paths):
    for path in paths:
        value = doc
        try:
            for key in path:
                value = value[key]
            return float(value)
        except (KeyError, TypeError):
            continue
    raise KeyError(paths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--joint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    baseline = json.load(open(args.baseline, encoding="utf-8"))
    joint = json.load(open(args.joint, encoding="utf-8"))
    paths = {
        "accuracy": (("three_class_accuracy",),),
        "macro_f1": (("three_class_report", "macro avg", "f1-score"),),
        "attack_success_recall": (("three_class_report", "attack_success", "recall"),),
        "localization_f1": (("localization", "component_micro_f1"),),
    }
    output = {"dataset_role": "validation", "baseline": {}, "joint": {}, "delta": {}}
    for name, candidates in paths.items():
        old = metric(baseline, *candidates)
        new = metric(joint, *candidates)
        output["baseline"][name] = old
        output["joint"][name] = new
        output["delta"][name] = new - old
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
