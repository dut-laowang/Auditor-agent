import argparse
import json


def get_nested(obj, path, default=0.0):
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True, help="metrics.json from baseline or v2 eval")
    parser.add_argument("--after", required=True, help="metrics.json from candidate eval")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()

    before = json.load(open(args.before, encoding="utf-8"))
    after = json.load(open(args.after, encoding="utf-8"))
    fields = {
        "binary_accuracy": ["binary_accuracy"],
        "safe_f1": ["binary_report", "safe", "f1-score"],
        "unsafe_f1": ["binary_report", "unsafe", "f1-score"],
        "macro_f1": ["binary_report", "macro avg", "f1-score"],
        "weighted_f1": ["binary_report", "weighted avg", "f1-score"],
        "valid_json_rate": ["audit_trace_quality", "valid_json_rate"],
        "has_audit_trace_rate": ["audit_trace_quality", "has_audit_trace_rate"],
        "avg_trace_steps": ["audit_trace_quality", "avg_trace_steps"],
        "avg_evidence_refs": ["audit_trace_quality", "avg_evidence_refs"],
        "evidence_ref_validity_rate": ["audit_trace_quality", "evidence_ref_validity_rate"],
    }
    rows = {}
    for name, path in fields.items():
        b = float(get_nested(before, path))
        a = float(get_nested(after, path))
        rows[name] = {"before": b, "after": a, "delta": a - b}
    result = {
        "before": args.before,
        "after": args.after,
        "comparison": rows,
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        json.dump(result, open(args.output, "w", encoding="utf-8"), indent=2)


if __name__ == "__main__":
    main()
