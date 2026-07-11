import argparse
import json
import os


def load_json(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def sft_row(name, metrics):
    br = metrics.get("binary_report", {})
    return {
        "method": name,
        "n": metrics.get("n"),
        "binary_accuracy": metrics.get("binary_accuracy"),
        "safe_f1": br.get("safe", {}).get("f1-score"),
        "unsafe_f1": br.get("unsafe", {}).get("f1-score"),
        "run_auroc": "N/A",
        "agent_loc_top1": "N/A",
        "agent_loc_top2": "N/A",
        "agent_target_f1": "N/A",
        "surface_f1": "available_in_sft_predictions",
        "objective_f1": "available_in_sft_predictions",
        "json_audit_quality": metrics.get("audit_trace_quality", "available"),
    }


def gnn_row(name, metrics):
    cm = metrics.get("comparable_metrics", {})
    br = cm.get("binary_report", {})
    ar = cm.get("agent_localization_report", {})
    return {
        "method": name,
        "agent_label_policy": metrics.get("agent_label_policy"),
        "n": metrics.get("n"),
        "binary_accuracy": cm.get("binary_accuracy"),
        "safe_f1": br.get("safe", {}).get("f1-score"),
        "unsafe_f1": br.get("unsafe", {}).get("f1-score"),
        "run_auroc": cm.get("run_auroc"),
        "agent_loc_top1": cm.get("agent_top1_hit"),
        "agent_loc_top2": cm.get("agent_top2_hit"),
        "agent_target_f1": ar.get("target_agent", {}).get("f1-score"),
        "surface_f1": "N/A",
        "objective_f1": "N/A",
        "json_audit_quality": "N/A",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-policy", default="unknown")
    parser.add_argument("--v8-metrics")
    parser.add_argument("--v12-metrics")
    parser.add_argument("--gsafeguard-metrics")
    parser.add_argument("--blindguard-metrics")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    v8 = load_json(args.v8_metrics)
    v12 = load_json(args.v12_metrics)
    gs = load_json(args.gsafeguard_metrics)
    bg = load_json(args.blindguard_metrics)
    if v8:
        rows.append(sft_row("V8-SFT", v8))
    if v12:
        rows.append(sft_row("V12-SFT", v12))
    if gs:
        rows.append(gnn_row("G-Safeguard-style GAT", gs))
    if bg:
        rows.append(gnn_row("BlindGuard-style TAM", bg))
    result = {
        "comparison_scope": "Common MARBLE balanced test set. GNN baselines are adapted to agent-level anomaly detection; non-native SFT audit metrics are N/A for GNN.",
        "agent_label_policy": args.label_policy,
        "label_policy_note": (
            "strict_agent: only N::agentX is converted to an agent label. "
            "agent_or_tool_owner: N::agentX and T::agentX are converted to agent labels. "
            "E::a->b and G::run are not converted to agent labels."
        ),
        "rows": rows,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
