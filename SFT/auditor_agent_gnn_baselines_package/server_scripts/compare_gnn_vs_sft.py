import argparse
import json
import os
import re


AGENT_RE = re.compile(r"agent\d+")


def load_json(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def safe_div(num, den):
    return num / den if den else None


def build_agent_gold(gnn_predictions):
    gold = {}
    agents = {}
    for row in gnn_predictions:
        rid = row.get("run_id")
        if not rid:
            continue
        gold[rid] = set(row.get("gold_agents") or [])
        agents[rid] = [item.get("agent") for item in row.get("ranked_agents", []) if item.get("agent")]
    return gold, agents


def parse_generation(row):
    try:
        return json.loads(row.get("generation", "{}"))
    except Exception:
        return {}


def sft_projected_agents(row, label_policy, valid_agents):
    obj = parse_generation(row)
    loc = obj.get("localization") or obj.get("audit", {}).get("localization") or {}
    projected = set()

    for agent in loc.get("affected_nodes") or []:
        if isinstance(agent, str) and AGENT_RE.fullmatch(agent):
            projected.add(agent)
    target_agent = loc.get("target_agent")
    if isinstance(target_agent, str) and AGENT_RE.fullmatch(target_agent):
        projected.add(target_agent)

    for component_id in loc.get("component_ids") or []:
        if not isinstance(component_id, str):
            continue
        node_match = re.fullmatch(r"N::(agent\d+)", component_id)
        tool_match = re.fullmatch(r"T::(agent\d+)", component_id)
        if node_match:
            projected.add(node_match.group(1))
        elif label_policy == "agent_or_tool_owner" and tool_match:
            projected.add(tool_match.group(1))
        # E::a->b and G::run are intentionally not projected to agents.

    return projected & set(valid_agents)


def agent_projection_metrics(gold_by_run, agents_by_run, pred_by_run):
    tp = fp = fn = tn = 0
    positive_graphs = hit = exact = empty = 0
    for rid, gold_set in gold_by_run.items():
        valid_agents = agents_by_run.get(rid, [])
        pred_set = set(pred_by_run.get(rid, set()))
        if not pred_set:
            empty += 1
        if gold_set:
            positive_graphs += 1
            hit += int(bool(pred_set & gold_set))
            exact += int(pred_set == gold_set)
        for agent in valid_agents:
            gold = agent in gold_set
            pred = agent in pred_set
            if gold and pred:
                tp += 1
            elif not gold and pred:
                fp += 1
            elif gold and not pred:
                fn += 1
            else:
                tn += 1
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall) if precision is not None and recall is not None and (precision + recall) else None
    return {
        "agent_target_precision": precision,
        "agent_target_recall": recall,
        "agent_target_f1": f1,
        "agent_target_tp": tp,
        "agent_target_fp": fp,
        "agent_target_fn": fn,
        "agent_target_tn": tn,
        "agent_hit": safe_div(hit, positive_graphs),
        "exact_agent_set": safe_div(exact, positive_graphs),
        "positive_agent_graphs": positive_graphs,
        "pred_empty_runs": empty,
    }


def sft_row(name, metrics, projection=None):
    br = metrics.get("binary_report", {})
    projection = projection or {}
    return {
        "method": name,
        "n": metrics.get("n"),
        "binary_accuracy": metrics.get("binary_accuracy"),
        "safe_f1": br.get("safe", {}).get("f1-score"),
        "unsafe_f1": br.get("unsafe", {}).get("f1-score"),
        "run_auroc": "N/A",
        "agent_loc_top1": "N/A",
        "agent_loc_top2": "N/A",
        "agent_target_precision": projection.get("agent_target_precision"),
        "agent_target_recall": projection.get("agent_target_recall"),
        "agent_target_f1": projection.get("agent_target_f1"),
        "agent_hit": projection.get("agent_hit"),
        "exact_agent_set": projection.get("exact_agent_set"),
        "surface_f1": "available_in_sft_predictions",
        "objective_f1": "available_in_sft_predictions",
        "edge_tool_global_localization": "available_in_sft_predictions",
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
        "agent_target_precision": ar.get("target_agent", {}).get("precision"),
        "agent_target_recall": ar.get("target_agent", {}).get("recall"),
        "agent_target_f1": ar.get("target_agent", {}).get("f1-score"),
        "agent_hit": cm.get("agent_hit"),
        "exact_agent_set": cm.get("exact_agent_set"),
        "surface_f1": "N/A",
        "objective_f1": "N/A",
        "edge_tool_global_localization": "N/A",
        "json_audit_quality": "N/A",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-policy", default="unknown")
    parser.add_argument("--v8-metrics")
    parser.add_argument("--v8-predictions")
    parser.add_argument("--v12-metrics")
    parser.add_argument("--v12-predictions")
    parser.add_argument("--openai-name", default="GPT-4o-mini")
    parser.add_argument("--openai-metrics")
    parser.add_argument("--openai-predictions")
    parser.add_argument("--gsafeguard-metrics")
    parser.add_argument("--gsafeguard-predictions")
    parser.add_argument("--blindguard-metrics")
    parser.add_argument("--blindguard-predictions")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    v8 = load_json(args.v8_metrics)
    v12 = load_json(args.v12_metrics)
    openai_metrics = load_json(args.openai_metrics)
    gs = load_json(args.gsafeguard_metrics)
    bg = load_json(args.blindguard_metrics)
    gs_predictions = load_jsonl(args.gsafeguard_predictions)
    bg_predictions = load_jsonl(args.blindguard_predictions)
    gold_by_run, agents_by_run = build_agent_gold(gs_predictions or bg_predictions)
    v8_projection = None
    v12_projection = None
    openai_projection = None
    if gold_by_run:
        v8_rows = load_jsonl(args.v8_predictions)
        v12_rows = load_jsonl(args.v12_predictions)
        openai_rows = load_jsonl(args.openai_predictions)
        v8_projection = agent_projection_metrics(
            gold_by_run,
            agents_by_run,
            {row.get("run_id"): sft_projected_agents(row, args.label_policy, agents_by_run.get(row.get("run_id"), [])) for row in v8_rows},
        )
        v12_projection = agent_projection_metrics(
            gold_by_run,
            agents_by_run,
            {row.get("run_id"): sft_projected_agents(row, args.label_policy, agents_by_run.get(row.get("run_id"), [])) for row in v12_rows},
        )
        openai_projection = agent_projection_metrics(
            gold_by_run,
            agents_by_run,
            {row.get("run_id"): sft_projected_agents(row, args.label_policy, agents_by_run.get(row.get("run_id"), [])) for row in openai_rows},
        )
    if v8:
        rows.append(sft_row("V8-SFT", v8, v8_projection))
    if v12:
        rows.append(sft_row("V12-SFT", v12, v12_projection))
    if openai_metrics:
        rows.append(sft_row(args.openai_name, openai_metrics, openai_projection))
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
        "sft_agent_projection_note": (
            "SFT agent localization is a projected diagnostic metric, not the native SFT output space. "
            "Under strict_agent, only N::agentX is counted; under agent_or_tool_owner, N::agentX and T::agentX are counted. "
            "E::a->b and G::run are not counted as agents."
        ),
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
