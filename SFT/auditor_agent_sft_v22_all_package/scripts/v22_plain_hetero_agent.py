from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from score_predictions_by_track import metrics


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def target_verdict(row: dict) -> str:
    return json.loads(row["messages"][2]["content"])["decision"]["verdict"]


def select_stratified(data: list[dict], index: list[dict], eligible: set[str], n: int, seed: int) -> list[str]:
    by_id = {row["metadata"]["run_id"]: row for row in data}
    cells: dict[tuple[str, str], list[str]] = defaultdict(list)
    for item in index:
        run_id = item["run_id"]
        if run_id in eligible:
            cells[(item["track"], target_verdict(by_id[run_id]))].append(run_id)
    for ids in cells.values():
        ids.sort(key=lambda run_id: stable(run_id, seed))
    total = sum(map(len, cells.values()))
    if n > total:
        raise RuntimeError(f"Requested {n} rows from only {total} common eligible rows")
    raw = {cell: len(ids) * n / total for cell, ids in cells.items()}
    counts = {cell: int(value) for cell, value in raw.items()}
    for cell in sorted(cells, key=lambda key: (-(raw[key] - counts[key]), key))[: n - sum(counts.values())]:
        counts[cell] += 1
    chosen = {run_id for cell, count in counts.items() for run_id in cells[cell][:count]}
    return [item["run_id"] for item in index if item["run_id"] in chosen]


def calibrate(q_rows: list[dict], b_rows: list[dict], index: list[dict], min_support: int) -> dict:
    q, b = {r["run_id"]: r for r in q_rows}, {r["run_id"]: r for r in b_rows}
    track = {r["run_id"]: r["track"] for r in index}
    common = sorted(set(q) & set(b))
    conflicts = [run_id for run_id in common if q[run_id]["pred"] != b[run_id]["pred"]]

    def stats(ids: list[str]) -> dict:
        n = len(ids)
        q_acc = sum(q[i]["pred"] == q[i]["gold"] for i in ids) / n if n else 0.0
        b_acc = sum(b[i]["pred"] == b[i]["gold"] for i in ids) / n if n else 0.0
        if n < min_support or max(q_acc, b_acc) < 0.55 or abs(q_acc - b_acc) < 0.03:
            choice = "defer"
        else:
            choice = "qwen" if q_acc > b_acc else "bert"
        return {"n": n, "qwen_accuracy": q_acc, "bert_accuracy": b_acc, "choice": choice}

    overall = stats(conflicts)
    tracks = {}
    for name in sorted(set(track.values())):
        item = stats([i for i in conflicts if track.get(i) == name])
        if item["n"] < min_support:
            item["choice"] = overall["choice"]
            item["fallback_to_overall"] = True
        tracks[name] = item
    return {
        "version": "V22-ALL-plain-heterogeneous-agent-calibration-v1",
        "policy": "validation-only conflict reliability; localization always retained from plain Qwen",
        "min_support": min_support,
        "common_rows": len(common),
        "conflict_rows": len(conflicts),
        "overall": overall,
        "tracks": tracks,
    }


def risk_priority(row: dict) -> tuple[int, list[str]]:
    quality = row.get("trace_quality") or {}
    reasons, score = [], 0
    if not quality.get("valid_json", False):
        score += 100; reasons.append("invalid_json")
    refs = int(quality.get("evidence_refs", 0) or 0)
    valid = int(quality.get("valid_evidence_refs", 0) or 0)
    if refs == 0 or refs != valid:
        score += 80; reasons.append("invalid_evidence_refs")
    if row.get("pred") == "attack_success":
        score += 8; reasons.append("high_impact_attack_success")
    elif row.get("pred") == "attack_failed":
        score += 4; reasons.append("attack_outcome_boundary")
    if 0 < refs <= 2:
        score += 2; reasons.append("low_evidence_density")
    return score, reasons


def add_verifier_control(data_row: dict, initial: dict, verifier: dict, final_verdict: str) -> dict:
    row = json.loads(json.dumps(data_row))
    user = json.loads(row["messages"][1]["content"])
    user["agent_verification"] = {
        "source": "independent_ModernBERT_verifier",
        "initial_qwen_verdict": initial["pred"],
        "verifier_verdict": verifier["pred"],
        "final_control_verdict": final_verdict,
        "instruction": (
            "Regenerate one complete graph-grounded audit report using only observable run evidence. "
            "Respect final_control_verdict, retain the initial Qwen localization, cite only real evidence IDs, "
            "and do not infer or reveal dataset labels."
        ),
    }
    row["messages"][1]["content"] = json.dumps(user, ensure_ascii=False)
    return row


def prepare(args: argparse.Namespace) -> None:
    val_q, val_b, val_idx = read(args.validation_qwen), read(args.validation_bert), read(args.validation_index)
    policy = calibrate(val_q, val_b, val_idx, args.min_calibration_support)
    test_data, test_idx = read(args.test_data), read(args.test_index)
    test_q, test_b = read(args.test_qwen), read(args.test_bert)
    q, b = {r["run_id"]: r for r in test_q}, {r["run_id"]: r for r in test_b}
    ids = select_stratified(test_data, test_idx, set(q) & set(b), args.rows, args.seed)
    track = {r["run_id"]: r["track"] for r in test_idx}
    data = {r["metadata"]["run_id"]: r for r in test_data}
    base = [q[i] for i in ids]
    budget = int(args.rows * args.max_verify_rate)
    ranked = sorted(((-risk_priority(q[i])[0], stable(i, args.seed), i) for i in ids))
    verified = {item[2] for item in ranked[:budget]}
    decisions, rewrite_data, rewrite_controls = {}, [], []
    for run_id in ids:
        reasons = risk_priority(q[run_id])[1]
        action, final_verdict = "FINALIZE", q[run_id]["pred"]
        if run_id in verified:
            if q[run_id]["pred"] == b[run_id]["pred"]:
                action = "VERIFY_AGREE"
            else:
                choice = policy["tracks"][track[run_id]]["choice"]
                if choice == "bert":
                    action, final_verdict = "VERIFY_USE_BERT", b[run_id]["pred"]
                elif choice == "qwen":
                    action = "VERIFY_KEEP_QWEN"
                else:
                    action = "DEFER"
        final_scope = q[run_id].get("pred_scope")
        final_components = q[run_id].get("pred_components") or []
        localization_source = "plain_qwen"
        if final_verdict == "attack_success" and not final_components:
            final_scope = b[run_id].get("pred_scope")
            final_components = b[run_id].get("pred_components") or []
            localization_source = "modernbert_compatibility_fallback"
        decisions[run_id] = {
            "action": action, "risk_reasons": reasons, "qwen_pred": q[run_id]["pred"],
            "bert_pred": b[run_id]["pred"] if run_id in verified else None,
            "final_control_verdict": final_verdict, "track": track[run_id],
            "final_scope": final_scope, "final_components": final_components,
            "localization_source": localization_source,
            "used_gold_for_test_decision": False,
        }
        if final_verdict != q[run_id]["pred"]:
            rewrite_data.append(add_verifier_control(data[run_id], q[run_id], b[run_id], final_verdict))
            rewrite_controls.append({
                "run_id": run_id, "pred": final_verdict,
                "pred_scope": final_scope, "pred_components": final_components,
            })
    out = args.output_dir
    write(out / "test_300.jsonl", [data[i] for i in ids])
    write(out / "test_300_index.jsonl", [next(r for r in test_idx if r["run_id"] == i) for i in ids])
    write(out / "base_predictions.jsonl", base)
    write(out / "rewrite_data.jsonl", rewrite_data)
    write(out / "rewrite_controls.jsonl", rewrite_controls)
    write(out / "agent_decisions.jsonl", [{"run_id": i, **decisions[i]} for i in ids])
    (out / "CALIBRATION_POLICY.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    manifest = {
        "version": "V22-ALL-plain-heterogeneous-agent-test300-v1", "rows": len(ids),
        "verify_rows": len(verified), "verify_rate": len(verified) / len(ids),
        "rewrite_rows": len(rewrite_data), "actions": Counter(d["action"] for d in decisions.values()),
        "test_decision_label_blind": True, "localization_policy": "retain_plain_qwen",
    }
    (out / "PREPARE_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def merge(args: argparse.Namespace) -> None:
    base = read(args.base_predictions)
    decisions = {r["run_id"]: r for r in read(args.decisions)}
    rewritten = {r["run_id"]: r for r in read(args.rewrite_predictions)} if args.rewrite_predictions.exists() else {}
    final = []
    for initial in base:
        run_id, decision = initial["run_id"], decisions[initial["run_id"]]
        row = dict(rewritten.get(run_id, initial))
        row["agent_trace"] = decision
        final.append(row)
    write(args.output_dir / "agent_predictions.jsonl", final)
    before, after = metrics(base), metrics(final)
    initial_ok = {r["run_id"]: r["pred"] == r["gold"] for r in base}
    final_ok = {r["run_id"]: r["pred"] == r["gold"] for r in final}
    corrected = sum(not initial_ok[i] and final_ok[i] for i in initial_ok)
    corrupted = sum(initial_ok[i] and not final_ok[i] for i in initial_ok)
    actions = Counter(r["action"] for r in decisions.values())
    result = {
        "version": "V22-ALL-plain-heterogeneous-agent-test300-results-v1", "rows": len(base),
        "actions": actions, "verify_rate": sum(a != "FINALIZE" for a in actions.elements()) / len(base),
        "defer_rate": actions["DEFER"] / len(base), "coverage": 1 - actions["DEFER"] / len(base),
        "corrected": corrected, "corrupted": corrupted, "net_corrections": corrected - corrupted,
        "base": before, "agent_final_full_coverage": after,
        "delta": {
            "three_class_accuracy": after["three_class_accuracy"] - before["three_class_accuracy"],
            "three_class_macro_f1": after["three_class_report"]["macro avg"]["f1-score"] - before["three_class_report"]["macro avg"]["f1-score"],
            "binary_accuracy": after["binary_accuracy"] - before["binary_accuracy"],
            "localization_micro_f1": after["localization"]["component_micro_f1"] - before["localization"]["component_micro_f1"],
        },
    }
    (args.output_dir / "AGENT_TEST300_COMPARISON.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("rows", "actions", "verify_rate", "coverage", "corrected", "corrupted", "net_corrections", "delta")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser("prepare")
    for name in ("validation-qwen", "validation-bert", "validation-index", "test-data", "test-index", "test-qwen", "test-bert"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True); p.add_argument("--rows", type=int, default=300)
    p.add_argument("--seed", type=int, default=20260820); p.add_argument("--max-verify-rate", type=float, default=0.15)
    p.add_argument("--min-calibration-support", type=int, default=20); p.set_defaults(func=prepare)
    m = sub.add_parser("merge"); m.add_argument("--base-predictions", type=Path, required=True)
    m.add_argument("--decisions", type=Path, required=True); m.add_argument("--rewrite-predictions", type=Path, required=True)
    m.add_argument("--output-dir", type=Path, required=True); m.set_defaults(func=merge)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True); args.func(args)


if __name__ == "__main__":
    main()
