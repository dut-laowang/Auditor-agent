#!/usr/bin/env python3
"""Render publication tables without inventing unavailable experiment results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = ("three_class_accuracy", "three_class_macro_f1", "attack_success_recall", "binary_accuracy", "localization_micro_f1")


def load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def first(paths: list[Path]) -> dict | None:
    for path in paths:
        data = load(path)
        if data is not None:
            return data
    return None


def metric(data: dict | None, key: str) -> str:
    if not data:
        return "TBD"
    value = data.get(key)
    if value is None and key == "three_class_macro_f1":
        value = data.get("three_class_report", {}).get("macro avg", {}).get("f1-score")
    if value is None and key == "localization_micro_f1":
        value = data.get("localization", {}).get("component_micro_f1")
    if value is None and key == "attack_success_recall":
        value = data.get("three_class_report", {}).get("attack_success", {}).get("recall")
    return "TBD" if value is None else f"{100 * float(value):.2f}"


def table(title: str, headers: list[str], rows: list[list[str]]) -> str:
    out = [f"## {title}", "", "| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def latex_escape(value: str) -> str:
    for old, new in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        value = value.replace(old, new)
    return value


def latex_from_markdown(markdown: str) -> str:
    blocks, title, rows = [], None, []
    def flush() -> None:
        nonlocal rows
        if not title or len(rows) < 2:
            rows = []; return
        headers, body = rows[0], rows[2:]
        cols = "l" + "r" * (len(headers) - 1)
        lines = [r"\begin{table*}[t]", r"\centering", r"\small", f"\\caption{{{latex_escape(title)}}}", f"\\begin{{tabular}}{{{cols}}}", r"\toprule"]
        lines += [" & ".join(map(latex_escape, headers)) + r" \\", r"\midrule"]
        lines += [" & ".join(map(latex_escape, row)) + r" \\" for row in body]
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
        blocks.append("\n".join(lines)); rows = []
    for line in markdown.splitlines():
        if line.startswith("## "):
            flush(); title = line[3:]
        elif title and line.startswith("|"):
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            rows.append(cells)
    flush()
    return "% Requires \\usepackage{booktabs}\n" + "\n\n".join(blocks) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--supplement-dir", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    supplement = args.supplement_dir or args.run_dir
    main_methods = [
        ("TF-IDF", ["baselines/tfidf/metrics.json"]),
        ("G-Safeguard (adapted)", ["baselines/gsafeguard/metrics.json"]),
        ("BlindGuard (adapted)", ["baselines/blindguard/metrics.json"]),
        ("XG-Guard (ACL'26, adapted)", ["baselines/xgguard_test/metrics.json"]),
        ("Plain Qwen3-8B SFT", ["qwen3_8b_plain_sft_test/metrics.json"]),
        ("ModernBERT", ["modernbert_sealed_test/metrics.json", "modernbert_eval/test/metrics.json"]),
        ("Bounded Audit Agent", ["plain_hetero_agent_full_test/metrics.json"]),
    ]
    rows = []
    for name, rels in main_methods:
        data = first([root / rel for rel in rels for root in (args.run_dir, supplement)])
        rows.append([name, *(metric(data, key) for key in METRICS), str(data.get("n", "TBD")) if data else "TBD"])
    sections = [table("Main table - Frozen V22-ALL three-track test", ["Method", "Acc.", "Macro-F1", "AS Recall", "Binary Acc.", "Loc. F1", "N"], rows)]

    held_rows = []
    for fold in ("topology__tree", "surface__message", "scenario__research"):
        for method in ("modernbert", "qwen"):
            data = load(supplement / "heldout" / fold / method / "metrics.json")
            held_rows.append([fold.replace("__", "="), method, *(metric(data, key) for key in METRICS), str(data.get("n", "TBD")) if data else "TBD"])
    sections.append(table("Supplement A - Held-out generalization", ["Held out", "Method", "Acc.", "Macro-F1", "AS Recall", "Binary Acc.", "Loc. F1", "N"], held_rows))

    flat = load(supplement / "single_transfer/qwen3_8b_flat_validation/metrics.json")
    graph = first([args.run_dir / "qwen3_8b_plain_sft_validation/metrics.json"])
    transfer = [["AgentDoG 1.5 official", "native action safety", "TBD", "TBD", "not label-equivalent"],
                ["AgentDoG 1.5 outcome-adapted", "flattened full trajectory", "TBD", "TBD", "transfer baseline"],
                ["Qwen3-8B Flat SFT", "same V22 IDs/events; graph removed", metric(flat, "three_class_macro_f1"), metric(flat, "localization_micro_f1"), "controlled ablation"],
                ["Qwen3-8B Graph SFT", "same V22 IDs/events; full graph", metric(graph, "three_class_macro_f1"), metric(graph, "localization_micro_f1"), "ours"]]
    sections.append(table("Supplement B - Single-agent/trajectory-to-MAS transfer", ["Method", "Input", "Macro-F1", "Loc. F1", "Role"], transfer))

    online = [["AgentForesight-7B", "prefix-only online", "Exact-F1", "ASS / FAR", "external AFTraj/Who&When only"],
              ["Our post-hoc auditor", "complete trajectory", "Acc./Macro-F1", "Loc. F1", "not directly comparable"],
              ["Our prefix adapter", "prefix-only online", "TBD", "TBD", "run only after decisive-step labels exist"]]
    sections.append(table("Supplement C - Online auditing protocol", ["Method", "Observation", "Primary", "Localization/cost", "Status"], online))

    agent = first([args.run_dir / "plain_hetero_agent_full_test" / "summary.json", args.run_dir / "plain_hetero_agent_full_test" / "AGENT_FULL_TEST_COMPLETE.json"])
    sections.append(table("Supplement D - Agent utility and cost", ["Variant", "Final Macro-F1", "Final Loc. F1", "Verify rate", "Defer rate", "Coverage", "Extra calls"], [[
        "Bounded Audit Agent", metric(agent, "three_class_macro_f1"), metric(agent, "localization_micro_f1"),
        "TBD" if not agent else f"{100*agent.get('verify_rate', 0):.2f}", "TBD" if not agent else f"{100*agent.get('defer_rate', 0):.2f}",
        "TBD" if not agent else f"{100*agent.get('coverage', 0):.2f}", str(agent.get("verify_rows", "TBD")) if agent else "TBD"
    ]]))
    counterfactual = load(supplement / "counterfactual_results/counterfactual_summary.json")
    cf_rows = []
    if counterfactual:
        clean = counterfactual["clean"]
        cf_rows.append(["Clean", f"{100*clean['three_class_macro_f1']:.2f}", f"{100*clean['localization_micro_f1']:.2f}", "0.00", "0.00"])
        for name, item in counterfactual["counterfactuals"].items():
            values, delta = item["metrics"], item["delta_from_clean"]
            cf_rows.append([name, f"{100*values['three_class_macro_f1']:.2f}", f"{100*values['localization_micro_f1']:.2f}", f"{100*delta['three_class_macro_f1']:+.2f}", f"{100*delta['localization_micro_f1']:+.2f}"])
    else:
        cf_rows = [[name, "TBD", "TBD", "TBD", "TBD"] for name in ("Clean", "Cross-label text rotation", "Event text removed", "Lexical shortcuts masked", "Event shuffling", "Task goal removed", "Structure links removed", "Outcome text removed")]
    sections.append(table("Supplement E - SFT counterfactual ablations (validation only)", ["Condition", "Macro-F1", "Loc. F1", "Delta Macro-F1", "Delta Loc. F1"], cf_rows))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    text = "# V22 experiment tables\n\n`TBD` means the experiment has not been run; no result is imputed.\n\n" + "\n\n".join(sections) + "\n"
    (args.output_dir / "EXPECTED_TABLES.md").write_text(text, encoding="utf-8")
    (args.output_dir / "EXPECTED_TABLES.tex").write_text(latex_from_markdown(text), encoding="utf-8")
    (args.output_dir / "TABLE_STATUS.json").write_text(json.dumps({"status": "PASS", "tbd_cells": text.count("TBD")}, indent=2), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
