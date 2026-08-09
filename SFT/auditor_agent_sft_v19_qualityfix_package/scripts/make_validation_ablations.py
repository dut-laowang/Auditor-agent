from __future__ import annotations

import argparse
import copy
import json
import random
import re
from pathlib import Path


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def unpack(row: dict) -> tuple[dict, dict, dict]:
    item = copy.deepcopy(row)
    user = json.loads(item["messages"][1]["content"])
    assistant = json.loads(item["messages"][2]["content"])
    return item, user, assistant


def pack(item: dict, user: dict, assistant: dict) -> dict:
    item["messages"][1]["content"] = json.dumps(user, ensure_ascii=False)
    item["messages"][2]["content"] = json.dumps(assistant, ensure_ascii=False)
    return item


def events(user: dict) -> list[dict]:
    evidence = user["run_evidence"]
    return evidence.get("observed", []) + evidence.get("final_output", [])


def transform(row: dict, fn) -> dict:
    item, user, assistant = unpack(row)
    fn(user, assistant)
    return pack(item, user, assistant)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-file", required=True, type=Path)
    parser.add_argument("--shortcut-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-features-per-class", type=int, default=30)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(line) for line in args.validation_file.open(encoding="utf-8-sig") if line.strip()]
    shortcut = json.loads(args.shortcut_report.read_text(encoding="utf-8"))
    phrases = sorted(
        {
            entry["feature"]
            for entries in shortcut["top_features"].values()
            for entry in entries[: args.top_features_per_class]
            if len(entry["feature"].strip()) >= 3
        },
        key=len,
        reverse=True,
    )
    lexical_pattern = re.compile(
        r"(?<!\w)(?:" + "|".join(re.escape(item) for item in phrases) + r")(?!\w)",
        re.I,
    )

    def task_goal_masked(user, _assistant):
        user["task"]["goal"] = "[TASK_GOAL_MASKED]"

    def event_text_masked(user, _assistant):
        for event in events(user):
            if "text" in event:
                event["text"] = "[EVENT_TEXT_MASKED]"

    def outcome_text_masked(user, _assistant):
        for event in user["run_evidence"].get("final_output", []):
            if "text" in event:
                event["text"] = "[OUTCOME_TEXT_MASKED]"

    def structure_links_masked(user, _assistant):
        user["graph"]["topology"] = "[TOPOLOGY_MASKED]"
        user["graph"]["edges"] = []
        for candidate in user.get("graph_candidates", []):
            for key in ("event_refs", "local_event_refs", "incoming_event_refs", "outgoing_event_refs"):
                if key in candidate:
                    candidate[key] = []

    def lexical_shortcuts_masked(user, _assistant):
        user["task"]["goal"] = lexical_pattern.sub("[LEXICAL_CUE_MASKED]", user["task"].get("goal", ""))
        for event in events(user):
            if isinstance(event.get("text"), str):
                event["text"] = lexical_pattern.sub("[LEXICAL_CUE_MASKED]", event["text"])

    def shuffle_events(row):
        def apply(user, _assistant):
            local = random.Random(f"{args.seed}:{row['metadata']['sample_uid']}")
            local.shuffle(user["run_evidence"]["observed"])
        return transform(row, apply)

    # Rotate full text bundles across different gold verdicts while retaining
    # each recipient's event types, graph, IDs, and target. This intentionally
    # breaks lexical-label correspondence without touching the sealed test.
    gold = [json.loads(row["messages"][2]["content"])["decision"]["verdict"] for row in rows]
    donors = []
    for index, label in enumerate(gold):
        donor = (index + 1) % len(rows)
        while gold[donor] == label:
            donor = (donor + 1) % len(rows)
        donors.append(donor)

    def rotate_text(index, row):
        donor_user = json.loads(rows[donors[index]]["messages"][1]["content"])
        donor_texts = [event.get("text", "") for event in events(donor_user) if isinstance(event.get("text"), str)]
        def apply(user, _assistant):
            if not donor_texts:
                return
            cursor = 0
            for event in events(user):
                if "text" in event:
                    event["text"] = donor_texts[cursor % len(donor_texts)]
                    cursor += 1
        return transform(row, apply)

    variants = {
        "task_goal_masked.jsonl": [transform(row, task_goal_masked) for row in rows],
        "event_text_masked.jsonl": [transform(row, event_text_masked) for row in rows],
        "outcome_text_masked.jsonl": [transform(row, outcome_text_masked) for row in rows],
        "structure_links_masked.jsonl": [transform(row, structure_links_masked) for row in rows],
        "lexical_shortcuts_masked.jsonl": [transform(row, lexical_shortcuts_masked) for row in rows],
        "events_shuffled.jsonl": [shuffle_events(row) for row in rows],
        "cross_label_text_rotated.jsonl": [rotate_text(index, row) for index, row in enumerate(rows)],
    }
    for name, payload in variants.items():
        write(args.output_dir / name, payload)
    manifest = {
        "source": str(args.validation_file),
        "rows_per_variant": len(rows),
        "test_accessed": False,
        "top_lexical_features_masked": phrases,
        "variants": {
            "task_goal_masked": "remove task-description cues",
            "event_text_masked": "retain structure and event types/IDs, remove all event text",
            "outcome_text_masked": "remove only final observable outcome text",
            "structure_links_masked": "retain text and component IDs, remove topology, edges, and candidate-event links",
            "lexical_shortcuts_masked": "mask train-derived top TF-IDF cues without using validation labels",
            "events_shuffled": "destroy trajectory order while retaining the same events",
            "cross_label_text_rotated": "replace texts with a different-verdict donor while retaining recipient structure",
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
