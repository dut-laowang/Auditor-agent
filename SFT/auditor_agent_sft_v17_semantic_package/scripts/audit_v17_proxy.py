from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


LABELS = ["clean_safe", "attack_failed", "attack_success"]
ENTITY_RE = re.compile(r"\[REDACTED_ENTITY_[A-F0-9]{10}\]")


def load(path: Path) -> tuple[list[str], list[str], list[list[str]]]:
    texts, labels, structural = [], [], []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            text = row["messages"][1]["content"]
            user = json.loads(text)
            meta = row["metadata"]
            texts.append(text)
            labels.append(meta["verdict"])
            structural.append(
                [
                    str(meta.get("scenario")),
                    str(meta.get("topology")),
                    str(meta.get("attack_mode")),
                    str(bool(ENTITY_RE.search(text))),
                    str("Protected context" in text),
                    str(len(user.get("run_evidence", {}).get("observed", []))),
                    str(len(user.get("run_evidence", {}).get("final_output", []))),
                ]
            )
    return texts, labels, structural


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    args = parser.parse_args()

    train_text, train_y, train_struct = load(args.dataset_dir / "train.jsonl")
    test_text, test_y, test_struct = load(args.dataset_dir / "test.jsonl")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
        sublinear_tf=True,
    )
    train_x = vectorizer.fit_transform(train_text)
    test_x = vectorizer.transform(test_text)
    lexical = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    )
    lexical.fit(train_x, train_y)
    lexical_pred = lexical.predict(test_x)

    encoder = OneHotEncoder(handle_unknown="ignore")
    train_s = encoder.fit_transform(train_struct)
    test_s = encoder.transform(test_struct)
    shallow = DecisionTreeClassifier(
        max_depth=3, min_samples_leaf=20, class_weight="balanced", random_state=42
    )
    shallow.fit(train_s, train_y)
    shallow_pred = shallow.predict(test_s)

    entity_by_label = Counter()
    rows_by_label = Counter(test_y)
    for text, label in zip(test_text, test_y):
        entity_by_label[label] += bool(ENTITY_RE.search(text))

    report = {
        "purpose": (
            "Diagnostic proxy audit. Lexical predictability can reflect legitimate "
            "attack semantics and is not by itself evidence of label leakage."
        ),
        "test_rows": len(test_y),
        "test_label_distribution": dict(rows_by_label),
        "tfidf_word_bigram": {
            "accuracy": accuracy_score(test_y, lexical_pred),
            "report": classification_report(
                test_y,
                lexical_pred,
                labels=LABELS,
                zero_division=0,
                output_dict=True,
            ),
            "prediction_distribution": dict(Counter(lexical_pred)),
        },
        "shallow_structural_proxy": {
            "features": [
                "scenario",
                "topology",
                "attack_mode",
                "has_redacted_entity",
                "has_protected_context",
                "observed_event_count",
                "final_output_count",
            ],
            "accuracy": accuracy_score(test_y, shallow_pred),
            "report": classification_report(
                test_y,
                shallow_pred,
                labels=LABELS,
                zero_division=0,
                output_dict=True,
            ),
            "prediction_distribution": dict(Counter(shallow_pred)),
        },
        "redacted_entity_visibility": {
            label: {
                "rows": rows_by_label[label],
                "rows_with_entity": entity_by_label[label],
                "rate": (
                    entity_by_label[label] / rows_by_label[label]
                    if rows_by_label[label]
                    else 0.0
                ),
            }
            for label in LABELS
        },
    }
    (args.dataset_dir / "proxy_audit_v17.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
