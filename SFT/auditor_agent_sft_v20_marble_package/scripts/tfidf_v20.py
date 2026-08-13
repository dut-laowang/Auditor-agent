from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


LABELS = ["clean_safe", "attack_failed", "attack_success"]


def load(path: Path):
    texts, labels, rows = [], [], []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        texts.append(row["messages"][1]["content"])
        labels.append(row["metadata"]["verdict"])
        rows.append(row)
    return texts, labels, rows


def target(row):
    answer = json.loads(row["messages"][2]["content"])
    localization = answer.get("localization", {})
    return (
        answer["decision"]["verdict"],
        str(localization.get("scope", "none")),
        {str(value) for value in localization.get("component_ids", [])},
    )


def candidate_ids(row):
    user = json.loads(row["messages"][1]["content"])
    return {
        str(candidate["id"])
        for candidate in user.get("graph_candidates", [])
        if isinstance(candidate, dict) and candidate.get("id")
    }


def localization_proxy(train_rows, val_rows):
    # Match the V19 train-only diagnostic proxy: learn localization solely from
    # attack-success training examples, then take Top-1 for a single predicted
    # scope and Top-2 for a predicted multi scope.
    positive_train = [row for row in train_rows if target(row)[0] == "attack_success"]
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True
    )
    train_x = vectorizer.fit_transform(
        [row["messages"][1]["content"] for row in positive_train]
    )
    val_x = vectorizer.transform([row["messages"][1]["content"] for row in val_rows])
    scopes = [target(row)[1] for row in positive_train]
    scope_model = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ).fit(train_x, scopes)
    predicted_scopes = scope_model.predict(val_x)
    encoder = MultiLabelBinarizer()
    component_y = encoder.fit_transform([target(row)[2] for row in positive_train])
    component_model = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        n_jobs=-1,
    ).fit(train_x, component_y)
    probabilities = component_model.predict_proba(val_x)
    records = []
    for index, row in enumerate(val_rows):
        verdict, gold_scope, gold = target(row)
        if verdict != "attack_success" or not gold:
            continue
        available = candidate_ids(row)
        ranked = sorted(
            (
                (float(probabilities[index, column]), str(component_id))
                for column, component_id in enumerate(encoder.classes_)
                if str(component_id) in available
            ),
            reverse=True,
        )
        predicted_scope = str(predicted_scopes[index])
        count = 2 if predicted_scope == "multi" else 1
        predicted = {component_id for _, component_id in ranked[:count]}
        records.append((gold, predicted, gold_scope, predicted_scope))
    tp = sum(len(gold & predicted) for gold, predicted, _, _ in records)
    fp = sum(len(predicted - gold) for gold, predicted, _, _ in records)
    fn = sum(len(gold - predicted) for gold, predicted, _, _ in records)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n_attack_success_with_gold_components": len(records),
        "component_micro_precision": precision,
        "component_micro_recall": recall,
        "component_micro_f1": f1,
        "component_hit_rate": sum(bool(gold & predicted) for gold, predicted, _, _ in records) / len(records),
        "component_exact_match": sum(gold == predicted for gold, predicted, _, _ in records) / len(records),
        "scope_accuracy": sum(gold == predicted for _, _, gold, predicted in records) / len(records),
        "policy": "V19-style train-only TF-IDF diagnostic proxy; attack-success train rows only; predicted-scope Top-1/Top-2 component selection",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    train_x, train_y, train_rows = load(args.data_dir / "train.jsonl")
    val_x, val_y, val_rows = load(args.data_dir / "validation.jsonl")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(vectorizer.fit_transform(train_x), train_y)
    pred = model.predict(vectorizer.transform(val_x))
    report = classification_report(val_y, pred, labels=LABELS, zero_division=0, output_dict=True)
    result = {
        "mode": "tfidf_word_bigram_logistic",
        "dataset_role": "validation",
        "n": len(val_y),
        "three_class_accuracy": accuracy_score(val_y, pred),
        "three_class_report": report,
        "prediction_distribution": dict(Counter(pred)),
        "localization": localization_proxy(train_rows, val_rows),
        "localization_note": "Train-only diagnostic proxy matching the V19 table convention; not a fully isomorphic structured predictor.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
