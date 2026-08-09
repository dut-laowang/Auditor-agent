from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def load(path: Path):
    rows = [json.loads(line) for line in path.open(encoding="utf-8-sig") if line.strip()]
    texts = [row["messages"][1]["content"] for row in rows]
    labels = [json.loads(row["messages"][2]["content"])["decision"]["verdict"] for row in rows]
    return texts, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", required=True, type=Path)
    parser.add_argument("--validation-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    train_x, train_y = load(args.train_file)
    val_x, val_y = load(args.validation_file)
    vectorizer = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2), min_df=3, max_features=75000, sublinear_tf=True
    )
    train_matrix = vectorizer.fit_transform(train_x)
    val_matrix = vectorizer.transform(val_x)
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    model.fit(train_matrix, train_y)
    predictions = model.predict(val_matrix)
    names = vectorizer.get_feature_names_out()
    top_features = {}
    for index, label in enumerate(model.classes_):
        order = model.coef_[index].argsort()[-30:][::-1]
        top_features[label] = [
            {"feature": str(names[item]), "weight": float(model.coef_[index, item])}
            for item in order
        ]
    report = {
        "evaluation_split": "validation_only",
        "test_accessed": False,
        "accuracy": float(accuracy_score(val_y, predictions)),
        "classification_report": classification_report(
            val_y, predictions, output_dict=True, zero_division=0
        ),
        "top_features": top_features,
        "interpretation": (
            "A high score is a shortcut-risk diagnostic, not an auditor result. "
            "Final test must remain sealed."
        ),
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
