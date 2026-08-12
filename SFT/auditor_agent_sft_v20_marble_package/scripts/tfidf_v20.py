from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    train_x, train_y, _ = load(args.data_dir / "train.jsonl")
    val_x, val_y, _ = load(args.data_dir / "validation.jsonl")
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
        "localization_note": "Classification-only TF-IDF baseline; no fabricated G/N/E/T localization output.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
