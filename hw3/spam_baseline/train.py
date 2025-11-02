import argparse
import json
import os
from typing import Tuple

import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from hw3.spam_baseline.data import load_from_url, prepare_dataset, ensure_artifacts_dir
from hw3.spam_baseline.evaluate import evaluate_and_persist


def build_pipeline() -> Pipeline:
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(strip_accents='unicode', lowercase=True, stop_words='english', ngram_range=(1,2), max_features=20000)),
        ("clf", LinearSVC(random_state=42, max_iter=10000)),
    ])
    return pipe


def train_from_url(url: str, output_dir: str = "artifacts") -> Tuple[Pipeline, dict]:
    df = load_from_url(url)
    X_train, X_test, y_train, y_test = prepare_dataset(df)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    ensure_artifacts_dir(output_dir)
    model_path = os.path.join(output_dir, "model.pkl")
    joblib.dump(pipeline, model_path)

    metrics = evaluate_and_persist(pipeline, X_test, y_test, output_dir=output_dir)
    return pipeline, metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train SMS spam baseline model")
    parser.add_argument("--dataset-url", type=str, required=True, help="Raw CSV URL for dataset")
    parser.add_argument("--output-dir", type=str, default="artifacts", help="Directory to write model and metrics")

    args = parser.parse_args(argv)

    pipeline, metrics = train_from_url(args.dataset_url, args.output_dir)
    print("Training complete. Metrics:\n", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
