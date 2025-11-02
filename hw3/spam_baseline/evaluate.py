import json
import os
from typing import Any, Dict

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def evaluate(model, X_test, y_test) -> Dict[str, Any]:
    preds = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }
    return metrics


def evaluate_and_persist(model, X_test, y_test, output_dir: str = "artifacts") -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    metrics = evaluate(model, X_test, y_test)
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics
