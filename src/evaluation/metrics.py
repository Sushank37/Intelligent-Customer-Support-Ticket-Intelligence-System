"""Evaluation Metrics for Ticket Classification."""

from typing import Dict, Any
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support


def evaluate_classification(y_true: list, y_pred: list) -> Dict[str, Any]:
    """Compute standard classification evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, output_dict=True)

    return {
        "accuracy": float(acc),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "report": report
    }
