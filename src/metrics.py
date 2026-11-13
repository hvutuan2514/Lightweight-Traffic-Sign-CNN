"""Evaluation metrics shared by every model (deep models and HOG + SVM).

Macro averages are taken over all 43 classes, so a class that is never
predicted counts as 0 instead of being silently dropped.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from src.config import NUM_CLASSES

SCALAR_KEYS = ("loss", "accuracy", "macro_precision", "macro_recall", "macro_f1")


def compute_metrics(y_true, y_pred, num_classes=NUM_CLASSES):
    """Accuracy, macro precision/recall/F1, per-class scores, and the confusion matrix."""
    labels = np.arange(num_classes)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist(),
            "support": support.tolist(),
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def scalar_metrics(metrics):
    """Only the headline numbers (drops per-class scores and the confusion matrix)."""
    return {k: metrics[k] for k in SCALAR_KEYS if k in metrics}


def format_metrics(metrics):
    return " | ".join(f"{k} {v:.4f}" for k, v in scalar_metrics(metrics).items())


def save_metrics(metrics, path, **extra):
    """Write metrics (plus any extra info such as run name or config) to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**extra, **metrics}, indent=2))
