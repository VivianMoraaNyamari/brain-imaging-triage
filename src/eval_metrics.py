from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score


def evaluate_model(y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[str] | None = None):
    """Compute common metrics for a classification model."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=list(range(len(labels))) if labels else None),
        'classification_report': classification_report(y_true, y_pred, target_names=labels, zero_division=0),
    }
    return metrics
