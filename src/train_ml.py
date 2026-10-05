from __future__ import annotations

import csv
import pickle
from pathlib import Path

import numpy as np

from config import MODELS_DIR, SEED

LABEL_NAMES = ["stroke", "hemorrhage", "mass_effect", "normal"]


class SoftmaxClassifier:
    """Pure NumPy multiclass logistic regression that avoids blocked native DLLs."""

    def __init__(self, learning_rate: float = 0.05, epochs: int = 250, seed: int = SEED):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n_samples, n_features = X.shape
        n_classes = int(y.max()) + 1 if y.size else 1

        self.weights = rng.normal(0.0, 0.1, size=(n_features, n_classes))
        self.bias = np.zeros(n_classes, dtype=float)

        targets = np.eye(n_classes, dtype=float)[y]
        for _ in range(self.epochs):
            logits = X @ self.weights + self.bias
            logits = logits - logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs = probs / probs.sum(axis=1, keepdims=True)

            grad_w = (X.T @ (probs - targets)) / n_samples
            grad_b = (probs - targets).mean(axis=0)

            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        logits = X @ self.weights + self.bias
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


def _accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def _f1_weighted(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    classes = np.unique(np.concatenate([y_true, y_pred]))
    scores = []
    for label in classes:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        support = np.sum(y_true == label)
        scores.append(f1 * support)
    total_support = len(y_true)
    return float(sum(scores) / total_support) if total_support else 0.0


def train_ml_models(X, y, output_dir: Path | str = MODELS_DIR) -> dict:
    """Train and save a pure-NumPy multiclass model."""
    X_array = np.asarray(X, dtype=float)
    y_array = np.asarray(y, dtype=int)

    if X_array.size == 0 or y_array.size == 0:
        raise ValueError("Training data is empty. Populate train/val/test data folders first.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if X_array.ndim == 1:
        X_array = X_array.reshape(1, -1)

    unique_classes = sorted(np.unique(y_array))
    if unique_classes != [0, 1, 2, 3]:
        y_array = np.asarray([LABEL_NAMES.index(label) for label in y_array], dtype=int)

    model = SoftmaxClassifier()
    model.fit(X_array, y_array)

    model_path = output_dir / "logistic_regression.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(model, handle)

    return {
        "logistic_regression": {
            "accuracy": 0.0,
            "f1_weighted": 0.0,
            "model_path": str(model_path),
        }
    }


def predict_with_model(model_path: str | Path, X) -> np.ndarray:
    with Path(model_path).open("rb") as handle:
        model = pickle.load(handle)
    return model.predict(np.asarray(X, dtype=float))


def save_prediction_summary(rows: list[dict], output_path: str | Path, model_name: str = "logistic_regression") -> list[dict]:
    """Save model predictions for the dataset to a CSV summary."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / f'{model_name}.pkl'
    if not model_path.exists():
        raise FileNotFoundError(f'Model file not found: {model_path}')

    feature_names = ['mean_intensity', 'std_intensity', 'min_intensity', 'max_intensity', 'energy', 'edge_density', 'mean_gradient', 'contrast']
    X = np.asarray([[row.get(feature, 0.0) for feature in feature_names] for row in rows], dtype=float)
    predictions = predict_with_model(model_path, X)

    summary = []
    for row, pred in zip(rows, predictions):
        summary_row = dict(row)
        summary_row['predicted_label_index'] = int(pred)
        summary_row['predicted_label'] = LABEL_NAMES[int(pred)]
        summary.append(summary_row)

    with output_path.open('w', newline='', encoding='utf-8') as csv_file:
        fieldnames = list(summary[0].keys()) if summary else ['split', 'label', 'filename', 'path', 'label_index', 'predicted_label_index', 'predicted_label']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)

    return summary
