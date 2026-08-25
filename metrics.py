from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def predict_classes(model, inputs: np.ndarray, class_count: int) -> np.ndarray:
    probabilities = model.predict(inputs, verbose=0)
    if class_count == 2:
        return (probabilities.reshape(-1) >= 0.5).astype(int)
    return np.argmax(probabilities, axis=1)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Weighted F1 is support-weighted one-versus-rest class F1."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
