"""Model evaluation helpers."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_model(pipeline, X_test: pd.DataFrame, y_test: pd.Series, roc_points: int = 200) -> dict:
    """Score a fitted pipeline on the held-out test set.

    Returns a JSON-serializable dict with headline metrics, the confusion
    matrix, and a downsampled ROC curve (so the app can plot it without
    reloading the model or data).
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    if len(fpr) > roc_points:
        idx = np.linspace(0, len(fpr) - 1, roc_points).astype(int)
        fpr, tpr = fpr[idx], tpr[idx]

    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "roc_curve": {"fpr": fpr.round(4).tolist(), "tpr": tpr.round(4).tolist()},
    }
