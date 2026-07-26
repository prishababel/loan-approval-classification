"""Fairness analysis (notebook section 12).

Compares model behavior across demographic groups: for gender and age band,
computes each group's actual and predicted approval rates, true/false
positive rates, and precision, plus the largest per-dimension gaps
(demographic-parity gap on predicted approval rate, equal-opportunity gap on
TPR).
"""

import numpy as np
import pandas as pd

from . import config
from .data import age_band


def _group_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    def rate(num: int, den: int) -> float | None:
        return round(num / den, 4) if den else None

    return {
        "n": len(y_true),
        "actual_approval_rate": rate(tp + fn, len(y_true)),
        "predicted_approval_rate": rate(tp + fp, len(y_true)),
        "accuracy": rate(tp + tn, len(y_true)),
        "tpr": rate(tp, tp + fn),
        "fpr": rate(fp, fp + tn),
        "precision": rate(tp, tp + fp),
    }


def fairness_report(pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Per-group metrics for gender and age band, with disparity gaps."""
    y_true = y_test.to_numpy()
    y_pred = pipeline.predict(X_test)

    dimensions = {
        "person_gender": X_test["person_gender"],
        "age_band": age_band(X_test["person_age"]),
    }

    report: dict[str, dict] = {}
    for dim, values in dimensions.items():
        groups: dict[str, dict] = {}
        for group in [g for g in values.unique() if pd.notna(g)]:
            mask = (values == group).to_numpy()
            groups[str(group)] = _group_row(y_true[mask], y_pred[mask])

        rates = [g["predicted_approval_rate"] for g in groups.values() if g["predicted_approval_rate"] is not None]
        tprs = [g["tpr"] for g in groups.values() if g["tpr"] is not None]
        report[dim] = {
            "groups": groups,
            "demographic_parity_gap": round(max(rates) - min(rates), 4) if rates else None,
            "equal_opportunity_gap": round(max(tprs) - min(tprs), 4) if tprs else None,
        }
    return report
