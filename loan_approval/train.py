"""Train and compare loan-approval classifiers, then save the best one.

Usage:
    python -m loan_approval.train [--data PATH] [--out DIR]

Follows the team notebook "Beyond the Credit Score": a class-weighted
Logistic Regression baseline, a GridSearchCV-tuned Logistic Regression
(their featured model), and Random Forest / Hist Gradient Boosting as the
validation-and-comparison extension. All share one preprocessing pipeline
(scale numerics, one-hot categoricals; loan_int_rate excluded as leakage).

Also produced, per notebook sections 12-13:
- a fairness report (gender and age-band group metrics) for the tuned
  Logistic Regression and for the saved best model,
- a "with vs without demographics" ablation of the tuned model.

Writes the best pipeline (by ROC-AUC) to models/model.joblib and the full
report to models/metrics.json.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .data import load_dataset, split_features_target
from .evaluate import evaluate_model
from .fairness import fairness_report

TUNED_LR = "Logistic Regression (tuned)"

PARAM_GRID = {
    "model__C": [0.01, 0.1, 1, 10, 100],
    "model__penalty": ["l1", "l2"],
    "model__solver": ["liblinear"],
}


def build_preprocessor(numeric_features=None, categorical_features=None) -> ColumnTransformer:
    """One-hot encode categoricals, scale numerics."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features or config.NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="if_binary"),
                categorical_features or config.CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_models() -> dict:
    return {
        "Logistic Regression (baseline)": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=config.RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=16, n_jobs=-1, random_state=config.RANDOM_STATE
        ),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=config.RANDOM_STATE),
    }


def _pipeline(model, numeric_features=None, categorical_features=None) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", build_preprocessor(numeric_features, categorical_features)),
            ("model", model),
        ]
    )


def tune_logistic_regression(X_train, y_train) -> tuple[Pipeline, dict]:
    """GridSearchCV over regularization (notebook section 11.3)."""
    grid = GridSearchCV(
        _pipeline(
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=config.RANDOM_STATE)
        ),
        param_grid=PARAM_GRID,
        scoring="f1",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE),
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    best_params = {k.removeprefix("model__"): v for k, v in grid.best_params_.items()}
    return grid.best_estimator_, {"best_params": best_params, "cv_f1": round(float(grid.best_score_), 4)}


def demographics_ablation(tuned_params: dict, X_train, y_train, X_test, y_test) -> dict:
    """Refit the tuned model without demographic features (notebook 13.1)."""
    numeric = [f for f in config.NUMERIC_FEATURES if f not in config.DEMOGRAPHIC_FEATURES]
    categorical = [f for f in config.CATEGORICAL_FEATURES if f not in config.DEMOGRAPHIC_FEATURES]
    model = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=config.RANDOM_STATE, **tuned_params
    )
    pipeline = _pipeline(model, numeric, categorical)
    pipeline.fit(X_train, y_train)
    metrics = evaluate_model(pipeline, X_test, y_test)
    metrics.pop("confusion_matrix")
    metrics.pop("roc_curve")
    return metrics


def train(data_path: Path = config.DATA_PATH, out_dir: Path = config.MODEL_DIR) -> dict:
    df = load_dataset(data_path)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )

    results: dict[str, dict] = {}
    fitted: dict[str, Pipeline] = {}

    for name, model in build_models().items():
        pipeline = _pipeline(model)
        pipeline.fit(X_train, y_train)
        results[name] = evaluate_model(pipeline, X_test, y_test)
        fitted[name] = pipeline
        print(f"{name:32s} ROC-AUC {results[name]['roc_auc']:.4f}  F1 {results[name]['f1']:.4f}")

    tuned, tuning = tune_logistic_regression(X_train, y_train)
    results[TUNED_LR] = evaluate_model(tuned, X_test, y_test)
    fitted[TUNED_LR] = tuned
    print(
        f"{TUNED_LR:32s} ROC-AUC {results[TUNED_LR]['roc_auc']:.4f}  F1 {results[TUNED_LR]['f1']:.4f}"
        f"  (best params {tuning['best_params']}, CV F1 {tuning['cv_f1']})"
    )

    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    print(f"\nBest model: {best_name}")

    ablation = demographics_ablation(tuning["best_params"], X_train, y_train, X_test, y_test)
    fairness = {
        name: fairness_report(fitted[name], X_test, y_test)
        for name in {TUNED_LR, best_name}
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[best_name], out_dir / "model.joblib")

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "best_model": best_name,
        "tuning": tuning,
        "models": results,
        "demographics_ablation": {
            "with": {
                k: results[TUNED_LR][k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]
            },
            "without": ablation,
        },
        "fairness": fairness,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved {out_dir / 'model.joblib'} and {out_dir / 'metrics.json'}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=config.DATA_PATH, help="Path to loan_data.csv")
    parser.add_argument("--out", type=Path, default=config.MODEL_DIR, help="Output directory for artifacts")
    args = parser.parse_args()
    train(args.data, args.out)


if __name__ == "__main__":
    main()
