"""Train and compare loan-approval classifiers, then save the best one.

Usage:
    python -m loan_approval.train [--data PATH] [--out DIR]

Trains Logistic Regression, Random Forest, and Histogram Gradient Boosting
inside a shared preprocessing pipeline, compares them on a stratified
hold-out set, and writes the best pipeline (by ROC-AUC) to models/model.joblib
plus a metrics report to models/metrics.json.
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .data import load_dataset, split_features_target
from .evaluate import evaluate_model


def build_preprocessor() -> ColumnTransformer:
    """One-hot encode categoricals, scale numerics."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), config.NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                config.CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=config.RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=16, n_jobs=-1, random_state=config.RANDOM_STATE
        ),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=config.RANDOM_STATE),
    }


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
        pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", model)])
        pipeline.fit(X_train, y_train)
        results[name] = evaluate_model(pipeline, X_test, y_test)
        fitted[name] = pipeline
        print(f"{name:24s} ROC-AUC {results[name]['roc_auc']:.4f}  F1 {results[name]['f1']:.4f}")

    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    print(f"\nBest model: {best_name}")

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[best_name], out_dir / "model.joblib")

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "best_model": best_name,
        "models": results,
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
