"""Smoke tests for the data module, training pipeline, and saved artifact."""

import joblib
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from loan_approval import config
from loan_approval.data import load_dataset, split_features_target
from loan_approval.evaluate import evaluate_model
from loan_approval.train import build_models, build_preprocessor


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return load_dataset()


def test_dataset_schema_and_cleaning(df):
    assert set(config.ALL_FEATURES) | {config.TARGET} <= set(df.columns)
    assert df.isna().sum().sum() == 0
    assert df["person_age"].max() <= config.MAX_AGE
    assert df["person_emp_exp"].max() <= config.MAX_EMP_EXP
    assert set(df[config.TARGET].unique()) == {0, 1}
    assert len(df) > 40_000


def test_category_options_match_data(df):
    for col, options in config.CATEGORY_OPTIONS.items():
        assert set(df[col].unique()) == set(options), col


def test_training_smoke(df):
    """Train the fastest model on a subsample; sanity-check its metrics."""
    X, y = split_features_target(df.sample(5_000, random_state=0))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0
    )
    model = build_models()["Logistic Regression"]
    pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", model)])
    pipeline.fit(X_train, y_train)
    metrics = evaluate_model(pipeline, X_test, y_test)
    assert metrics["roc_auc"] > 0.85
    assert 0 < metrics["f1"] <= 1


@pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="run `python -m loan_approval.train` first")
def test_saved_artifact_predicts(df):
    model = joblib.load(config.MODEL_PATH)
    sample = df[config.ALL_FEATURES].head(5)
    proba = model.predict_proba(sample)[:, 1]
    assert proba.shape == (5,)
    assert ((proba >= 0) & (proba <= 1)).all()
