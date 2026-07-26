"""Smoke tests for the data module, training pipeline, and saved artifact."""

import joblib
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from loan_approval import config
from loan_approval.data import age_band, load_dataset, split_features_target
from loan_approval.evaluate import evaluate_model
from loan_approval.fairness import fairness_report
from loan_approval.train import _pipeline, build_models


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return load_dataset()


def test_dataset_schema_and_cleaning(df):
    assert set(config.RAW_COLUMNS) <= set(df.columns)
    assert df.isna().sum().sum() == 0
    assert df["person_age"].max() <= config.MAX_AGE
    assert df["person_income"].max() <= config.MAX_INCOME
    assert (df["person_emp_exp"] < df["person_age"]).all()
    assert set(df[config.TARGET].unique()) == {0, 1}
    assert len(df) > 40_000


def test_leaky_feature_excluded():
    assert "loan_int_rate" not in config.ALL_FEATURES


def test_category_options_match_data(df):
    for col, options in config.CATEGORY_OPTIONS.items():
        assert set(df[col].unique()) == set(options), col


def test_age_band(df):
    bands = age_band(df["person_age"])
    assert bands.isna().sum() == 0
    assert set(bands.unique()) <= {b[0] for b in config.AGE_BANDS}


def test_training_smoke_and_fairness(df):
    """Train the baseline model on a subsample; sanity-check metrics + fairness."""
    X, y = split_features_target(df.sample(5_000, random_state=0))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0
    )
    pipeline = _pipeline(build_models()["Logistic Regression (baseline)"])
    pipeline.fit(X_train, y_train)

    metrics = evaluate_model(pipeline, X_test, y_test)
    assert metrics["roc_auc"] > 0.85
    assert 0 < metrics["f1"] <= 1

    report = fairness_report(pipeline, X_test, y_test)
    assert set(report) == {"person_gender", "age_band"}
    for dim in report.values():
        assert dim["groups"]
        assert 0 <= dim["demographic_parity_gap"] <= 1


@pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="run `python -m loan_approval.train` first")
def test_saved_artifact_predicts(df):
    model = joblib.load(config.MODEL_PATH)
    sample = df[config.ALL_FEATURES].head(5)
    proba = model.predict_proba(sample)[:, 1]
    assert proba.shape == (5,)
    assert ((proba >= 0) & (proba <= 1)).all()
