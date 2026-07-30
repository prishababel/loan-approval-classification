"""Data loading and cleaning (notebook sections 2-4)."""

from pathlib import Path

import pandas as pd

from . import config


def load_dataset(path: str | Path = config.DATA_PATH) -> pd.DataFrame:
    """Load the loan dataset, validate its schema, and clean bad rows.

    Cleaning rules from the team notebook:
    - drop ages above 110 (the raw file tops out at 144),
    - drop incomes above $1,000,000,
    - drop rows where employment experience >= age (impossible).

    The dataset has no missing values or duplicates, so nothing else is
    touched. loan_int_rate stays in the frame for exploration but is not a
    model feature (see config).
    """
    df = pd.read_csv(path)

    missing = set(config.RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")

    df = df[df["person_age"] <= config.MAX_AGE]
    df = df[df["person_income"] <= config.MAX_INCOME]
    df = df[df["person_emp_exp"] < df["person_age"]]
    return df.reset_index(drop=True)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) with features in the canonical column order."""
    return df[config.ALL_FEATURES], df[config.TARGET]


def age_band(ages: pd.Series) -> pd.Series:
    """Map ages to the fairness-analysis age bands."""
    bins = [b[1] - 0.5 for b in config.AGE_BANDS] + [config.AGE_BANDS[-1][2]]
    labels = [b[0] for b in config.AGE_BANDS]
    return pd.cut(ages, bins=bins, labels=labels)
