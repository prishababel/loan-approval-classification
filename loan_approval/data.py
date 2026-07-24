"""Data loading and cleaning."""

from pathlib import Path

import pandas as pd

from . import config


def load_dataset(path: str | Path = config.DATA_PATH) -> pd.DataFrame:
    """Load the loan dataset, validate its schema, and clean outliers.

    Cleaning is intentionally light: the dataset ships complete (no missing
    values), so only rows with implausible ages or employment experience are
    dropped.
    """
    df = pd.read_csv(path)

    expected = set(config.ALL_FEATURES) | {config.TARGET}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")

    df = df[df["person_age"] <= config.MAX_AGE]
    df = df[df["person_emp_exp"] <= config.MAX_EMP_EXP]
    return df.reset_index(drop=True)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) with features in the canonical column order."""
    return df[config.ALL_FEATURES], df[config.TARGET]
