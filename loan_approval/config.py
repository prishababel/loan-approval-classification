"""Central configuration: paths, feature schema, and cleaning bounds."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = REPO_ROOT / "data" / "loan_data.csv"
MODEL_DIR = REPO_ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

TARGET = "loan_status"
TARGET_LABELS = {0: "Rejected", 1: "Approved"}

NUMERIC_FEATURES = [
    "person_age",
    "person_income",
    "person_emp_exp",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
]

CATEGORICAL_FEATURES = [
    "person_gender",
    "person_education",
    "person_home_ownership",
    "loan_intent",
    "previous_loan_defaults_on_file",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Options shown in the Streamlit form (fixed order = fixed display order).
CATEGORY_OPTIONS = {
    "person_gender": ["female", "male"],
    "person_education": ["High School", "Associate", "Bachelor", "Master", "Doctorate"],
    "person_home_ownership": ["RENT", "MORTGAGE", "OWN", "OTHER"],
    "loan_intent": [
        "PERSONAL",
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "HOMEIMPROVEMENT",
        "DEBTCONSOLIDATION",
    ],
    "previous_loan_defaults_on_file": ["No", "Yes"],
}

# Cleaning bounds — the raw file contains a handful of implausible rows
# (ages up to 144, employment experience up to 125 years).
MAX_AGE = 90
MAX_EMP_EXP = 60

RANDOM_STATE = 42
TEST_SIZE = 0.2
