"""Central configuration: paths, feature schema, and cleaning bounds.

The schema and cleaning rules follow the team notebook "Beyond the Credit
Score" (Kaggle: prishababel/loan-approval-classification): ages above 110 and
incomes above $1M are treated as data errors, rows where employment
experience >= age are impossible, and loan_int_rate is excluded from the
features because the interest rate is set *after* the approval decision —
using it to predict approval would leak the answer.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = REPO_ROOT / "data" / "loan_data.csv"
CREDIT_RISK_DATA_PATH = REPO_ROOT / "data" / "credit_risk_dataset.csv"  # optional, see README
MODEL_DIR = REPO_ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

TARGET = "loan_status"
TARGET_LABELS = {0: "Rejected", 1: "Approved"}

# loan_int_rate is deliberately absent (leakage — see module docstring).
NUMERIC_FEATURES = [
    "person_age",
    "person_income",
    "person_emp_exp",
    "loan_amnt",
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

# Raw file schema (features + leaked column + target), used for validation.
RAW_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["loan_int_rate", TARGET]

# Demographic attributes examined in the fairness analysis and excluded in
# the "without demographics" ablation (notebook sections 12–13).
DEMOGRAPHIC_FEATURES = ["person_gender", "person_age", "person_education"]

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

# Cleaning bounds (notebook section 4).
MAX_AGE = 110
MAX_INCOME = 1_000_000

# Age bands for the fairness analysis (notebook section 12).
AGE_BANDS = [
    ("<25", 0, 24),
    ("25-34", 25, 34),
    ("35-44", 35, 44),
    ("45+", 45, 200),
]

RANDOM_STATE = 42
TEST_SIZE = 0.2
