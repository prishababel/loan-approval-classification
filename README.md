# 🏦 Loan Approval Classification — Beyond the Credit Score

End-to-end machine-learning project that predicts whether a loan application
will be **approved or rejected**, trained on 45,000 applications, with an
interactive **Streamlit** web app for live predictions and a fairness
analysis.

Restructured from the team notebook
[prishababel/loan-approval-classification](https://www.kaggle.com/code/prishababel/loan-approval-classification)
(*"Beyond the Credit Score"* — Prisha Babel, Chad Rampersad, Dedeepya
Pidaparthi, Brocatto200, Nishat, Samuel Oyekan; Apache 2.0) into a clean,
tested, deployable Python project — same methodology, packaged as a single
scikit-learn pipeline, plus an implementation of the fairness analysis the
notebook scopes out.

## Methodology (from the team notebook)

- **Cleaning:** drop implausible rows — age > 110, income > $1M, employment
  experience ≥ age. No missing values or duplicates in the raw data.
- **No leakage:** `loan_int_rate` is excluded from the features — the
  interest rate is set *after* a loan is approved, so using it to predict
  approval would leak the answer.
- **Preprocessing:** standardize the 7 numeric features, one-hot encode the
  5 categoricals (binary columns kept as single 0/1 columns), fit on the
  training split only. Stratified 80/20 split, seed 42.
- **Class imbalance:** only 22% of applications are approved, so the
  Logistic Regression models use `class_weight="balanced"` and evaluation
  leans on ROC-AUC / F1 rather than accuracy.
- **Models:** class-weighted Logistic Regression baseline, a
  GridSearchCV-tuned Logistic Regression (regularization strength + L1/L2,
  scored on F1), and Random Forest / Hist Gradient Boosting as the
  validation-and-comparison extension.

## Results

Stratified 20% hold-out set (8,995 loans):

| Model | ROC-AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **Hist Gradient Boosting** (selected) | **0.9583** | 0.9002 | 0.8264 | 0.6975 | 0.7565 |
| Random Forest | 0.9504 | 0.8938 | 0.8636 | 0.6205 | 0.7221 |
| Logistic Regression (baseline) | 0.9310 | 0.8107 | 0.5443 | 0.9115 | 0.6816 |
| Logistic Regression (tuned) | 0.9310 | 0.8103 | 0.5439 | 0.9105 | 0.6810 |

The baseline reproduces the team notebook's reported numbers (ROC-AUC 0.931,
F1 ≈ 0.68, recall ≈ 0.91 — the balanced class weights deliberately trade
precision for recall on the approved class). The best pipeline by ROC-AUC is
saved to `models/model.joblib`; the full report — including per-model ROC
curves, the fairness report, and the demographics ablation — goes to
`models/metrics.json`. Both are committed so the app runs without a training
step.

### Fairness (notebook section 12, implemented here)

Group metrics on the hold-out set for the saved model:

- **Gender:** near-parity — demographic-parity gap 1.4pp, equal-opportunity
  gap 0.3pp.
- **Age bands:** larger spread — equal-opportunity gap 11.5pp (approvable
  under-25 applicants are approved 76% of the time vs. 64–68% for older
  bands).
- **Ablation:** retraining the tuned model *without* gender/age/education
  changes every metric by ≤ 0.001 — decisions are driven by financial
  features, dominated by `previous_loan_defaults_on_file` (no applicant with
  a prior default is approved in this dataset).

## Dataset

[Loan Approval Classification Dataset](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data)
(Kaggle, taweilo) — 45,000 rows × 14 columns of synthetic credit-risk data
(SMOTENC-enhanced from the original Credit Risk dataset). A copy ships in
`data/loan_data.csv`. Target: `loan_status` — 1 = approved, 0 = rejected
(22% approved).

The notebook's cross-dataset consistency check (section 13.2) uses the
original [Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset);
it is not redistributed here — download `credit_risk_dataset.csv` from
Kaggle into `data/` and the walkthrough notebook will pick it up.

## Project structure

```
├── streamlit_app.py            # Streamlit app (predict, metrics, fairness, explorer)
├── loan_approval/              # Python package
│   ├── config.py               #   paths, feature schema, cleaning bounds
│   ├── data.py                 #   loading + cleaning
│   ├── train.py                #   train, tune, compare, ablate, save best
│   ├── fairness.py             #   group metrics by gender / age band
│   └── evaluate.py             #   metrics helpers
├── data/loan_data.csv          # dataset (45,000 rows)
├── models/                     # trained pipeline + metrics report
├── notebooks/                  # walkthrough mirroring the team notebook
├── tests/                      # pytest smoke tests
└── requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt

# retrain (optional — a trained model is already committed)
python -m loan_approval.train

# run the app
streamlit run streamlit_app.py
```

Then open http://localhost:8501. Run the tests with `pytest`.

## The Streamlit app

**Streamlit** ([streamlit.io](https://streamlit.io)) is an open-source Python
framework that turns scripts into shareable web apps — no frontend code
required. The app has four tabs:

- **🔮 Predict** — enter an application (income, loan amount, credit score, …)
  and get the approval probability. No interest-rate input, by design (see
  leakage note above).
- **📊 Model performance** — comparison table, ROC curves, confusion matrix,
  and the with/without-demographics ablation.
- **⚖️ Fairness** — per-group approval rates, TPR/FPR, and disparity gaps by
  gender and age band.
- **🗂️ Data explorer** — dataset stats, feature distributions, and approval
  rates by category.

### Deploy to Streamlit Community Cloud (free)

1. Push this repo to GitHub.
2. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
3. Click **Create app → Deploy a public app from GitHub**, pick this repo and
   branch, and set the main file to `streamlit_app.py`.
4. In **Advanced settings**, choose **Python 3.12** (matches the version the
   committed model was trained with, per `scikit-learn==1.5.1` in
   `requirements.txt`).
5. Click **Deploy** — the app builds from `requirements.txt` and goes live at
   a public `*.streamlit.app` URL in a few minutes.

## Team

- Dedeepya Pidaparthi
- Prisha Babel
- Nishat
- Erwin
- Samuel

## Acknowledgements

- Original team notebook: [*Beyond the Credit Score*](https://www.kaggle.com/code/prishababel/loan-approval-classification)
  by Prisha Babel, Chad Rampersad, Dedeepya Pidaparthi, Brocatto200, Nishat,
  and Samuel Oyekan (Apache 2.0).
- Dataset: [taweilo on Kaggle](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data).

*Educational project — not financial advice.*
