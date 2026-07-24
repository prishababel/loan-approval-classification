# 🏦 Loan Approval Classification

End-to-end machine-learning project that predicts whether a loan application
will be **approved or rejected**, trained on 45,000 applications, with an
interactive **Streamlit** web app for live predictions.

Based on the Kaggle notebook
[prishababel/loan-approval-classification](https://www.kaggle.com/code/prishababel/loan-approval-classification),
restructured into a clean, tested, deployable Python project.

## Results

Three classifiers are trained inside a shared preprocessing pipeline and
compared on a stratified 20% hold-out set (8,998 loans):

| Model | ROC-AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **Hist Gradient Boosting** (selected) | **0.9771** | 0.9320 | 0.8881 | 0.7940 | 0.8384 |
| Random Forest | 0.9736 | 0.9275 | 0.9002 | 0.7580 | 0.8230 |
| Logistic Regression | 0.9540 | 0.8976 | 0.7820 | 0.7480 | 0.7646 |

The best pipeline (by ROC-AUC) is saved to `models/model.joblib` and the full
report to `models/metrics.json` — both are committed so the app runs without a
training step.

## Dataset

[Loan Approval Classification Dataset](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data)
(Kaggle, taweilo) — 45,000 rows × 14 columns of synthetic credit-risk data
(SMOTENC-enhanced from the original Credit Risk dataset). A copy ships in
`data/loan_data.csv`.

**Features:** age, gender, education, income, employment experience, home
ownership, loan amount, loan intent, interest rate, loan-to-income ratio,
credit-history length, credit score, previous defaults on file.
**Target:** `loan_status` — 1 = approved, 0 = rejected (22% approved).

Cleaning drops a handful of implausible rows (age > 90, employment
experience > 60 years); the data has no missing values or duplicates.

## Project structure

```
├── streamlit_app.py            # Streamlit app (predict, metrics, explorer)
├── loan_approval/              # Python package
│   ├── config.py               #   paths, feature schema, constants
│   ├── data.py                 #   loading + cleaning
│   ├── train.py                #   train, compare, save best model
│   └── evaluate.py             #   metrics helpers
├── data/loan_data.csv          # dataset (45,000 rows)
├── models/                     # trained pipeline + metrics report
├── notebooks/                  # EDA + model-comparison walkthrough
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
required. The app has three tabs:

- **🔮 Predict** — enter an application (income, loan amount, credit score, …)
  and get the approval probability from the trained model.
- **📊 Model performance** — comparison table, ROC curves, and the confusion
  matrix from the held-out test set.
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

## Acknowledgements

- Original notebook: [prishababel on Kaggle](https://www.kaggle.com/code/prishababel/loan-approval-classification)
- Dataset: [taweilo on Kaggle](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data)

*Educational project — not financial advice.*
