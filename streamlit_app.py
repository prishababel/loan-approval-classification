"""Loan Approval Predictor — Streamlit app.

Run locally:
    streamlit run streamlit_app.py

The app loads the trained pipeline from models/model.joblib (produced by
`python -m loan_approval.train`) and serves four views: an interactive
prediction form, the model comparison report, the fairness analysis, and a
dataset explorer.
"""

import sys
from pathlib import Path

import altair as alt
import joblib
import json
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from loan_approval import config  # noqa: E402

# Validated chart palette (see README): categorical slots + status colors.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"
SERIES = [BLUE, ORANGE, AQUA, YELLOW]

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(config.MODEL_PATH)


@st.cache_data
def load_metrics() -> dict:
    with open(config.METRICS_PATH) as f:
        return json.load(f)


@st.cache_data
def load_data() -> pd.DataFrame:
    from loan_approval.data import load_dataset

    return load_dataset()


def predict_tab(model, metrics: dict) -> None:
    st.subheader("Check a loan application")
    st.caption(
        f"Predictions come from the best model in training "
        f"({metrics['best_model']}, test ROC-AUC {metrics['models'][metrics['best_model']]['roc_auc']:.3f}). "
        "The interest rate is deliberately not an input — it is set after the "
        "approval decision, so using it would leak the answer."
    )

    with st.form("application"):
        person_col, loan_col = st.columns(2)

        with person_col:
            st.markdown("**Applicant**")
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            gender = st.selectbox("Gender", config.CATEGORY_OPTIONS["person_gender"])
            education = st.selectbox("Education", config.CATEGORY_OPTIONS["person_education"], index=2)
            income = st.number_input("Annual income ($)", min_value=1_000, max_value=1_000_000, value=60_000, step=1_000)
            emp_exp = st.number_input("Employment experience (years)", min_value=0, max_value=80, value=5)
            home = st.selectbox("Home ownership", config.CATEGORY_OPTIONS["person_home_ownership"])

        with loan_col:
            st.markdown("**Loan**")
            amount = st.number_input("Loan amount ($)", min_value=500, max_value=100_000, value=10_000, step=500)
            intent = st.selectbox("Loan intent", config.CATEGORY_OPTIONS["loan_intent"])
            credit_score = st.number_input("Credit score", min_value=300, max_value=850, value=630)
            hist_len = st.number_input("Credit history length (years)", min_value=0, max_value=40, value=4)
            defaults = st.selectbox(
                "Previous loan defaults on file",
                config.CATEGORY_OPTIONS["previous_loan_defaults_on_file"],
            )

        submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

    if not submitted:
        st.info("Fill in the application and press **Predict**.")
        return

    percent_income = round(amount / income, 4)
    row = pd.DataFrame(
        [
            {
                "person_age": float(age),
                "person_income": float(income),
                "person_emp_exp": int(emp_exp),
                "loan_amnt": float(amount),
                "loan_percent_income": percent_income,
                "cb_person_cred_hist_length": float(hist_len),
                "credit_score": int(credit_score),
                "person_gender": gender,
                "person_education": education,
                "person_home_ownership": home,
                "loan_intent": intent,
                "previous_loan_defaults_on_file": defaults,
            }
        ]
    )[config.ALL_FEATURES]

    proba = float(model.predict_proba(row)[0, 1])
    approved = proba >= 0.5

    result_col, detail_col = st.columns([1, 2])
    with result_col:
        st.metric("Approval probability", f"{proba:.1%}")
        if approved:
            st.success("✅ Likely **approved**")
        else:
            st.error("❌ Likely **rejected**")
    with detail_col:
        st.progress(proba, text=f"Model confidence that this loan is approved: {proba:.1%}")
        st.caption(
            f"Loan is {percent_income:.1%} of annual income. "
            "Verdict uses a 50% threshold — treat borderline scores as manual-review cases."
        )


def performance_tab(metrics: dict) -> None:
    st.subheader("Model comparison")
    best = metrics["best_model"]
    best_metrics = metrics["models"][best]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best model", best)
    m2.metric("ROC-AUC", f"{best_metrics['roc_auc']:.4f}")
    m3.metric("Accuracy", f"{best_metrics['accuracy']:.4f}")
    m4.metric("F1 score", f"{best_metrics['f1']:.4f}")
    tuning = metrics["tuning"]
    st.caption(
        f"Hold-out test set: {metrics['n_test']:,} loans ({metrics['n_train']:,} used for training). "
        f"Tuned Logistic Regression grid search picked {tuning['best_params']} "
        f"(cross-validated F1 {tuning['cv_f1']:.3f}). "
        f"Trained {metrics['trained_at']} · scikit-learn {metrics['sklearn_version']}."
    )

    table = pd.DataFrame(
        {
            name: {
                "ROC-AUC": m["roc_auc"],
                "Accuracy": m["accuracy"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "F1": m["f1"],
            }
            for name, m in metrics["models"].items()
        }
    ).T.sort_values("ROC-AUC", ascending=False)
    st.dataframe(table.style.format("{:.4f}"), use_container_width=True)

    chart_col, cm_col = st.columns([3, 2])

    with chart_col:
        st.markdown("**ROC curves (test set)**")
        roc_frames = []
        for name, m in metrics["models"].items():
            roc_frames.append(
                pd.DataFrame(
                    {
                        "False positive rate": m["roc_curve"]["fpr"],
                        "True positive rate": m["roc_curve"]["tpr"],
                        "Model": name,
                    }
                )
            )
        roc_df = pd.concat(roc_frames, ignore_index=True)
        model_order = list(metrics["models"].keys())

        diagonal = (
            alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
            .mark_line(strokeDash=[4, 4], color="#898781", strokeWidth=1)
            .encode(x="x", y="y")
        )
        roc = (
            alt.Chart(roc_df)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("False positive rate", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("True positive rate", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color(
                    "Model",
                    scale=alt.Scale(domain=model_order, range=SERIES[: len(model_order)]),
                    legend=alt.Legend(orient="bottom", columns=2),
                ),
                tooltip=["Model", "False positive rate", "True positive rate"],
            )
        )
        st.altair_chart(diagonal + roc, use_container_width=True)

    with cm_col:
        st.markdown(f"**Confusion matrix — {best}**")
        cm = best_metrics["confusion_matrix"]
        cm_df = pd.DataFrame(
            cm,
            index=["Actually rejected", "Actually approved"],
            columns=["Predicted rejected", "Predicted approved"],
        )
        st.dataframe(cm_df.style.format("{:,}"), use_container_width=True)
        st.caption("Rows are the true outcome; columns are the model's prediction.")

        ablation = metrics["demographics_ablation"]
        st.markdown("**Do demographics matter? (ablation)**")
        ab_df = pd.DataFrame(
            {
                "With demographics": ablation["with"],
                "Without demographics": {k: ablation["without"][k] for k in ablation["with"]},
            }
        )
        ab_df["Difference"] = ab_df["Without demographics"] - ab_df["With demographics"]
        st.dataframe(ab_df.style.format("{:+.4f}", subset=["Difference"]).format("{:.4f}", subset=ab_df.columns[:2]), use_container_width=True)
        st.caption(
            "Tuned Logistic Regression retrained without gender, age, or education: "
            "metrics barely move, so approval decisions are driven by financial features."
        )


def fairness_tab(metrics: dict) -> None:
    st.subheader("Fairness analysis")
    model_names = list(metrics["fairness"].keys())
    model_name = model_names[0] if len(model_names) == 1 else st.radio(
        "Model", model_names, horizontal=True
    )
    report = metrics["fairness"][model_name]

    st.caption(
        "Group-level behavior on the hold-out test set. Demographic-parity gap = spread in "
        "predicted approval rates between groups; equal-opportunity gap = spread in true-positive "
        "rates (how often genuinely approvable applicants are approved)."
    )

    for dim, title in [("person_gender", "By gender"), ("age_band", "By age band")]:
        block = report[dim]
        st.markdown(f"**{title}**")

        rows = pd.DataFrame(block["groups"]).T
        rows.index.name = "Group"
        rows = rows.rename(
            columns={
                "n": "Applicants",
                "actual_approval_rate": "Actual approval",
                "predicted_approval_rate": "Predicted approval",
                "accuracy": "Accuracy",
                "tpr": "TPR",
                "fpr": "FPR",
                "precision": "Precision",
            }
        )
        if dim == "age_band":
            order = [b[0] for b in config.AGE_BANDS if b[0] in rows.index]
            rows = rows.loc[order]

        table_col, chart_col = st.columns([3, 2])
        with table_col:
            pct_cols = [c for c in rows.columns if c != "Applicants"]
            st.dataframe(
                rows.style.format("{:.1%}", subset=pct_cols, na_rep="—").format("{:,.0f}", subset=["Applicants"]),
                use_container_width=True,
            )
            g1, g2 = st.columns(2)
            g1.metric("Demographic-parity gap", f"{block['demographic_parity_gap']:.1%}")
            g2.metric("Equal-opportunity gap", f"{block['equal_opportunity_gap']:.1%}")
        with chart_col:
            chart_df = rows.reset_index()[["Group", "Actual approval", "Predicted approval"]].melt(
                id_vars="Group", var_name="Rate", value_name="value"
            )
            chart = (
                alt.Chart(chart_df)
                .mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
                .encode(
                    x=alt.X("Group", sort=list(rows.index), title=None),
                    xOffset="Rate",
                    y=alt.Y("value", axis=alt.Axis(format="%"), title="Approval rate"),
                    color=alt.Color(
                        "Rate",
                        scale=alt.Scale(domain=["Actual approval", "Predicted approval"], range=[BLUE, ORANGE]),
                        legend=alt.Legend(orient="bottom"),
                    ),
                    tooltip=["Group", "Rate", alt.Tooltip("value", format=".1%")],
                )
            )
            st.altair_chart(chart, use_container_width=True)

    st.info(
        "The dominant predictor is **previous loan defaults on file** — no applicant with a prior "
        "default is approved in this dataset. Demographic features carry almost no weight (see the "
        "ablation in Model performance), which is what the small gaps above reflect."
    )


def explorer_tab(df: pd.DataFrame) -> None:
    st.subheader("Dataset explorer")

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Applications", f"{len(df):,}")
    d2.metric("Approval rate", f"{df[config.TARGET].mean():.1%}")
    d3.metric("Median income", f"${df['person_income'].median():,.0f}")
    d4.metric("Median loan", f"${df['loan_amnt'].median():,.0f}")

    outcome = df[config.TARGET].map(config.TARGET_LABELS)

    hist_col, rate_col = st.columns(2)

    with hist_col:
        feature = st.selectbox("Distribution of…", config.NUMERIC_FEATURES, index=6)
        split = st.checkbox("Split by outcome", value=False)
        base = df[[feature]].assign(Outcome=outcome)
        if split:
            chart = (
                alt.Chart(base)
                .mark_bar(opacity=0.75, cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
                .encode(
                    x=alt.X(feature, bin=alt.Bin(maxbins=40)),
                    y=alt.Y("count()", stack=None, title="Applications"),
                    color=alt.Color(
                        "Outcome",
                        scale=alt.Scale(domain=["Approved", "Rejected"], range=[BLUE, ORANGE]),
                        legend=alt.Legend(orient="bottom"),
                    ),
                    tooltip=["Outcome", alt.Tooltip("count()", title="Applications")],
                )
            )
        else:
            chart = (
                alt.Chart(base)
                .mark_bar(color=BLUE, cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
                .encode(
                    x=alt.X(feature, bin=alt.Bin(maxbins=40)),
                    y=alt.Y("count()", title="Applications"),
                    tooltip=[alt.Tooltip("count()", title="Applications")],
                )
            )
        st.altair_chart(chart, use_container_width=True)

    with rate_col:
        cat = st.selectbox("Approval rate by…", config.CATEGORICAL_FEATURES, index=3)
        rate_df = (
            df.groupby(cat)[config.TARGET].mean().rename("Approval rate").reset_index()
        )
        rate_chart = (
            alt.Chart(rate_df)
            .mark_bar(color=BLUE, cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
            .encode(
                x=alt.X("Approval rate", axis=alt.Axis(format="%")),
                y=alt.Y(cat, sort="-x"),
                tooltip=[cat, alt.Tooltip("Approval rate", format=".1%")],
            )
        )
        st.altair_chart(rate_chart, use_container_width=True)

    with st.expander("Preview raw data"):
        st.dataframe(df.head(100), use_container_width=True)
        st.caption(
            "Source: Kaggle — “Loan Approval Classification Dataset” (taweilo). "
            "Synthetic credit-risk data; loan_status 1 = approved, 0 = rejected."
        )


def main() -> None:
    st.title("🏦 Loan Approval Predictor")
    st.caption(
        "Machine-learning demo trained on 45,000 loan applications, based on the team notebook "
        "“Beyond the Credit Score” (Prisha Babel, Chad Rampersad, Dedeepya Pidaparthi, Brocatto200, "
        "Nishat, Samuel Oyekan). Educational project — not financial advice."
    )

    if not config.MODEL_PATH.exists() or not config.METRICS_PATH.exists():
        st.error(
            "Model artifacts not found. Train first:\n\n"
            "```\npython -m loan_approval.train\n```"
        )
        st.stop()

    model = load_model()
    metrics = load_metrics()
    df = load_data()

    tab_predict, tab_perf, tab_fair, tab_data = st.tabs(
        ["🔮 Predict", "📊 Model performance", "⚖️ Fairness", "🗂️ Data explorer"]
    )
    with tab_predict:
        predict_tab(model, metrics)
    with tab_perf:
        performance_tab(metrics)
    with tab_fair:
        fairness_tab(metrics)
    with tab_data:
        explorer_tab(df)


if __name__ == "__main__":
    main()
