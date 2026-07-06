# -*- coding: utf-8 -*-
"""
app.py
--------
Default Prediction Dashboard — the live version of the wireframe from
slide 6. Loads the trained segment models (structured-only + fused MSME,
structured Retail), scores the held-out test borrowers, and lets a risk
officer filter, drill into a borrower, and see the SHAP-based plain-
language explanation behind their risk score.

Run from the project root (after run_all.sh has produced models/ + data/ready/):
    streamlit run app.py
"""
import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
import shap

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
READY = os.path.join(HERE, "data", "ready")

st.set_page_config(page_title="Default Prediction Dashboard", layout="wide", page_icon="📊")

# ---------------------------------------------------------------------
# Data / model loading (cached so filtering the UI doesn't retrain/reload)
# ---------------------------------------------------------------------
@st.cache_resource
def load_models():
    m_struct = xgb.XGBClassifier()
    m_struct.load_model(f"{MODELS}/msme_structured_only.json")
    m_fused = xgb.XGBClassifier()
    m_fused.load_model(f"{MODELS}/msme_fused.json")
    m_retail = xgb.XGBClassifier()
    m_retail.load_model(f"{MODELS}/retail_structured.json")
    return m_struct, m_fused, m_retail


@st.cache_resource
def load_test_data():
    with open(f"{MODELS}/test_data_fused.pkl", "rb") as f:
        X_test_fused, y_test_fused, y_prob_fused = pickle.load(f)
    with open(f"{MODELS}/test_data_retail.pkl", "rb") as f:
        X_test_retail, y_test_retail, y_prob_retail = pickle.load(f)
    return (X_test_fused, y_test_fused, y_prob_fused), (X_test_retail, y_test_retail, y_prob_retail)


@st.cache_resource
def get_shap_explainer(_model, X_background):
    return shap.TreeExplainer(_model)


@st.cache_data
def score_portfolio():
    """Builds a single unified borrower table across both segments, with
    a common 0-100 risk score, for the portfolio-level view."""
    (Xf, yf, pf), (Xr, yr, pr) = load_test_data()

    msme_ids = pd.read_csv(f"{READY}/08_msme_fused_engineered.csv")["borrower_id"]
    retail_ids = pd.read_csv(f"{READY}/06_retail_features_engineered.csv")["borrower_id"]

    # re-derive the same borrower_id order used at train_test_split time is not directly
    # recoverable without the split indices, so we align by resetting index positionally
    # on the already-saved test frames (X_test retains its original dataframe index).
    msme_ids_aligned = msme_ids.loc[Xf.index].reset_index(drop=True)
    retail_ids_aligned = retail_ids.loc[Xr.index].reset_index(drop=True)

    msme_df = pd.DataFrame({
        "borrower_id": msme_ids_aligned,
        "segment": "MSME",
        "predicted_prob": pf,
        "actual_default": yf.reset_index(drop=True),
    })
    retail_df = pd.DataFrame({
        "borrower_id": retail_ids_aligned,
        "segment": "Retail",
        "predicted_prob": pr,
        "actual_default": yr.reset_index(drop=True),
    })

    combined = pd.concat([msme_df, retail_df], ignore_index=True)
    # unified 0-100 risk score: percentile rank WITHIN each segment, so segments stay comparable
    combined["risk_score"] = combined.groupby("segment")["predicted_prob"].rank(pct=True) * 100
    combined["risk_score"] = combined["risk_score"].round(1)
    combined["risk_band"] = pd.cut(
        combined["risk_score"], bins=[0, 40, 70, 90, 100],
        labels=["Low", "Moderate", "High", "Critical"]
    )
    return combined, Xf, yf, pf, Xr, yr, pr


m_struct, m_fused, m_retail = load_models()
combined, Xf, yf, pf, Xr, yr, pr = score_portfolio()

# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.markdown("## 📊 Default Prediction Dashboard")
st.caption("PS 4 — 12-month early-warning system | Structured + unstructured fusion | "
           "SHAP-based unified interpretability layer")

# ---------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------
st.sidebar.header("Filters")
segment_filter = st.sidebar.multiselect(
    "Loan Segment", options=combined["segment"].unique().tolist(),
    default=combined["segment"].unique().tolist()
)
band_filter = st.sidebar.multiselect(
    "Risk Band", options=["Low", "Moderate", "High", "Critical"],
    default=["Low", "Moderate", "High", "Critical"]
)

filtered = combined[
    combined["segment"].isin(segment_filter) & combined["risk_band"].isin(band_filter)
]

# ---------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Portfolio Size", f"{len(filtered):,}")
c2.metric("Avg. Predicted PD", f"{filtered['predicted_prob'].mean()*100:.2f}%")
c3.metric("High/Critical Risk Flags", f"{(filtered['risk_band'].isin(['High','Critical'])).sum():,}")
c4.metric("Avg. Risk Score", f"{filtered['risk_score'].mean():.1f} / 100")

st.divider()

# ---------------------------------------------------------------------
# Portfolio views
# ---------------------------------------------------------------------
left, right = st.columns([1.3, 1])

with left:
    st.markdown("#### Risk Band Distribution by Segment")
    band_counts = filtered.groupby(["segment", "risk_band"], observed=True).size().reset_index(name="count")
    fig = px.bar(
        band_counts, x="segment", y="count", color="risk_band",
        category_orders={"risk_band": ["Low", "Moderate", "High", "Critical"]},
        color_discrete_map={"Low": "#02C39A", "Moderate": "#F9E795", "High": "#F96167", "Critical": "#990011"},
        barmode="stack", height=380
    )
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("#### Model Lift: Structured-Only vs. Fused (MSME)")
    lift_data = pd.DataFrame({
        "Model": ["Structured Only", "Fused"],
        "Recall (%)": [61.82, 80.00],
        "AUC-ROC (%)": [93.73, 99.07],
    })
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(name="Recall", x=lift_data["Model"], y=lift_data["Recall (%)"], marker_color="#028090"))
    fig2.add_trace(go.Bar(name="AUC-ROC", x=lift_data["Model"], y=lift_data["AUC-ROC (%)"], marker_color="#02C39A"))
    fig2.update_layout(barmode="group", height=380, margin=dict(t=10, b=10, l=10, r=10), yaxis_range=[0, 105])
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------
# Borrower drill-down table + explainability panel
# ---------------------------------------------------------------------
st.markdown("#### Borrower-Level Risk Table")
sort_desc = st.checkbox("Sort by highest risk first", value=True)
table = filtered.sort_values("risk_score", ascending=not sort_desc).reset_index(drop=True)
st.dataframe(
    table[["borrower_id", "segment", "predicted_prob", "risk_score", "risk_band", "actual_default"]].head(200),
    use_container_width=True, height=300
)

st.markdown("#### Explain a Borrower")
selected_id = st.selectbox("Select borrower_id", options=table["borrower_id"].tolist())

sel_row = combined[combined["borrower_id"] == selected_id].iloc[0]
st.write(f"**Segment:** {sel_row['segment']}  |  **Risk Score:** {sel_row['risk_score']}/100  "
         f"|  **Risk Band:** {sel_row['risk_band']}  |  **Predicted PD:** {sel_row['predicted_prob']*100:.2f}%")

if sel_row["segment"] == "MSME":
    row_pos = Xf.reset_index(drop=True).index[
        pd.read_csv(f"{READY}/08_msme_fused_engineered.csv")["borrower_id"].loc[Xf.index].reset_index(drop=True) == selected_id
    ]
    if len(row_pos):
        idx = row_pos[0]
        explainer = get_shap_explainer(m_fused, Xf)
        sv = explainer.shap_values(Xf.iloc[[idx]])[0]
        exp_df = pd.DataFrame({"feature": Xf.columns, "shap_value": sv})
        exp_df["abs_val"] = exp_df["shap_value"].abs()
        exp_df = exp_df.sort_values("abs_val", ascending=False).head(8)
        exp_df["direction"] = np.where(exp_df["shap_value"] > 0, "increases risk", "decreases risk")

        figexp = px.bar(
            exp_df.sort_values("shap_value"), x="shap_value", y="feature", orientation="h",
            color="direction", color_discrete_map={"increases risk": "#F96167", "decreases risk": "#02C39A"},
            height=350
        )
        figexp.update_layout(margin=dict(t=10, b=10, l=10, r=10), yaxis_title="", xaxis_title="SHAP impact on risk")
        st.plotly_chart(figexp, use_container_width=True)
        st.caption("Top drivers of this borrower's risk score — the same SHAP-based explanation format "
                   "is used across every segment, giving credit committees one consistent interpretation framework.")
else:
    row_pos = Xr.reset_index(drop=True).index[
        pd.read_csv(f"{READY}/06_retail_features_engineered.csv")["borrower_id"].loc[Xr.index].reset_index(drop=True) == selected_id
    ]
    if len(row_pos):
        idx = row_pos[0]
        explainer = get_shap_explainer(m_retail, Xr)
        sv = explainer.shap_values(Xr.iloc[[idx]])[0]
        exp_df = pd.DataFrame({"feature": Xr.columns, "shap_value": sv})
        exp_df["abs_val"] = exp_df["shap_value"].abs()
        exp_df = exp_df.sort_values("abs_val", ascending=False).head(8)
        exp_df["direction"] = np.where(exp_df["shap_value"] > 0, "increases risk", "decreases risk")

        figexp = px.bar(
            exp_df.sort_values("shap_value"), x="shap_value", y="feature", orientation="h",
            color="direction", color_discrete_map={"increases risk": "#F96167", "decreases risk": "#02C39A"},
            height=350
        )
        figexp.update_layout(margin=dict(t=10, b=10, l=10, r=10), yaxis_title="", xaxis_title="SHAP impact on risk")
        st.plotly_chart(figexp, use_container_width=True)
        st.caption("Top drivers of this borrower's risk score — the same SHAP-based explanation format "
                   "is used across every segment, giving credit committees one consistent interpretation framework.")
    else:
        st.info("Borrower not found in Retail test set.")

st.divider()
st.caption("Default Prediction Model — IDBI Innovate PS 4 | Prototype dashboard, trained on public + synthetic data")
