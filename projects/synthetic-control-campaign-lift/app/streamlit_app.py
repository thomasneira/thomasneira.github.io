"""Synthetic Control Campaign Lift Dashboard."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.prepare import (
    load_campaign_data,
    build_daily_series,
    select_treatment_and_donors,
    define_intervention_period,
    build_causal_impact_input,
)
from src.model import fit_synthetic_control, get_pointwise_effects, get_cumulative_effects, run_placebo_tests, format_summary

st.set_page_config(page_title="Campaign Lift — Synthetic Control", layout="wide")
st.title("Synthetic Control for Campaign Lift Measurement")
st.caption("Causal inference when you can't randomize at the individual level")


@st.cache_data
def load_data():
    return load_campaign_data()


try:
    raw = load_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# --- Sidebar ---
st.sidebar.header("Configuration")

numeric_cols = raw.select_dtypes(include="number").columns.tolist()
metric_col = st.sidebar.selectbox("Metric to Analyze", numeric_cols, index=numeric_cols.index("engagement_score") if "engagement_score" in numeric_cols else 0)

unique_companies = sorted(raw["company"].unique().tolist())
treatment = st.sidebar.selectbox("Treatment Company", unique_companies)

# --- Build series ---
try:
    pivot = build_daily_series(raw, "date", metric_col, "company")
    treatment_series, donors = select_treatment_and_donors(pivot, treatment)
    ci_data = build_causal_impact_input(treatment_series, donors)

    pre_start, intervention, post_end = define_intervention_period(treatment_series)

    intervention_date = st.sidebar.date_input(
        "Intervention Start",
        value=intervention.date(),
        min_value=pre_start.date(),
        max_value=post_end.date(),
    )
    pre_period = [str(pre_start.date()), str(intervention_date - pd.Timedelta(days=1))]
    post_period = [str(intervention_date), str(post_end.date())]

except Exception as e:
    st.error(f"Error building time series: {e}")
    st.stop()

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["Causal Impact", "Pointwise Effects", "Placebo Tests"])

with tab1:
    with st.spinner("Fitting synthetic control model..."):
        result = fit_synthetic_control(ci_data, pre_period, post_period)
        summary = result["summary"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Causal Effect", f"{summary['avg_abs_effect']:.3f}")
    col2.metric("Relative Effect", f"{summary['avg_rel_effect']:.1%}")
    col3.metric("Cumulative Effect", f"{summary['cum_abs_effect']:.2f}")
    col4.metric("p-value", f"{summary['p_value']:.4f}")

    st.subheader("Actual vs. Counterfactual")
    inf = result["inferences"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=inf["date"], y=inf["response"], name="Actual", line=dict(color="#2563eb")))
    fig.add_trace(go.Scatter(x=inf["date"], y=inf["preds"], name="Predicted (no campaign)", line=dict(color="#94a3b8", dash="dash")))
    fig.add_vrect(x0=post_period[0], x1=post_period[1], fillcolor="#2563eb", opacity=0.05, line_width=0)
    fig.add_vline(x=post_period[0], line_dash="dot", line_color="#ef4444")
    fig.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=10), xaxis_title="Date", yaxis_title=metric_col)
    st.plotly_chart(fig, use_container_width=True)

    st.code(format_summary(result))

with tab2:
    st.subheader("Pointwise & Cumulative Effects")
    pointwise = get_pointwise_effects(result)
    cumulative = get_cumulative_effects(result)

    post_pw = pointwise[result["inferences"]["is_post"]]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=post_pw["date"], y=post_pw["point_effects"], name="Point Effect", line=dict(color="#2563eb")))
    fig2.add_trace(go.Scatter(x=post_pw["date"], y=post_pw["point_effects_upper"], line=dict(color="#94a3b8", dash="dot"), showlegend=False))
    fig2.add_trace(go.Scatter(x=post_pw["date"], y=post_pw["point_effects_lower"], line=dict(color="#94a3b8", dash="dot"), fill="tonexty", showlegend=False))
    fig2.add_hline(y=0, line_dash="dash", line_color="#64748b")
    fig2.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=10), xaxis_title="Date", yaxis_title="Pointwise Effect")
    st.plotly_chart(fig2, use_container_width=True)

    post_cum = cumulative[result["inferences"]["is_post"]]
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=post_cum["date"], y=post_cum["cum_effects"], name="Cumulative Effect", line=dict(color="#7c3aed")))
    fig3.add_trace(go.Scatter(x=post_cum["date"], y=post_cum["cum_effects_upper"], line=dict(color="#94a3b8", dash="dot"), showlegend=False))
    fig3.add_trace(go.Scatter(x=post_cum["date"], y=post_cum["cum_effects_lower"], line=dict(color="#94a3b8", dash="dot"), fill="tonexty", showlegend=False))
    fig3.add_hline(y=0, line_dash="dash", line_color="#64748b")
    fig3.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=10), xaxis_title="Date", yaxis_title="Cumulative Effect")
    st.plotly_chart(fig3, use_container_width=True)

with tab3:
    st.subheader("Placebo Tests (Falsification)")
    st.caption("Runs the same model on donor series that should show NO effect. If placebos are significant, the model may be unreliable.")

    with st.spinner("Running placebo tests..."):
        placebos = run_placebo_tests(ci_data, pre_period, post_period)

    if placebos:
        placebo_df = pd.DataFrame(placebos)
        st.dataframe(
            placebo_df.style.format({"rel_effect": "{:.1%}", "p_value": "{:.4f}"}),
            use_container_width=True,
        )
        n_sig = sum(1 for p in placebos if p["significant"])
        if n_sig == 0:
            st.success("No placebo tests are significant — model is credible.")
        else:
            st.warning(f"{n_sig}/{len(placebos)} placebo tests are significant — interpret main result with caution.")
    else:
        st.info("Not enough donor series for placebo tests.")

st.divider()
st.caption("Data: Kaggle Marketing Campaign Performance Dataset (200K rows, 2 years)")
