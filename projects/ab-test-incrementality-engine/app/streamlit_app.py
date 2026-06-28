"""Marketing A/B Test Incrementality Engine Dashboard."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.frequentist import load_ab_data, conversion_rate_test, chi_square_test, compute_effect_size
from src.bayesian import bayesian_ab_test
from src.sequential import sequential_z_scores, earliest_stopping_point
from src.power import power_curve, post_hoc_power

st.set_page_config(page_title="A/B Test Incrementality Engine", layout="wide")
st.title("Marketing A/B Test Incrementality Engine")
st.caption("Frequentist + Bayesian + Sequential Testing + Power Analysis")


@st.cache_data
def load_data():
    return load_ab_data()


try:
    raw = load_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# --- Detect columns ---
st.sidebar.header("Column Mapping")
group_col = st.sidebar.selectbox("Group Column", raw.columns.tolist(), index=0)
unique_groups = raw[group_col].dropna().unique().tolist()

control_label = st.sidebar.selectbox("Control Group Value", unique_groups)
treatment_label = st.sidebar.selectbox(
    "Treatment Group Value",
    [g for g in unique_groups if g != control_label],
)

conversion_col = st.sidebar.selectbox("Conversion Column", raw.columns.tolist(), index=min(1, len(raw.columns) - 1))

# --- Compute stats ---
control = raw[raw[group_col] == control_label]
treatment = raw[raw[group_col] == treatment_label]

control_conv = int(control[conversion_col].sum())
control_total = len(control)
treat_conv = int(treatment[conversion_col].sum())
treat_total = len(treatment)

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["Frequentist", "Bayesian", "Sequential Testing", "Power Analysis"])

with tab1:
    st.subheader("Frequentist Analysis")
    result = conversion_rate_test(control_conv, control_total, treat_conv, treat_total)
    chi2 = chi_square_test(control_conv, control_total, treat_conv, treat_total)
    effect = compute_effect_size(result["control_rate"], result["treatment_rate"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Control Rate", f"{result['control_rate']:.4f}")
    col2.metric("Treatment Rate", f"{result['treatment_rate']:.4f}")
    col3.metric("Relative Lift", f"{result['relative_lift']:.2%}")
    sig_color = "normal" if result["significant"] else "off"
    col4.metric("p-value", f"{result['p_value']:.4f}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Z-statistic", f"{result['z_statistic']:.3f}")
    col6.metric("95% CI", f"[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
    col7.metric("Cohen's h", f"{effect['cohens_h']:.3f} ({effect['magnitude']})")
    col8.metric("Chi-square p", f"{chi2['p_value']:.4f}")

    if result["significant"]:
        st.success(f"Statistically significant at alpha=0.05 (p={result['p_value']:.4f})")
    else:
        st.warning(f"Not statistically significant (p={result['p_value']:.4f})")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Control", "Treatment"], y=[result["control_rate"], result["treatment_rate"]], marker_color=["#94a3b8", "#2563eb"]))
    fig.add_trace(go.Scatter(
        x=["Treatment", "Treatment"],
        y=[result["treatment_rate"] + result["ci_lower"], result["treatment_rate"] + result["ci_upper"]],
        mode="lines",
        line=dict(color="#ef4444", width=3),
        name="95% CI on diff",
    ))
    fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=10), yaxis_title="Conversion Rate", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Bayesian Analysis")
    with st.spinner("Running Bayesian inference..."):
        bayes = bayesian_ab_test(control_conv, control_total, treat_conv, treat_total)

    col1, col2, col3 = st.columns(3)
    col1.metric("P(Treatment > Control)", f"{bayes['prob_treatment_better']:.1%}")
    col2.metric("Expected Lift", f"{bayes['expected_lift']:.2%}")
    col3.metric("95% Credible Interval", f"[{bayes['lift_ci_95'][0]:.2%}, {bayes['lift_ci_95'][1]:.2%}]")

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=bayes["diff_samples"], nbinsx=80, name="P(treatment) - P(control)", marker_color="#2563eb", opacity=0.7))
    fig.add_vline(x=0, line_dash="dash", line_color="#ef4444")
    fig.add_vline(x=bayes["expected_diff"], line_dash="solid", line_color="#16a34a")
    fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=10), xaxis_title="Treatment - Control", yaxis_title="Density")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Sequential Testing (O'Brien-Fleming)")
    st.caption("Could we have stopped this test earlier without inflating the false positive rate?")

    control_outcomes = control[conversion_col].values.astype(float)
    treatment_outcomes = treatment[conversion_col].values.astype(float)

    n_looks = st.slider("Number of Interim Looks", 3, 20, 10)
    seq = sequential_z_scores(control_outcomes, treatment_outcomes, n_looks)
    stop = earliest_stopping_point(seq)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=seq["look"], y=seq["z_score"], name="Z-score", mode="lines+markers", line=dict(color="#2563eb")))
    fig.add_trace(go.Scatter(x=seq["look"], y=seq["boundary"], name="O'Brien-Fleming Boundary", mode="lines", line=dict(color="#ef4444", dash="dash")))
    fig.add_trace(go.Scatter(x=seq["look"], y=-seq["boundary"], name="Lower Boundary", mode="lines", line=dict(color="#ef4444", dash="dash"), showlegend=False))
    fig.add_hline(y=0, line_dash="dot", line_color="#94a3b8")
    fig.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=10), xaxis_title="Interim Look", yaxis_title="Z-score")
    st.plotly_chart(fig, use_container_width=True)

    if stop:
        st.success(f"Could have stopped at look {stop['look']} (sample size {stop['sample_size']:,}) — {stop['pct_of_full_sample']:.0f}% of the full sample.")
    else:
        st.info("No interim look crossed the O'Brien-Fleming boundary — full sample was needed.")

with tab4:
    st.subheader("Power Analysis & MDE")

    post_hoc = post_hoc_power(result["control_rate"], result["treatment_rate"], control_total, treat_total)
    col1, col2, col3 = st.columns(3)
    col1.metric("Achieved Power", f"{post_hoc['achieved_power']:.1%}")
    col2.metric("MDE at 80% Power", f"{post_hoc['mde_at_80_power']:.4f}")
    col3.metric("Adequately Powered?", "Yes" if post_hoc["adequately_powered"] else "No")

    curve = power_curve(result["control_rate"], min(control_total, treat_total))
    curve_df = pd.DataFrame(curve)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve_df["mde"], y=curve_df["power"], name="Power Curve", line=dict(color="#2563eb")))
    fig.add_hline(y=0.8, line_dash="dash", line_color="#ef4444", annotation_text="80% power")
    fig.add_vline(x=post_hoc["observed_effect"], line_dash="dot", line_color="#16a34a", annotation_text="Observed effect")
    fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=10), xaxis_title="Minimum Detectable Effect", yaxis_title="Power")
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("Data: Kaggle Marketing A/B Testing Dataset")
