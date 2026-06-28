"""Ecommerce Top-of-Funnel Conversion Analysis Dashboard."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.extract import load_from_csv, FUNNEL_EVENTS
from src.transform import sessionize, build_session_funnel, compute_funnel_rates, top_dropoff_stage
from src.analyze import score_channels, volume_vs_quality_matrix, funnel_by_brand

st.set_page_config(page_title="Top-of-Funnel Analysis", layout="wide")
st.title("Top-of-Funnel Conversion Analysis")
st.caption("885K ecommerce events | view → cart → purchase by category & brand")


@st.cache_data
def load_and_process():
    df = load_from_csv()
    df = sessionize(df)
    session_funnel = build_session_funnel(df)
    return df, session_funnel


try:
    raw, session_funnel = load_and_process()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# --- Sidebar ---
st.sidebar.header("Filters")
categories = ["All"] + sorted(session_funnel["category_top"].dropna().unique().tolist())
selected_cat = st.sidebar.selectbox("Product Category", categories)

top_brands = session_funnel["brand"].value_counts().head(20).index.tolist()
brands = ["All"] + sorted([b for b in top_brands if pd.notna(b)])
selected_brand = st.sidebar.selectbox("Brand", brands)

filtered = session_funnel.copy()
if selected_cat != "All":
    filtered = filtered[filtered["category_top"] == selected_cat]
if selected_brand != "All":
    filtered = filtered[filtered["brand"] == selected_brand]

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["Funnel Overview", "Category Quality", "Brand Breakdown"])

with tab1:
    st.subheader("Funnel Drop-off")
    rates = compute_funnel_rates(filtered)
    reached_cols = [f"reached_{e}" for e in FUNNEL_EVENTS]
    funnel_data = pd.DataFrame({
        "Stage": [e.replace("_", " ").title() for e in FUNNEL_EVENTS],
        "Sessions": [int(rates[col].iloc[0]) for col in reached_cols],
    })
    funnel_data["Pct of Start"] = (funnel_data["Sessions"] / funnel_data["Sessions"].iloc[0] * 100).round(1)

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = go.Figure(go.Funnel(
            y=funnel_data["Stage"],
            x=funnel_data["Sessions"],
            textinfo="value+percent initial",
        ))
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        dropoff = top_dropoff_stage(rates)
        st.metric("Biggest Drop-off", f"{dropoff['from_stage']} → {dropoff['to_stage']}")
        st.metric("Progression Rate", f"{dropoff['avg_progression_rate']:.1f}%")
        st.metric("Total Sessions", f"{len(filtered):,}")
        purchase_rate = funnel_data["Pct of Start"].iloc[-1]
        st.metric("Overall Purchase Rate", f"{purchase_rate:.2f}%")

with tab2:
    st.subheader("Product Category: Volume vs. Quality")
    channels = score_channels(session_funnel, "category_top")
    channels = volume_vs_quality_matrix(channels)

    fig = px.scatter(
        channels[channels["total_sessions"] >= 50],
        x="total_sessions",
        y="quality_score",
        size="sessions_purchase",
        color="quadrant",
        hover_name="category_top",
        labels={"total_sessions": "Session Volume", "quality_score": "Quality Score"},
        height=500,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        channels[["category_top", "total_sessions", "cart_rate", "purchase_rate", "quality_score", "quadrant"]]
        .head(20)
        .style.format({"cart_rate": "{:.1%}", "purchase_rate": "{:.1%}", "quality_score": "{:.4f}"}),
        use_container_width=True,
    )

with tab3:
    st.subheader("Funnel by Top Brands")
    brand_data = funnel_by_brand(session_funnel)

    pct_cols = [c for c in brand_data.columns if c.startswith("pct_")]
    melted = brand_data.melt(
        id_vars=["brand"],
        value_vars=pct_cols,
        var_name="stage",
        value_name="pct",
    )
    melted["stage"] = melted["stage"].str.replace("pct_", "").str.replace("_", " ").str.title()

    fig = px.bar(
        melted,
        x="brand",
        y="pct",
        color="stage",
        barmode="group",
        labels={"pct": "% of Viewers", "brand": "Brand"},
        height=450,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=10), xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("Data: Kaggle eCommerce Events History (Electronics Store, 885K events)")
