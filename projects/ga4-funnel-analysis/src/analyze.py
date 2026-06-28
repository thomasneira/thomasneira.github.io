"""Channel quality scoring and funnel analysis."""

import pandas as pd
from .extract import FUNNEL_EVENTS


def score_channels(session_funnel: pd.DataFrame, channel_col: str = "category_top") -> pd.DataFrame:
    channels = (
        session_funnel.groupby(channel_col)
        .agg(
            total_sessions=("session_key", "count"),
            sessions_view=("reached_view", "sum"),
            sessions_cart=("reached_cart", "sum"),
            sessions_purchase=("reached_purchase", "sum"),
        )
        .reset_index()
    )

    channels["view_rate"] = channels["sessions_view"] / channels["total_sessions"]
    channels["cart_rate"] = channels["sessions_cart"] / channels["total_sessions"]
    channels["purchase_rate"] = channels["sessions_purchase"] / channels["total_sessions"]

    channels["quality_score"] = (
        channels["view_rate"] * 0.2
        + channels["cart_rate"] * 0.3
        + channels["purchase_rate"] * 0.5
    )
    channels["quality_rank"] = channels["quality_score"].rank(ascending=False, method="min")

    return channels.sort_values("quality_score", ascending=False)


def volume_vs_quality_matrix(channels: pd.DataFrame) -> pd.DataFrame:
    median_sessions = channels["total_sessions"].median()
    median_quality = channels["quality_score"].median()

    def classify(row):
        high_vol = row["total_sessions"] >= median_sessions
        high_qual = row["quality_score"] >= median_quality
        if high_vol and high_qual:
            return "Star (High Vol + Quality)"
        elif high_vol and not high_qual:
            return "Volume Play (Low Quality)"
        elif not high_vol and high_qual:
            return "Hidden Gem (High Quality)"
        else:
            return "Underperformer"

    channels["quadrant"] = channels.apply(classify, axis=1)
    return channels


def funnel_by_brand(session_funnel: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    brand_counts = session_funnel["brand"].value_counts().head(top_n).index.tolist()
    filtered = session_funnel[session_funnel["brand"].isin(brand_counts)]

    reached_cols = [f"reached_{e}" for e in FUNNEL_EVENTS]
    brand_funnel = (
        filtered.groupby("brand")[reached_cols]
        .sum()
        .reset_index()
    )
    for col in reached_cols:
        pct_col = col.replace("reached_", "pct_")
        brand_funnel[pct_col] = (
            brand_funnel[col] / brand_funnel["reached_view"] * 100
        )
    return brand_funnel
