"""Sessionize events and compute funnel progression per session."""

from typing import Optional

import pandas as pd
from .extract import FUNNEL_EVENTS, FUNNEL_ORDER


def sessionize(df: pd.DataFrame) -> pd.DataFrame:
    if "user_session" in df.columns:
        df["session_key"] = df["user_id"].astype(str) + "_" + df["user_session"].astype(str)
    else:
        df = df.sort_values(["user_id", "event_time"])
        df["time_gap"] = df.groupby("user_id")["event_time"].diff()
        df["new_session"] = (df["time_gap"] > pd.Timedelta(minutes=30)) | df["time_gap"].isna()
        df["session_id"] = df.groupby("user_id")["new_session"].cumsum()
        df["session_key"] = df["user_id"].astype(str) + "_" + df["session_id"].astype(str)
    return df


def build_session_funnel(df: pd.DataFrame) -> pd.DataFrame:
    session_events = (
        df.groupby("session_key")["event_type"]
        .apply(set)
        .reset_index()
        .rename(columns={"event_type": "events_set"})
    )

    for event in FUNNEL_EVENTS:
        session_events[f"reached_{event}"] = session_events["events_set"].apply(
            lambda s: event in s
        )

    session_meta = (
        df.groupby("session_key")
        .agg(
            user_id=("user_id", "first"),
            category_top=("category_top", "first"),
            brand=("brand", "first"),
            event_date=("event_date", "first"),
            max_funnel_stage=("funnel_stage", "max"),
        )
        .reset_index()
    )

    return session_meta.merge(session_events.drop(columns=["events_set"]), on="session_key")


def compute_funnel_rates(session_funnel: pd.DataFrame, group_col: Optional[str] = None) -> pd.DataFrame:
    reached_cols = [f"reached_{e}" for e in FUNNEL_EVENTS]

    if group_col:
        counts = session_funnel.groupby(group_col)[reached_cols].sum().reset_index()
    else:
        counts = pd.DataFrame([session_funnel[reached_cols].sum()])

    for i in range(1, len(FUNNEL_EVENTS)):
        prev = f"reached_{FUNNEL_EVENTS[i-1]}"
        curr = f"reached_{FUNNEL_EVENTS[i]}"
        rate_col = f"rate_{FUNNEL_EVENTS[i-1]}_to_{FUNNEL_EVENTS[i]}"
        counts[rate_col] = (counts[curr] / counts[prev].replace(0, float("nan"))) * 100

    return counts


def top_dropoff_stage(funnel_rates: pd.DataFrame) -> dict:
    rate_cols = [c for c in funnel_rates.columns if c.startswith("rate_")]
    if not rate_cols:
        return {"from_stage": "unknown", "to_stage": "unknown", "avg_progression_rate": 0}

    worst_col = funnel_rates[rate_cols].mean().idxmin()
    stages = worst_col.replace("rate_", "").split("_to_")
    return {
        "from_stage": stages[0],
        "to_stage": stages[1],
        "avg_progression_rate": funnel_rates[worst_col].mean(),
    }
