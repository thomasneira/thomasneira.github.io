"""Prepare time series data for synthetic control analysis."""

from typing import Optional, Tuple

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def load_campaign_data(filename: str = "marketing_campaign_dataset.csv") -> pd.DataFrame:
    filepath = DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"{filepath} not found. See data/README.md for download instructions."
        )
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def build_daily_series(df: pd.DataFrame, date_col: str, metric_col: str, group_col: str) -> pd.DataFrame:
    daily = (
        df.groupby([date_col, group_col])[metric_col]
        .mean()
        .reset_index()
    )
    daily[date_col] = pd.to_datetime(daily[date_col])
    pivot = daily.pivot(index=date_col, columns=group_col, values=metric_col)
    pivot = pivot.sort_index().ffill().bfill()
    return pivot


def select_treatment_and_donors(
    pivot: pd.DataFrame,
    treatment_col: str,
    min_correlation: float = 0.5,
) -> Tuple[pd.Series, pd.DataFrame]:
    treatment = pivot[treatment_col]
    donors = pivot.drop(columns=[treatment_col])

    correlations = donors.corrwith(treatment)
    valid_donors = correlations[correlations.abs() >= min_correlation].index.tolist()

    if len(valid_donors) < 2:
        valid_donors = correlations.abs().nlargest(3).index.tolist()

    return treatment, donors[valid_donors]


def define_intervention_period(
    series: pd.Series,
    intervention_start: Optional[str] = None,
    pre_pct: float = 0.7,
) -> Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    dates = series.index
    if intervention_start:
        start = pd.Timestamp(intervention_start)
    else:
        n = len(dates)
        start = dates[int(n * pre_pct)]

    pre_start = dates[0]
    post_end = dates[-1]
    return pre_start, start, post_end


def build_causal_impact_input(
    treatment: pd.Series,
    donors: pd.DataFrame,
) -> pd.DataFrame:
    ci_data = pd.DataFrame({"y": treatment})
    for col in donors.columns:
        ci_data[col] = donors[col]
    ci_data.index = treatment.index
    return ci_data
