"""Higher-level analysis: sensitivity, effect decomposition."""

import pandas as pd
import numpy as np
from .model import fit_causal_impact, extract_summary


def sensitivity_analysis(
    data: pd.DataFrame,
    pre_period: list[str],
    post_period: list[str],
    pre_shifts: list[int] = [-7, -3, 0, 3, 7],
) -> pd.DataFrame:
    base_start = pd.Timestamp(post_period[0])
    results = []

    for shift in pre_shifts:
        shifted_start = base_start + pd.Timedelta(days=shift)
        if shifted_start <= pd.Timestamp(pre_period[0]):
            continue
        shifted_pre = [pre_period[0], str(shifted_start - pd.Timedelta(days=1))]
        shifted_post = [str(shifted_start), post_period[1]]

        try:
            ci = fit_causal_impact(data, shifted_pre, shifted_post)
            summary = extract_summary(ci)
            summary["shift_days"] = shift
            results.append(summary)
        except Exception:
            continue

    return pd.DataFrame(results)


def effect_over_time(ci_inferences: pd.DataFrame) -> pd.DataFrame:
    df = ci_inferences.copy()
    df["week"] = df["date"].dt.isocalendar().week
    weekly = (
        df.groupby("week")
        .agg(
            avg_effect=("point_effects", "mean"),
            effect_lower=("point_effects_lower", "mean"),
            effect_upper=("point_effects_upper", "mean"),
        )
        .reset_index()
    )
    return weekly
