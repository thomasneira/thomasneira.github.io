"""Synthetic control model using statsmodels (no heavy PyMC dependency)."""

from typing import Optional

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from statsmodels.tsa.seasonal import seasonal_decompose


def fit_synthetic_control(
    data: pd.DataFrame,
    pre_period: list,
    post_period: list,
    alpha: float = 1.0,
) -> dict:
    pre_start, pre_end = pd.Timestamp(pre_period[0]), pd.Timestamp(pre_period[1])
    post_start, post_end = pd.Timestamp(post_period[0]), pd.Timestamp(post_period[1])

    y_col = data.columns[0]
    x_cols = data.columns[1:]

    pre_mask = (data.index >= pre_start) & (data.index <= pre_end)
    post_mask = (data.index >= post_start) & (data.index <= post_end)

    y_pre = data.loc[pre_mask, y_col].values
    X_pre = data.loc[pre_mask, x_cols].values
    y_post = data.loc[post_mask, y_col].values
    X_post = data.loc[post_mask, x_cols].values

    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X_pre, y_pre)

    y_pred_pre = model.predict(X_pre)
    y_pred_post = model.predict(X_post)

    residuals_pre = y_pre - y_pred_pre
    sigma = np.std(residuals_pre, ddof=1)

    point_effects = y_post - y_pred_post
    cum_effects = np.cumsum(point_effects)

    post_dates = data.index[post_mask]
    pre_dates = data.index[pre_mask]

    inferences = pd.DataFrame({
        "date": list(pre_dates) + list(post_dates),
        "response": list(y_pre) + list(y_post),
        "preds": list(y_pred_pre) + list(y_pred_post),
        "point_effects": list(residuals_pre) + list(point_effects),
        "point_effects_lower": list(residuals_pre - 1.96 * sigma) + list(point_effects - 1.96 * sigma),
        "point_effects_upper": list(residuals_pre + 1.96 * sigma) + list(point_effects + 1.96 * sigma),
        "cum_effects": list(np.cumsum(residuals_pre)) + list(cum_effects + np.sum(residuals_pre)),
        "is_post": [False] * len(pre_dates) + [True] * len(post_dates),
    })
    inferences["cum_effects_lower"] = inferences["cum_effects"] - 1.96 * sigma * np.sqrt(range(1, len(inferences) + 1))
    inferences["cum_effects_upper"] = inferences["cum_effects"] + 1.96 * sigma * np.sqrt(range(1, len(inferences) + 1))

    avg_effect = np.mean(point_effects)
    avg_pred = np.mean(y_pred_post)
    t_stat = avg_effect / (sigma / np.sqrt(len(point_effects)))
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(point_effects) - 1))

    return {
        "model": model,
        "inferences": inferences,
        "sigma": sigma,
        "weights": dict(zip(x_cols, model.coef_)),
        "intercept": model.intercept_,
        "summary": {
            "avg_actual": float(np.mean(y_post)),
            "avg_predicted": float(avg_pred),
            "avg_abs_effect": float(avg_effect),
            "avg_rel_effect": float(avg_effect / avg_pred) if avg_pred != 0 else 0.0,
            "cum_abs_effect": float(np.sum(point_effects)),
            "cum_rel_effect": float(np.sum(point_effects) / np.sum(y_pred_post)) if np.sum(y_pred_post) != 0 else 0.0,
            "p_value": float(p_value),
            "significant": p_value < 0.05,
        },
    }


def get_pointwise_effects(result: dict) -> pd.DataFrame:
    inf = result["inferences"]
    return inf[["date", "point_effects", "point_effects_lower", "point_effects_upper"]]


def get_cumulative_effects(result: dict) -> pd.DataFrame:
    inf = result["inferences"]
    return inf[["date", "cum_effects", "cum_effects_lower", "cum_effects_upper"]]


def run_placebo_tests(
    data: pd.DataFrame,
    pre_period: list,
    post_period: list,
    n_placebos: int = 5,
) -> list:
    donor_cols = list(data.columns[1:])
    results = []

    for col in donor_cols[:n_placebos]:
        placebo_data = data.copy()
        placebo_data.iloc[:, 0] = placebo_data[col]
        remaining = [c for c in data.columns[1:] if c != col]
        if not remaining:
            continue
        placebo_data = placebo_data[[data.columns[0]] + remaining]

        try:
            r = fit_synthetic_control(placebo_data, pre_period, post_period)
            results.append({
                "donor": col,
                "rel_effect": r["summary"]["avg_rel_effect"],
                "p_value": r["summary"]["p_value"],
                "significant": r["summary"]["significant"],
            })
        except Exception:
            continue

    return results


def format_summary(result: dict) -> str:
    s = result["summary"]
    lines = [
        "Synthetic Control Analysis Summary",
        "=" * 40,
        f"Average actual value (post):      {s['avg_actual']:.4f}",
        f"Average predicted (counterfact.):  {s['avg_predicted']:.4f}",
        f"Average causal effect:            {s['avg_abs_effect']:.4f}",
        f"Relative effect:                  {s['avg_rel_effect']:.2%}",
        f"Cumulative effect:                {s['cum_abs_effect']:.4f}",
        f"p-value:                          {s['p_value']:.4f}",
        f"Significant at 0.05:              {'Yes' if s['significant'] else 'No'}",
    ]
    return "\n".join(lines)
