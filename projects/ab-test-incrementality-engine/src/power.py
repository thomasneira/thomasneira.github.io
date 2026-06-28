"""Power analysis and minimum detectable effect (MDE) sizing."""

from typing import Optional

import numpy as np
from scipy import stats


def minimum_detectable_effect(
    baseline_rate: float,
    sample_size_per_group: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    se = np.sqrt(2 * baseline_rate * (1 - baseline_rate) / sample_size_per_group)
    mde = (z_alpha + z_beta) * se
    return mde


def required_sample_size(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    p1 = baseline_rate
    p2 = baseline_rate + mde
    n = ((z_alpha * np.sqrt(2 * p1 * (1 - p1)) + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) / mde) ** 2
    return int(np.ceil(n))


def power_curve(
    baseline_rate: float,
    sample_size_per_group: int,
    alpha: float = 0.05,
    mde_range: Optional[np.ndarray] = None,
) -> list[dict]:
    if mde_range is None:
        mde_range = np.linspace(0.001, 0.05, 50)

    results = []
    for mde in mde_range:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        p1 = baseline_rate
        p2 = baseline_rate + mde
        se = np.sqrt(p1 * (1 - p1) / sample_size_per_group + p2 * (1 - p2) / sample_size_per_group)
        z_beta = (abs(p2 - p1) / se) - z_alpha
        pwr = stats.norm.cdf(z_beta)
        results.append({"mde": mde, "power": pwr})

    return results


def post_hoc_power(
    control_rate: float,
    treatment_rate: float,
    control_n: int,
    treatment_n: int,
    alpha: float = 0.05,
) -> dict:
    observed_effect = abs(treatment_rate - control_rate)
    se = np.sqrt(
        control_rate * (1 - control_rate) / control_n
        + treatment_rate * (1 - treatment_rate) / treatment_n
    )
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = (observed_effect / se) - z_alpha
    achieved_power = stats.norm.cdf(z_beta)

    mde = minimum_detectable_effect(control_rate, min(control_n, treatment_n), alpha)

    return {
        "observed_effect": observed_effect,
        "achieved_power": achieved_power,
        "mde_at_80_power": mde,
        "adequately_powered": achieved_power >= 0.8,
        "could_detect_observed": observed_effect >= mde,
    }
