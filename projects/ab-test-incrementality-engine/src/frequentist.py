"""Frequentist A/B test analysis: z-tests, chi-square, effect sizes."""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def load_ab_data(filename: str = "marketing_AB.csv") -> pd.DataFrame:
    filepath = DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"{filepath} not found. See data/README.md for download instructions."
        )
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def conversion_rate_test(
    control_conversions: int,
    control_total: int,
    treatment_conversions: int,
    treatment_total: int,
) -> dict:
    p_control = control_conversions / control_total
    p_treatment = treatment_conversions / treatment_total
    p_pooled = (control_conversions + treatment_conversions) / (control_total + treatment_total)

    se = np.sqrt(p_pooled * (1 - p_pooled) * (1 / control_total + 1 / treatment_total))
    z_stat = (p_treatment - p_control) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    lift = (p_treatment - p_control) / p_control if p_control > 0 else 0
    ci_95 = 1.96 * np.sqrt(
        p_treatment * (1 - p_treatment) / treatment_total
        + p_control * (1 - p_control) / control_total
    )

    return {
        "control_rate": p_control,
        "treatment_rate": p_treatment,
        "absolute_lift": p_treatment - p_control,
        "relative_lift": lift,
        "z_statistic": z_stat,
        "p_value": p_value,
        "ci_lower": (p_treatment - p_control) - ci_95,
        "ci_upper": (p_treatment - p_control) + ci_95,
        "significant": p_value < 0.05,
    }


def chi_square_test(control_conv: int, control_total: int, treat_conv: int, treat_total: int) -> dict:
    table = np.array([
        [control_conv, control_total - control_conv],
        [treat_conv, treat_total - treat_conv],
    ])
    chi2, p_value, dof, expected = stats.chi2_contingency(table)
    n = table.sum()
    cramers_v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))

    return {
        "chi2": chi2,
        "p_value": p_value,
        "dof": dof,
        "cramers_v": cramers_v,
        "significant": p_value < 0.05,
    }


def cohens_h(p1: float, p2: float) -> float:
    return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))


def compute_effect_size(control_rate: float, treatment_rate: float) -> dict:
    h = cohens_h(treatment_rate, control_rate)
    if abs(h) < 0.2:
        magnitude = "Negligible"
    elif abs(h) < 0.5:
        magnitude = "Small"
    elif abs(h) < 0.8:
        magnitude = "Medium"
    else:
        magnitude = "Large"

    return {"cohens_h": h, "magnitude": magnitude}
