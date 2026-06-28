"""Sequential testing with O'Brien-Fleming boundaries."""

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


def obrien_fleming_boundary(n_looks: int, alpha: float = 0.05) -> list[float]:
    z_final = stats.norm.ppf(1 - alpha / 2)
    boundaries = []
    for k in range(1, n_looks + 1):
        fraction = k / n_looks
        z_k = z_final / np.sqrt(fraction)
        boundaries.append(z_k)
    return boundaries


def sequential_z_scores(
    control_conversions: np.ndarray,
    treatment_conversions: np.ndarray,
    n_looks: int = 10,
) -> pd.DataFrame:
    n_control = len(control_conversions)
    n_treatment = len(treatment_conversions)
    look_sizes = np.linspace(
        max(100, min(n_control, n_treatment) // n_looks),
        min(n_control, n_treatment),
        n_looks,
        dtype=int,
    )

    boundaries = obrien_fleming_boundary(n_looks)
    results = []

    for i, size in enumerate(look_sizes):
        c_sample = control_conversions[:size]
        t_sample = treatment_conversions[:size]

        p_c = c_sample.mean()
        p_t = t_sample.mean()
        p_pool = np.concatenate([c_sample, t_sample]).mean()

        se = np.sqrt(p_pool * (1 - p_pool) * (2 / size)) if p_pool > 0 and p_pool < 1 else 1
        z = (p_t - p_c) / se if se > 0 else 0

        results.append({
            "look": i + 1,
            "sample_size": int(size),
            "control_rate": p_c,
            "treatment_rate": p_t,
            "z_score": z,
            "boundary": boundaries[i],
            "crossed": abs(z) > boundaries[i],
        })

    return pd.DataFrame(results)


def earliest_stopping_point(seq_results: pd.DataFrame) -> Optional[dict]:
    crossed = seq_results[seq_results["crossed"]]
    if crossed.empty:
        return None
    first = crossed.iloc[0]
    return {
        "look": int(first["look"]),
        "sample_size": int(first["sample_size"]),
        "z_score": first["z_score"],
        "boundary": first["boundary"],
        "pct_of_full_sample": first["sample_size"] / seq_results["sample_size"].max() * 100,
    }
