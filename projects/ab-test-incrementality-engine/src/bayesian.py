"""Bayesian A/B test analysis with PyMC."""

import numpy as np

try:
    import pymc as pm
    import arviz as az
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False


def bayesian_ab_test(
    control_conversions: int,
    control_total: int,
    treatment_conversions: int,
    treatment_total: int,
    n_samples: int = 10_000,
) -> dict:
    if not HAS_PYMC:
        return _beta_analytical(control_conversions, control_total, treatment_conversions, treatment_total, n_samples)

    with pm.Model() as model:
        p_control = pm.Beta("p_control", alpha=1, beta=1)
        p_treatment = pm.Beta("p_treatment", alpha=1, beta=1)

        pm.Binomial("obs_control", n=control_total, p=p_control, observed=control_conversions)
        pm.Binomial("obs_treatment", n=treatment_total, p=p_treatment, observed=treatment_conversions)

        lift = pm.Deterministic("lift", (p_treatment - p_control) / p_control)
        diff = pm.Deterministic("diff", p_treatment - p_control)

        trace = pm.sample(n_samples, return_inferencedata=True, progressbar=False)

    diff_samples = trace.posterior["diff"].values.flatten()
    lift_samples = trace.posterior["lift"].values.flatten()

    return {
        "prob_treatment_better": float((diff_samples > 0).mean()),
        "expected_lift": float(np.median(lift_samples)),
        "lift_ci_95": [float(np.percentile(lift_samples, 2.5)), float(np.percentile(lift_samples, 97.5))],
        "expected_diff": float(np.median(diff_samples)),
        "diff_ci_95": [float(np.percentile(diff_samples, 2.5)), float(np.percentile(diff_samples, 97.5))],
        "diff_samples": diff_samples,
        "lift_samples": lift_samples,
    }


def _beta_analytical(
    control_conv: int, control_total: int,
    treat_conv: int, treat_total: int,
    n_samples: int,
) -> dict:
    rng = np.random.default_rng(42)
    control_samples = rng.beta(control_conv + 1, control_total - control_conv + 1, n_samples)
    treat_samples = rng.beta(treat_conv + 1, treat_total - treat_conv + 1, n_samples)

    diff_samples = treat_samples - control_samples
    lift_samples = np.where(control_samples > 0, diff_samples / control_samples, 0)

    return {
        "prob_treatment_better": float((diff_samples > 0).mean()),
        "expected_lift": float(np.median(lift_samples)),
        "lift_ci_95": [float(np.percentile(lift_samples, 2.5)), float(np.percentile(lift_samples, 97.5))],
        "expected_diff": float(np.median(diff_samples)),
        "diff_ci_95": [float(np.percentile(diff_samples, 2.5)), float(np.percentile(diff_samples, 97.5))],
        "diff_samples": diff_samples,
        "lift_samples": lift_samples,
    }
