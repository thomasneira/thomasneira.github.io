# Synthetic Control for Campaign Lift Measurement

Implements the gold-standard causal inference method for measuring brand campaign incrementality when individual-level randomization isn't possible. Builds a synthetic counterfactual ("what would engagement have looked like without the campaign?") using pre-campaign donor series, then measures the causal lift.

## Key Questions

1. **Did the campaign actually cause a lift, or was it organic growth?** Synthetic control isolates the causal effect from trend/seasonality.
2. **How large was the incremental impact?** Posterior interval for the cumulative causal effect over the campaign window.
3. **When did the effect start and fade?** Pointwise impact estimates show the time profile of campaign influence.

## Dataset

- **Source**: [Marketing Campaign Performance Dataset](https://www.kaggle.com/datasets/manishabhatt22/marketing-campaign-performance-dataset) (200K rows, 2 years)
- **Scope**: Multi-brand, multi-channel campaign data with engagement scores, clicks, impressions, and customer segments
- **Key fields**: `Campaign_Type`, `Target_Audience`, `Clicks`, `Impressions`, `Engagement_Score`, `Customer_Segment`, `Date`

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python (CausalImpact) | Bayesian structural time series for synthetic control |
| Python (statsmodels) | Pre/post regression, trend decomposition |
| SQL | Data aggregation & time series construction |
| dbt | Staging and mart models |
| Streamlit | Interactive results dashboard |

## Project Structure

```
synthetic-control-campaign-lift/
├── README.md
├── requirements.txt
├── data/                    # CSV exports (gitignored)
│   └── README.md            # Data download instructions
├── src/
│   ├── prepare.py           # Time series aggregation, donor pool selection
│   ├── model.py             # CausalImpact model fitting & diagnostics
│   └── analyze.py           # Effect size estimation, sensitivity checks
└── app/
    └── streamlit_app.py     # Interactive lift measurement dashboard
```

## JD Alignment

> **Experimentation**: Design and analyze experiments across Mindshare programs — helping the team test messaging, channels, formats, and targeting.

> **Brand Marketing Measurement**: Develop frameworks to assess brand campaign effectiveness — tracking how brand investments translate into audience reach, mindshare, and downstream consideration.
