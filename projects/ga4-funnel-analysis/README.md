# GA4 Top-of-Funnel Conversion Analysis

Real event-level ecommerce data from Google's own store (GA4 BigQuery public export). Models the top-of-funnel drop-off — `session_start → view_item → add_to_cart → purchase` — by traffic source and identifies which acquisition channels deliver the highest-quality early-stage engagement.

## Key Questions

1. **Which traffic sources drive volume vs. quality?** Compare acquisition channels by funnel progression rate, not just session count.
2. **Where is the biggest drop-off?** Identify the funnel stage with the highest abandonment and which sources are worst there.
3. **What does a high-quality early session look like?** Profile sessions that make it past `view_item` vs. those that bounce.

## Dataset

- **Source**: [GA4 BigQuery Public Dataset](https://support.google.com/analytics/answer/10937659) / [Kaggle mirror](https://www.kaggle.com/datasets/pdaasha/ga4-obfuscated-sample-ecommerce-jan2021)
- **Scope**: Google Merchandise Store, 3 months of event-level data (Nov 2020 – Jan 2021)
- **Key fields**: `event_name`, `event_timestamp`, `user_pseudo_id`, `traffic_source`, `device`, `geo`

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python (pandas) | Data cleaning & transformation |
| SQL | Funnel stage extraction from event-level data |
| dbt | Data modeling layer (staging → intermediate → marts) |
| Streamlit | Interactive dashboard |

## Project Structure

```
ga4-funnel-analysis/
├── README.md
├── requirements.txt
├── data/                    # CSV exports (gitignored)
│   └── README.md            # Data download instructions
├── src/
│   ├── extract.py           # BigQuery extraction or CSV loading
│   ├── transform.py         # Funnel stage assignment & sessionization
│   └── analyze.py           # Drop-off rates, channel scoring
└── app/
    └── streamlit_app.py     # Interactive funnel dashboard
```

## JD Alignment

> **Top of Funnel Insights**: Own reporting and analysis for upper funnel programs — measuring reach, awareness, engagement, and early-stage interest.

> **Audience & Reach Analysis**: Analyze who is being reached — segmenting by ICP, account tier, persona, and geography.
