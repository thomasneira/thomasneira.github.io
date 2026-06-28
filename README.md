# Thomas Neira — GTM & Marketing Analytics Portfolio

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![SQL](https://img.shields.io/badge/SQL-BigQuery%20%7C%20Snowflake-orange.svg)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)

Portfolio of analytics projects demonstrating top-of-funnel measurement, brand marketing analytics, experimentation, and audience segmentation — built on real public data.

---

## GTM Analytics Projects

### Tier 1 — Core Portfolio

| # | Project | What It Demonstrates | Tech Stack |
|---|---------|---------------------|------------|
| 3 | **[GA4 Top-of-Funnel Conversion Analysis](projects/ga4-funnel-analysis/)** | Funnel drop-off modeling by traffic source, channel quality scoring | BigQuery SQL, Python, dbt, Streamlit |
| 7 | **[Synthetic Control for Campaign Lift](projects/synthetic-control-campaign-lift/)** | Causal inference for brand incrementality — gold standard when you can't randomize | Python (CausalImpact), SQL, Streamlit |
| 4 | **[A/B Test Incrementality Engine](projects/ab-test-incrementality-engine/)** | Frequentist + Bayesian + sequential testing + power analysis | Python (scipy, PyMC), SQL, Streamlit |

### Tier 2 — In Progress

| # | Project | What It Demonstrates |
|---|---------|---------------------|
| 2 | GDELT Brand Media Coverage & Sentiment | Share of voice from global news data |
| 1 | Brand Share-of-Search Tracker | Google Trends mindshare proxy with changepoint detection |
| 5 | LinkedIn ICP Audience Reach Segmentation | ICP fit scoring and tier clustering |
| 8 | Executive Outreach Signal Detector | Propensity modeling with XGBoost + SHAP |

### Tier 3 — Planned

| # | Project | What It Demonstrates |
|---|---------|---------------------|
| 9 | Geographic Market Opportunity Analysis | Coverage gap mapping with GeoPandas |
| 10 | Brand Consideration Funnel | Full awareness → intent modeling (capstone) |

---

## Previous Projects

| Project | What It Demonstrates |
|---------|---------------------|
| [Lead Conversion Optimization (ExtraaLearn)](ExtraaLearn_Analysis.html) | End-to-end ML pipeline: hypothesis testing, XGBoost, SHAP, ROI quantification |
| [Tennis Match Prediction](TennisAnalytics_Prediction.html) | Feature engineering, ensemble methods, sports analytics |

---

## Quick Start

```bash
# Clone
git clone https://github.com/thomasneira/thomasneira.github.io.git
cd thomasneira.github.io

# Set up Kaggle API (one-time)
pip install kaggle
# Download token from https://www.kaggle.com/settings → "Create New Token"
# Place kaggle.json at ~/.kaggle/kaggle.json

# Download all datasets
cd projects && bash download_all_data.sh

# Run any project dashboard
cd ga4-funnel-analysis
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

---

## Skills Demonstrated

| Category | Skills |
|----------|--------|
| **Top-of-Funnel Analytics** | Funnel modeling, drop-off analysis, channel quality scoring, reach measurement |
| **Brand Measurement** | Share of search, share of voice, campaign lift, mindshare proxies |
| **Experimentation** | A/B testing, Bayesian inference, sequential testing, MDE sizing, synthetic control |
| **Causal Inference** | CausalImpact, synthetic counterfactuals, placebo tests, geo holdouts |
| **Data Engineering** | SQL (BigQuery, Snowflake), dbt, Python ETL pipelines |
| **Visualization** | Streamlit dashboards, Plotly, Tableau |

---

**Thomas Neira** · [GitHub](https://github.com/thomasneira) · Chicago, IL
