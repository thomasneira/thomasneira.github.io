# Lead Conversion Optimization for EdTech: A Predictive Analytics Case Study

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5+-green.svg)](https://xgboost.readthedocs.io/)

## Executive Summary

This project demonstrates end-to-end data science capabilities through a lead conversion optimization analysis for ExtraaLearn, an EdTech startup. The analysis combines rigorous statistical methods with machine learning to identify high-potential leads, resulting in actionable business recommendations with quantified ROI.

**Key Results:**
- Developed a predictive model achieving **88% recall** and **0.92 ROC-AUC**
- Identified conversion drivers with statistical validation (p < 0.001)
- Projected **$8.7M projected annual impact** through model-guided lead prioritization (~$726K/month incremental revenue, computed from model lift × observed conversion rate × estimated revenue per conversion)
- Delivered tiered strategic recommendations with clear investment/impact tradeoffs

---

## Business Context

### The Problem

ExtraaLearn faces a common challenge in high-growth companies: **lead prioritization**. With thousands of leads entering the funnel monthly, the sales team cannot contact everyone. Contacting low-probability leads wastes resources, while missing high-probability leads means lost revenue.

### The Opportunity

By building a predictive model to score leads, ExtraaLearn can:
1. Focus sales efforts on leads most likely to convert
2. Reduce customer acquisition cost (CAC)
3. Increase sales team productivity
4. Improve marketing ROI through channel optimization

### Success Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **Recall** | % of actual converters correctly identified | > 80% |
| **Precision** | % of predicted converters who actually convert | > 65% |
| **Lift** | Improvement over random selection | > 2x in top decile |
| **Incremental Revenue** | Additional revenue from model-guided prioritization | > $500K annually |

---

## Methodology

### 1. Research Framework

The analysis follows a hypothesis-driven approach:

| Hypothesis | Rationale |
|------------|-----------|
| H1: Website-first leads convert higher than mobile app | Richer content and information architecture |
| H2: Time on website positively correlates with conversion | Indicates genuine interest and engagement |
| H3: Profile completion predicts conversion | Signals commitment and intent |
| H4: Referral leads convert at higher rates | Social proof and pre-qualification effects |

### 2. Statistical Analysis

**Exploratory Data Analysis:**
- Univariate distributions with skewness/kurtosis analysis
- Bivariate relationships with 95% confidence intervals
- Correlation analysis with point-biserial coefficients

**Hypothesis Testing:**
- Chi-square tests for categorical variables with Cramér's V effect sizes
- Welch's t-tests for continuous variables with Cohen's d effect sizes
- Multiple comparison corrections where appropriate

### 3. Feature Engineering

| Feature Type | Examples | Technique |
|--------------|----------|-----------|
| Interaction terms | `professional_website` | Domain knowledge |
| Aggregations | `media_exposure_count` | Sum of binary indicators |
| Ordinal encoding | `profile_score` | Ordered categorical mapping |
| Engagement metrics | `engagement_score` | Composite of time, pages, visits |
| Outlier treatment | `website_visits` | IQR capping at 1.5x bounds |

### 4. Model Development

**Models Evaluated:**
1. Logistic Regression (interpretable baseline)
2. Decision Tree with pruning (visual interpretability)
3. Random Forest (ensemble robustness)
4. XGBoost with early stopping (maximum performance)

**Validation Strategy:**
- 70/30 stratified train-test split
- 5-fold stratified cross-validation
- RandomizedSearchCV with 50 iterations for hyperparameter tuning
- Early stopping to prevent overfitting

**Class Imbalance Handling:**
- Stratified sampling in all splits
- Class weights inversely proportional to frequency
- `scale_pos_weight` parameter for XGBoost

### 5. Model Interpretability

Since black-box models require explanation for business adoption:

- **Permutation Importance**: Model-agnostic feature importance with confidence intervals
- **Partial Dependence Plots**: Marginal effect of features on predictions
- **Feature Importance Comparison**: Cross-model validation of key drivers

---

## Key Findings

### Statistical Results

All hypotheses were supported with strong effect sizes:

| Hypothesis | Test Statistic | p-value | Effect Size |
|------------|---------------|---------|-------------|
| Website > Mobile App | χ² = 612.4 | < 0.001 | V = 0.36 (Large) |
| Time on site → Conversion | t = 28.7 | < 0.001 | d = 1.2 (Large) |
| Profile completion → Conversion | χ² = 358.2 | < 0.001 | V = 0.28 (Medium) |
| Referral → Higher conversion | χ² = 98.5 | < 0.001 | V = 0.15 (Small) |

### Model Performance

| Model | Recall | Precision | ROC-AUC | F1 |
|-------|--------|-----------|---------|-----|
| Logistic Regression (Tuned) | 83.5% | 63.0% | 0.882 | 0.72 |
| Decision Tree (Tuned) | 90.1% | 59.7% | 0.870 | 0.72 |
| Random Forest (Tuned) | 83.3% | 69.8% | 0.917 | 0.76 |
| **XGBoost (Tuned)** | **88.1%** | **66.7%** | **0.920** | **0.76** |

### Business Impact

| Contact Strategy | Leads Contacted | Conversions | Incremental vs. Random |
|------------------|-----------------|-------------|------------------------|
| Top 10% by score | 100 | 68 | +38 conversions |
| Top 30% by score | 300 | 156 | +66 conversions |
| Top 50% by score | 500 | 210 | +60 conversions |

**Projected Annual Impact**: $8.7M (~$726K/month × 12 months)

> Unit economics: model lift applied to observed conversion rate × estimated revenue per conversion × monthly lead volume. Full computation in notebook Section 8.

---

## Strategic Recommendations

### Tier 1: Quick Wins (0-30 days)

1. **Deploy Lead Scoring Model** — Rank incoming leads daily using the trained model
2. **Sales Prioritization** — Focus on top 30% of scored leads first
3. **Referral Fast-Track** — Expedited process for high-converting referral leads

### Tier 2: Product & Marketing (30-90 days)

1. **Mobile App Overhaul** — Website-first leads convert 4x higher; close the gap
2. **Progressive Profiling** — Gamify profile completion to increase engagement signals
3. **Referral Program** — Formal incentives to scale highest-converting channel

### Tier 3: Strategic Investments (90+ days)

1. **Real-Time Scoring API** — Enable trigger-based outreach
2. **A/B Testing Framework** — Validate model impact with controlled experiments
3. **Marketing Mix Optimization** — Reallocate budget from print to digital/referral

---

## Technical Implementation

### Repository Structure

```
ExtraaLearn/
├── README.md                      # This file
├── ExtraaLearn_Analysis.ipynb     # Complete analysis notebook
├── ExtraaLearn.csv                # Source data
└── requirements.txt               # Python dependencies
```

### Requirements

```
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
scipy>=1.7.0
statsmodels>=0.12.0
xgboost>=1.5.0
```

### Running the Analysis

```bash
# Clone repository
git clone https://github.com/thomasneira/thomasneira.github.io.git
cd ExtraaLearn

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook ExtraaLearn_Analysis.ipynb
```

---

## Skills Demonstrated

| Category | Skills |
|----------|--------|
| **Statistical Analysis** | Hypothesis testing, effect sizes, confidence intervals, chi-square, t-tests |
| **Machine Learning** | Classification, ensemble methods, hyperparameter tuning, cross-validation |
| **Data Engineering** | Feature engineering, outlier treatment, encoding strategies |
| **Model Interpretability** | Permutation importance, partial dependence, feature importance |
| **Business Acumen** | ROI quantification, strategic recommendations, stakeholder communication |
| **Python** | pandas, numpy, scikit-learn, XGBoost, matplotlib, seaborn, statsmodels |

---

## About the Author

**Thomas Neira**

This analysis was completed as a demonstration of applied data science capabilities, combining rigorous statistical methods with practical business application. The project showcases:

- End-to-end data science workflow
- Hypothesis-driven analysis approach
- Clear communication of technical findings to business stakeholders
- Actionable recommendations with quantified impact

---

## License

This project is for portfolio demonstration purposes. The dataset is publicly available for educational use.
