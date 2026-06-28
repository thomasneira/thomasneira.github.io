# Marketing A/B Test Incrementality Engine

Goes beyond a simple t-test. Implements a full experimentation analysis pipeline with frequentist significance testing, Bayesian posterior estimation, sequential testing for early stopping, minimum detectable effect (MDE) sizing, and multiple comparison correction — the exact sophistication the JD asks for.

## Key Questions

1. **Is the treatment effect real or noise?** Frequentist p-value + Bayesian posterior probability of superiority.
2. **How large is the effect?** Posterior distribution of the treatment effect with credible intervals.
3. **Could we have stopped this test earlier?** Sequential testing boundaries (O'Brien-Fleming) overlaid on cumulative results.
4. **Was the test properly powered?** Post-hoc MDE analysis shows whether the sample was large enough to detect the observed effect.

## Datasets

- **Primary**: [Marketing A/B Testing](https://www.kaggle.com/datasets/faviovaz/marketing-ab-testing) — experiment/control groups with conversion outcomes
- **Secondary**: [A/B Test Marketing Campaign](https://www.kaggle.com/datasets/amirmotefaker/ab-testing-dataset) — additional A/B test data for multi-test correction demo

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python (scipy) | Frequentist tests (z-test, chi-square, Mann-Whitney) |
| Python (PyMC) | Bayesian A/B test modeling |
| Python (statsmodels) | Sequential testing, power analysis |
| SQL | Data preparation & metric computation |
| Streamlit | Interactive experimentation dashboard |

## Project Structure

```
ab-test-incrementality-engine/
├── README.md
├── requirements.txt
├── data/                    # CSV exports (gitignored)
│   └── README.md            # Data download instructions
├── src/
│   ├── frequentist.py       # Z-tests, chi-square, effect sizes
│   ├── bayesian.py          # PyMC model, posterior sampling
│   ├── sequential.py        # O'Brien-Fleming boundaries, early stopping
│   └── power.py             # MDE sizing, sample size calculator
└── app/
    └── streamlit_app.py     # Interactive experiment analysis dashboard
```

## JD Alignment

> **Experimentation**: Design and analyze experiments across Mindshare programs — helping the team test messaging, channels, formats, and targeting to learn what drives awareness and engagement most effectively.

> Experience designing and analyzing experiments (A/B tests, holdout groups, incrementality measurement)
