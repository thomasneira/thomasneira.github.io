# Data Download Instructions

```bash
# Install Kaggle CLI if needed
pip install kaggle

# Dataset 1: Marketing A/B Testing (primary)
kaggle datasets download -d faviovaz/marketing-ab-testing -p .
unzip marketing-ab-testing.zip -d .

# Dataset 2: A/B Testing Dataset (secondary — for multi-test correction demo)
kaggle datasets download -d amirmotefaker/ab-testing-dataset -p .
unzip ab-testing-dataset.zip -d .
```

## Expected Files

- `marketing_AB.csv` — Primary A/B test with experiment/control groups
- `ab_data.csv` — Secondary A/B test data
