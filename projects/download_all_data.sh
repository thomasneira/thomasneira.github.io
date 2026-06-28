#!/bin/bash
# Downloads all datasets for the portfolio projects.
# Requires: pip install kaggle && kaggle API key at ~/.kaggle/kaggle.json
# Get your key from: https://www.kaggle.com/settings → "Create New Token"

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Project 3: GA4 Funnel Analysis ==="
cd "$SCRIPT_DIR/ga4-funnel-analysis/data"
kaggle datasets download -d pdaasha/ga4-obfuscated-sample-ecommerce-jan2021 -p . --unzip
echo "Done."

echo ""
echo "=== Project 7: Synthetic Control Campaign Lift ==="
cd "$SCRIPT_DIR/synthetic-control-campaign-lift/data"
kaggle datasets download -d manishabhatt22/marketing-campaign-performance-dataset -p . --unzip
echo "Done."

echo ""
echo "=== Project 4: A/B Test Incrementality Engine ==="
cd "$SCRIPT_DIR/ab-test-incrementality-engine/data"
kaggle datasets download -d faviovaz/marketing-ab-testing -p . --unzip
kaggle datasets download -d amirmotefaker/ab-testing-dataset -p . --unzip
echo "Done."

echo ""
echo "=== All datasets downloaded ==="
echo "You can now run each project's Streamlit app:"
echo "  cd ga4-funnel-analysis && streamlit run app/streamlit_app.py"
echo "  cd synthetic-control-campaign-lift && streamlit run app/streamlit_app.py"
echo "  cd ab-test-incrementality-engine && streamlit run app/streamlit_app.py"
