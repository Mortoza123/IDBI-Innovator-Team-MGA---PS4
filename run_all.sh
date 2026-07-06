#!/usr/bin/env bash
# Runs the full pipeline end-to-end: download -> synthetic data -> clean ->
# feature engineering -> train -> explainability.
set -e
cd "$(dirname "$0")"

echo "=== [1/6] Downloading source datasets ==="
python scripts/01_download_data.py

echo "=== [2/6] Generating synthetic unstructured MSME text ==="
python scripts/02_generate_synthetic_unstructured.py

echo "=== [3/6] Cleaning + scoring into ready datasets ==="
python scripts/03_prepare_ready_data.py

echo "=== [4/6] Feature engineering ==="
python scripts/04_feature_engineering.py

echo "=== [5/6] Training + benchmarking models ==="
python scripts/05_train_models.py

echo "=== [6/6] SHAP explainability ==="
python scripts/06_shap_explainability.py

echo ""
echo "Done. Results in ./results/, trained models in ./models/"
