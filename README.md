# Default Prediction Model — IDBI Innovate PS 4

AI/ML early-warning system that predicts probability of default (PD) 12 months
in advance by fusing structured financial data with unstructured signals
(GST filings, news, complaint/call logs), using segment-specific models
unified under one SHAP-based explainable risk score.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.9+.

## Run everything

```bash
chmod +x run_all.sh
./run_all.sh
```

Or run each stage individually (in order — each stage depends on the previous one's output):

```bash
python scripts/01_download_data.py                    # pulls 3 public datasets via GitHub mirrors
python scripts/02_generate_synthetic_unstructured.py   # builds risk-correlated synthetic GST/call-log text
python scripts/03_prepare_ready_data.py                # cleans, imputes, tags, scores sentiment
python scripts/04_feature_engineering.py                # derived features + correlation/MI feature selection
python scripts/05_train_models.py                       # trains 3 XGBoost models, benchmarks them
python scripts/06_shap_explainability.py                 # SHAP global importance + per-borrower explanations
```

## What each dataset is

| Dataset | Real or Synthetic | Simulates |
|---|---|---|
| `cs-training.csv` (Give Me Some Credit) | Real (public) | Retail structured data |
| `company_bankruptcy.csv` (Taiwan Economic Journal) | Real (public) | MSME/Corporate structured data — 95 financial ratios |
| `financial_news_sentiment.csv` (Financial PhraseBank) | Real (public) | Unstructured news-sentiment signal |
| `msme_unstructured_text.csv` | **Synthetic**, risk-correlated, linked to real MSME borrower IDs | GST filing notes + collection call/complaint logs |

Real bank-internal unstructured data (actual GST filings, call transcripts) is
regulated and not publicly available — the synthetic generator is deliberately
built so struggling borrowers are more likely to show late-filing/stress
language, mirroring what a real feed would carry. This is disclosed
explicitly, not hidden.

## Pipeline stages

1. **Download** — pulls the 3 real public datasets from GitHub mirrors (no Kaggle account needed).
2. **Synthetic generation** — builds the unstructured MSME text layer.
3. **Data prep** — imputation, outlier clipping, ID/segment tagging, VADER sentiment scoring.
4. **Feature engineering** —
   - Retail: log transforms, delinquency totals, affordability ratios, binned utilization/age (15 new features)
   - MSME: 95 raw financial ratios reduced to 30 via variance filter → correlation filter (drops |r|>0.95 pairs) → mutual-information ranking
5. **Model training** — 3 segment-aware XGBoost models: MSME structured-only (baseline), MSME fused (structured+unstructured), Retail structured. Reports recall, AUC-ROC, KS-statistic, precision, F1, accuracy.
6. **Explainability** — SHAP TreeExplainer → global feature importance + per-borrower 0-100 risk score with top-3 plain-language reasons (the PS's "common interpretation framework").

## Key result to know

Fusing unstructured GST/call-log signals into the MSME model lifts **recall
from ~62% to ~80%** and **AUC-ROC from ~94% to ~99%** vs. the structured-only
baseline. `gst_delay_flag` (unstructured) is the single strongest global risk
driver, ahead of every financial ratio — direct evidence for the PS's
central thesis that structured+unstructured fusion beats structured-only
scoring.

**Important:** raw accuracy is a misleading metric here — MSME default rate
is only ~3.2%, so a model predicting "no one defaults" already scores ~97%
accuracy. Lead with recall/AUC/KS in any pitch, not raw accuracy.

## Honest limitations

- Trained on public + synthetic data; real bank-scale data is expected to close the gap toward the 90% target.
- MSME test set is small (~1,700 borrowers, ~55 actual defaulters) — read results directionally.
- The Corporate/sequence-model segment (LSTM on financial+text trends) from the full architecture isn't trained here — no suitable public time-series-labeled dataset was available. Would need real bank data or a synthetic panel-data generator to build.
- Retail model's precision is low at the default 0.5 threshold — tune against the bank's actual cost-of-false-positive vs. cost-of-missed-default tradeoff before production use.

## Project structure

```
default-prediction-model/
├── requirements.txt
├── run_all.sh
├── scripts/
│   ├── 01_download_data.py
│   ├── 02_generate_synthetic_unstructured.py
│   ├── 03_prepare_ready_data.py
│   ├── 04_feature_engineering.py
│   ├── 05_train_models.py
│   └── 06_shap_explainability.py
├── data/
│   ├── raw/          (created by script 01)
│   ├── processed/    (created by script 02)
│   └── ready/         (created by scripts 03-04)
├── models/            (created by script 05 — saved .json XGBoost models)
└── results/           (created by scripts 05-06 — benchmark CSVs, SHAP outputs)
```
