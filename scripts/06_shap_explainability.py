# -*- coding: utf-8 -*-
"""
06_shap_explainability.py
-----------------------------
Builds the "common interpretation framework" required by the problem
statement: SHAP-based global feature importance + per-borrower 0-100 risk
score with plain-language top-3 reasons.

Run from the project root (after 05_train_models.py):
    python scripts/06_shap_explainability.py
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS = os.path.join(ROOT, "models")
RESULTS = os.path.join(ROOT, "results")


def main():
    model = xgb.XGBClassifier()
    model.load_model(f"{MODELS}/msme_fused.json")

    with open(f"{MODELS}/test_data_fused.pkl", "rb") as f:
        X_test, y_test, y_prob = pickle.load(f)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    global_importance = pd.Series(mean_abs_shap, index=X_test.columns).sort_values(ascending=False)
    global_importance.to_csv(f"{RESULTS}/shap_global_feature_importance.csv", header=["mean_abs_shap"])
    print("Top 10 global drivers of MSME default risk (SHAP):")
    print(global_importance.head(10).round(4))

    risk_score_0_100 = pd.Series(y_prob).rank(pct=True) * 100

    sample_idx = list(range(min(15, len(X_test))))
    explain_rows = []
    for i in sample_idx:
        row_shap = shap_values[i]
        top3_idx = np.argsort(-np.abs(row_shap))[:3]
        reasons = []
        for j in top3_idx:
            feat = X_test.columns[j]
            direction = "increases" if row_shap[j] > 0 else "decreases"
            reasons.append(f"{feat} ({direction} risk, impact={row_shap[j]:.3f})")
        explain_rows.append({
            "row_index": i,
            "predicted_default_prob": round(float(y_prob[i]), 4),
            "unified_risk_score_0_100": round(float(risk_score_0_100.iloc[i]), 1),
            "actual_label": int(y_test.iloc[i]) if hasattr(y_test, "iloc") else int(y_test[i]),
            "top_3_reasons": " | ".join(reasons),
        })

    explain_df = pd.DataFrame(explain_rows)
    explain_df.to_csv(f"{RESULTS}/sample_borrower_explanations.csv", index=False)
    print("\nSample borrower-level explanations (first 5):")
    print(explain_df.head(5).to_string(index=False))

    summary = {
        "top_10_global_drivers": global_importance.head(10).round(4).to_dict(),
        "explainability_method": "SHAP TreeExplainer on XGBoost fused model",
        "risk_score_scale": "0-100 percentile-calibrated across the MSME portfolio; "
                             "same scale can be applied per-segment for cross-segment comparability",
        "note": "This SHAP layer is the 'common interpretation framework' required by the PS - "
                "every prediction, regardless of segment/model, resolves to the same 0-100 score "
                "+ a ranked list of plain-language reasons.",
    }
    with open(f"{RESULTS}/shap_explainability_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved: shap_global_feature_importance.csv, sample_borrower_explanations.csv, "
          "shap_explainability_summary.json")


if __name__ == "__main__":
    main()
