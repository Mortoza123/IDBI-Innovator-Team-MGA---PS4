# -*- coding: utf-8 -*-
"""
05_train_models.py
---------------------
Trains three segment-aware XGBoost models:
    1. MSME - structured only (baseline)
    2. MSME - fused (structured + unstructured)   <-- the key comparison
    3. Retail - structured

Reports recall, AUC-ROC, KS-statistic, precision, F1 and accuracy (with a
caveat: accuracy is a weak metric on this imbalanced data, recall/AUC/KS
matter more).

Run from the project root (after 04_feature_engineering.py):
    python scripts/05_train_models.py
"""
import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, roc_curve
import xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
READY = os.path.join(ROOT, "data", "ready")
RESULTS = os.path.join(ROOT, "results")
MODELS = os.path.join(ROOT, "models")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)


def ks_statistic(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return round(max(tpr - fpr) * 100, 2)


def train_eval(X, y, label, params=None):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    default_params = dict(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight, eval_metric="auc",
        random_state=42, n_jobs=-1
    )
    if params:
        default_params.update(params)
    model = xgb.XGBClassifier(**default_params)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "label": label,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": X.shape[1],
        "default_rate_test": round(y_test.mean() * 100, 2),
        "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
        "auc_roc": round(roc_auc_score(y_test, y_prob) * 100, 2),
        "ks_statistic": ks_statistic(y_test, y_prob),
        "precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        "recall": round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
        "f1": round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
    }
    return model, metrics, (X_test, y_test, y_prob)


def main():
    all_metrics = []

    # 1. MSME structured-only baseline
    msme_struct = pd.read_csv(f"{READY}/07_msme_features_selected.csv")
    X_struct = msme_struct.drop(columns=["borrower_id", "loan_segment", "default_flag"])
    y_struct = msme_struct["default_flag"]
    model_struct, metrics_struct, _ = train_eval(X_struct, y_struct, "MSME - Structured Only (baseline)")
    all_metrics.append(metrics_struct)
    model_struct.save_model(f"{MODELS}/msme_structured_only.json")

    # 2. MSME fused
    msme_fused = pd.read_csv(f"{READY}/08_msme_fused_engineered.csv")
    X_fused = msme_fused.drop(columns=["borrower_id", "loan_segment", "default_flag"])
    y_fused = msme_fused["default_flag"]
    model_fused, metrics_fused, test_data_fused = train_eval(X_fused, y_fused, "MSME - Fused (structured + unstructured)")
    all_metrics.append(metrics_fused)
    model_fused.save_model(f"{MODELS}/msme_fused.json")

    # 3. Retail structured
    retail = pd.read_csv(f"{READY}/06_retail_features_engineered.csv")
    X_retail = retail.drop(columns=["borrower_id", "loan_segment", "default_flag"])
    y_retail = retail["default_flag"]
    model_retail, metrics_retail, test_data_retail = train_eval(X_retail, y_retail, "Retail - Structured (XGBoost)")
    all_metrics.append(metrics_retail)
    model_retail.save_model(f"{MODELS}/retail_structured.json")

    results_df = pd.DataFrame(all_metrics)
    results_df.to_csv(f"{RESULTS}/model_benchmark_results.csv", index=False)
    print(results_df.to_string(index=False))

    lift_auc = metrics_fused["auc_roc"] - metrics_struct["auc_roc"]
    lift_acc = metrics_fused["accuracy"] - metrics_struct["accuracy"]
    lift_ks = metrics_fused["ks_statistic"] - metrics_struct["ks_statistic"]
    lift_recall = metrics_fused["recall"] - metrics_struct["recall"]
    print(f"\n>>> Unstructured data fusion lift on MSME segment:")
    print(f"    Recall: +{lift_recall:.2f} pts | AUC-ROC: +{lift_auc:.2f} pts | "
          f"Accuracy: +{lift_acc:.2f} pts | KS: +{lift_ks:.2f} pts")

    with open(f"{RESULTS}/fusion_lift_summary.json", "w") as f:
        json.dump({
            "structured_only": metrics_struct,
            "fused": metrics_fused,
            "recall_lift": round(lift_recall, 2),
            "auc_lift": round(lift_auc, 2),
            "accuracy_lift": round(lift_acc, 2),
            "ks_lift": round(lift_ks, 2),
        }, f, indent=2)

    with open(f"{MODELS}/test_data_fused.pkl", "wb") as f:
        pickle.dump(test_data_fused, f)
    with open(f"{MODELS}/test_data_retail.pkl", "wb") as f:
        pickle.dump(test_data_retail, f)

    print("\nSaved: results/model_benchmark_results.csv, results/fusion_lift_summary.json")
    print("Saved models to:", MODELS)


if __name__ == "__main__":
    main()
