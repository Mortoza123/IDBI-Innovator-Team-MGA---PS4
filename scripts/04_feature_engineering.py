# -*- coding: utf-8 -*-
"""
04_feature_engineering.py
----------------------------
Retail: adds derived/interaction features (log transforms, delinquency
totals, affordability ratios, binned utilization/age).

MSME: reduces the 95 raw financial ratios via variance filter -> correlation
filter (drops one of every pair with |r| > 0.95) -> mutual-information
ranking, keeping the top 30. Then fuses the selected structured features
with the unstructured (GST/call-log) features into one model-ready table.

Produces:
    06_retail_features_engineered.csv
    07_msme_features_selected.csv
    08_msme_fused_engineered.csv

Run from the project root (after 03_prepare_ready_data.py):
    python scripts/04_feature_engineering.py
"""
import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
READY = os.path.join(ROOT, "data", "ready")


def engineer_retail():
    retail = pd.read_csv(f"{READY}/01_retail_structured_clean.csv")

    retail["log_monthly_income"] = np.log1p(retail["MonthlyIncome"])
    retail["log_debt_ratio"] = np.log1p(retail["DebtRatio"])
    retail["total_times_past_due"] = (
        retail["NumberOfTime30-59DaysPastDueNotWorse"]
        + retail["NumberOfTime60-89DaysPastDueNotWorse"]
        + retail["NumberOfTimes90DaysLate"]
    )
    retail["has_dependents"] = (retail["NumberOfDependents"] > 0).astype(int)
    retail["income_per_dependent"] = retail["MonthlyIncome"] / (retail["NumberOfDependents"] + 1)

    # fix known data-quality quirk: a small number of rows have age=0 in this public dataset
    retail.loc[retail["age"] < 18, "age"] = retail["age"].median()
    retail["credit_lines_per_age"] = retail["NumberOfOpenCreditLinesAndLoans"] / retail["age"]

    retail["utilization_bucket"] = pd.cut(
        retail["RevolvingUtilizationOfUnsecuredLines"],
        bins=[-0.01, 0.3, 0.7, 1.0, 2.01], labels=["low", "moderate", "high", "maxed"]
    )
    retail["age_bucket"] = pd.cut(
        retail["age"], bins=[0, 30, 45, 60, 120], labels=["under30", "30to45", "45to60", "60plus"]
    )
    retail["real_estate_flag"] = (retail["NumberRealEstateLoansOrLines"] > 0).astype(int)

    retail = pd.get_dummies(retail, columns=["utilization_bucket", "age_bucket"], prefix=["util", "age"])

    retail.to_csv(f"{READY}/06_retail_features_engineered.csv", index=False)
    print("Retail engineered:", retail.shape)


def select_msme_features():
    msme = pd.read_csv(f"{READY}/02_msme_structured_clean.csv")
    id_cols = ["borrower_id", "loan_segment", "default_flag"]
    X = msme.drop(columns=id_cols)
    y = msme["default_flag"]

    vt = VarianceThreshold(threshold=1e-5)
    vt.fit(X)
    X = X.loc[:, vt.get_support()]
    print(f"After variance filter: {X.shape[1]} features (from {msme.shape[1] - 3})")

    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
    X = X.drop(columns=to_drop)
    print(f"After correlation filter (dropped {len(to_drop)}): {X.shape[1]} features")

    mi = mutual_info_classif(X, y, random_state=42)
    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    top_features = mi_series.head(30).index.tolist()
    print("\nTop 10 features by mutual information:")
    print(mi_series.head(10).round(4))

    msme_selected = msme[id_cols + top_features]
    msme_selected.to_csv(f"{READY}/07_msme_features_selected.csv", index=False)
    print("\nMSME selected:", msme_selected.shape)
    return msme_selected, mi_series


def fuse_msme(msme_selected, mi_series):
    unstruct = pd.read_csv(f"{READY}/04_msme_unstructured_features.csv")
    unstruct_feats = unstruct[["borrower_id", "gst_sentiment", "call_sentiment",
                                "gst_delay_flag", "call_stress_flag", "unstructured_risk_score"]]

    fused = msme_selected.merge(unstruct_feats, on="borrower_id", how="left")

    top_struct_feat = mi_series.index[0]
    fused["structured_x_unstructured_risk"] = fused[top_struct_feat] * fused["unstructured_risk_score"]

    fused.to_csv(f"{READY}/08_msme_fused_engineered.csv", index=False)
    print("MSME fused engineered:", fused.shape)


def main():
    engineer_retail()
    msme_selected, mi_series = select_msme_features()
    fuse_msme(msme_selected, mi_series)

    print("\nAll ready files now:")
    for f in sorted(os.listdir(READY)):
        print(" -", f)


if __name__ == "__main__":
    main()
