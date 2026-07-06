# -*- coding: utf-8 -*-
"""
03_prepare_ready_data.py
--------------------------
Cleans the raw structured datasets (imputation, outlier clipping, ID/segment
tagging) and scores the unstructured text (VADER sentiment) into model-ready
CSVs. Produces:
    01_retail_structured_clean.csv
    02_msme_structured_clean.csv
    03_news_sentiment_scored.csv
    04_msme_unstructured_features.csv
    05_msme_fused_model_ready.csv   (structured + unstructured, no feature selection yet)
    00_data_dictionary.csv

Run from the project root (after 01_download_data.py and
02_generate_synthetic_unstructured.py):
    python scripts/03_prepare_ready_data.py
"""
import os
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed")
READY = os.path.join(ROOT, "data", "ready")
os.makedirs(READY, exist_ok=True)

sia = SentimentIntensityAnalyzer()


def prepare_retail():
    retail = pd.read_csv(f"{RAW}/cs-training.csv", index_col=0)
    retail["MonthlyIncome"] = retail["MonthlyIncome"].fillna(retail["MonthlyIncome"].median())
    retail["NumberOfDependents"] = retail["NumberOfDependents"].fillna(0)

    for col in ["NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTime60-89DaysPastDueNotWorse",
                "NumberOfTimes90DaysLate"]:
        retail[col] = retail[col].clip(upper=retail[col].quantile(0.999))
    retail["DebtRatio"] = retail["DebtRatio"].clip(upper=retail["DebtRatio"].quantile(0.995))
    retail["RevolvingUtilizationOfUnsecuredLines"] = retail["RevolvingUtilizationOfUnsecuredLines"].clip(upper=2.0)

    retail = retail.rename(columns={"SeriousDlqin2yrs": "default_flag"})
    retail.insert(0, "borrower_id", ["RET" + str(i).zfill(6) for i in range(len(retail))])
    retail.insert(1, "loan_segment", "Retail")

    retail.to_csv(f"{READY}/01_retail_structured_clean.csv", index=False)
    print("Retail clean:", retail.shape)
    return retail


def prepare_msme():
    msme = pd.read_csv(f"{RAW}/company_bankruptcy.csv")
    msme.columns = [c.strip() for c in msme.columns]
    msme = msme.rename(columns={"Bankrupt?": "default_flag"})
    msme = msme.reset_index().rename(columns={"index": "row_id"})
    msme.insert(0, "borrower_id", "MSME" + msme["row_id"].astype(str).str.zfill(5))
    msme.insert(1, "loan_segment", "MSME")
    msme = msme.drop(columns=["row_id"])

    msme.to_csv(f"{READY}/02_msme_structured_clean.csv", index=False)
    print("MSME structured clean:", msme.shape)
    return msme


def prepare_news():
    news = pd.read_csv(f"{RAW}/financial_news_sentiment.csv")
    news.columns = ["sentiment_label", "headline"]
    news["vader_compound"] = news["headline"].apply(lambda t: sia.polarity_scores(str(t))["compound"])
    news["vader_pos"] = news["headline"].apply(lambda t: sia.polarity_scores(str(t))["pos"])
    news["vader_neg"] = news["headline"].apply(lambda t: sia.polarity_scores(str(t))["neg"])
    news.insert(0, "news_id", ["NEWS" + str(i).zfill(5) for i in range(len(news))])

    news.to_csv(f"{READY}/03_news_sentiment_scored.csv", index=False)
    print("News sentiment scored:", news.shape)
    return news


def prepare_msme_unstructured():
    unstruct = pd.read_csv(f"{PROC}/msme_unstructured_text.csv")

    def score_text(t):
        return sia.polarity_scores(str(t))["compound"]

    unstruct["gst_sentiment"] = unstruct["gst_filing_notes"].apply(score_text)
    unstruct["call_sentiment"] = unstruct["collection_call_notes"].apply(score_text)

    unstruct["gst_delay_flag"] = unstruct["gst_filing_notes"].str.contains(
        "delay|after due date|nil/near-nil|repeated late", case=False).astype(int)
    unstruct["call_stress_flag"] = unstruct["collection_call_notes"].str.contains(
        "deferment|escalated|unresponsive|dispute|shutdown|restructuring", case=False).astype(int)

    unstruct["unstructured_risk_score"] = (
        (1 - (unstruct["gst_sentiment"] + 1) / 2) * 0.5
        + (1 - (unstruct["call_sentiment"] + 1) / 2) * 0.3
        + unstruct["gst_delay_flag"] * 0.1
        + unstruct["call_stress_flag"] * 0.1
    ).round(4)

    unstruct_out = unstruct[[
        "borrower_id", "segment", "gst_sentiment", "call_sentiment",
        "gst_delay_flag", "call_stress_flag", "unstructured_risk_score",
        "gst_filing_notes", "collection_call_notes", "label_bankrupt"
    ]]
    unstruct_out.to_csv(f"{READY}/04_msme_unstructured_features.csv", index=False)
    print("MSME unstructured features:", unstruct_out.shape)
    return unstruct_out


def build_data_dictionary():
    dict_rows = [
        ("01_retail_structured_clean.csv", "borrower_id", "Unique retail borrower ID (synthetic)"),
        ("01_retail_structured_clean.csv", "loan_segment", "Always 'Retail'"),
        ("01_retail_structured_clean.csv", "default_flag", "Target: 1 = serious delinquency within 2yrs, 0 = none"),
        ("01_retail_structured_clean.csv", "RevolvingUtilizationOfUnsecuredLines",
         "Credit card & line balance / credit limits, clipped at 2.0"),
        ("01_retail_structured_clean.csv", "age", "Borrower age in years"),
        ("01_retail_structured_clean.csv", "DebtRatio",
         "Monthly debt payments / monthly gross income, outlier-clipped"),
        ("01_retail_structured_clean.csv", "MonthlyIncome", "Monthly income; missing values imputed with median"),
        ("01_retail_structured_clean.csv", "NumberOfOpenCreditLinesAndLoans", "Count of open loans/credit lines"),
        ("01_retail_structured_clean.csv", "NumberOfTimes90DaysLate", "Times borrower was 90+ days late"),
        ("01_retail_structured_clean.csv", "NumberRealEstateLoansOrLines", "Count of mortgage/real estate loans"),
        ("01_retail_structured_clean.csv", "NumberOfDependents", "Dependents; missing imputed with 0"),
        ("02_msme_structured_clean.csv", "borrower_id",
         "Unique MSME/corporate borrower ID (synthetic, links to files 04 & 05)"),
        ("02_msme_structured_clean.csv", "loan_segment", "Always 'MSME'"),
        ("02_msme_structured_clean.csv", "default_flag", "Target: 1 = bankrupt, 0 = healthy"),
        ("02_msme_structured_clean.csv", "[93 financial ratio columns]",
         "ROA, operating margin, liquidity/solvency ratios etc. - raw Taiwan Economic Journal features, no nulls"),
        ("03_news_sentiment_scored.csv", "news_id", "Unique headline ID"),
        ("03_news_sentiment_scored.csv", "sentiment_label",
         "Human-annotated label: positive/negative/neutral (Financial PhraseBank)"),
        ("03_news_sentiment_scored.csv", "headline", "Financial news sentence/headline text"),
        ("03_news_sentiment_scored.csv", "vader_compound", "VADER compound sentiment score, range -1 to +1"),
        ("03_news_sentiment_scored.csv", "vader_pos / vader_neg", "VADER positive/negative sub-scores, 0 to 1"),
        ("04_msme_unstructured_features.csv", "borrower_id", "Links to files 02 & 05"),
        ("04_msme_unstructured_features.csv", "gst_sentiment / call_sentiment",
         "VADER compound score of GST filing notes / call notes"),
        ("04_msme_unstructured_features.csv", "gst_delay_flag",
         "1 if GST notes mention late/delayed filing"),
        ("04_msme_unstructured_features.csv", "call_stress_flag",
         "1 if call notes mention deferment/escalation/dispute etc."),
        ("04_msme_unstructured_features.csv", "unstructured_risk_score",
         "Composite 0-1 risk score blended from sentiment + flags (higher = riskier)"),
        ("04_msme_unstructured_features.csv", "gst_filing_notes / collection_call_notes",
         "Raw synthetic text (concatenated entries, '||' separated)"),
        ("05_msme_fused_model_ready.csv", "[all columns from 02 + engineered columns from 04]",
         "Final fused table - one row per MSME borrower"),
    ]
    data_dict = pd.DataFrame(dict_rows, columns=["file", "column", "description"])
    data_dict.to_csv(f"{READY}/00_data_dictionary.csv", index=False)
    print("Data dictionary rows:", len(data_dict))


def main():
    retail = prepare_retail()
    msme = prepare_msme()
    prepare_news()
    unstruct_out = prepare_msme_unstructured()

    fused = msme.merge(unstruct_out.drop(columns=["segment", "label_bankrupt"]), on="borrower_id", how="left")
    fused.to_csv(f"{READY}/05_msme_fused_model_ready.csv", index=False)
    print("MSME fused (structured+unstructured):", fused.shape)

    build_data_dictionary()

    print("\nAll ready files:")
    for f in sorted(os.listdir(READY)):
        print(" -", f)


if __name__ == "__main__":
    main()
