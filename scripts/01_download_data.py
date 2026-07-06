# -*- coding: utf-8 -*-
"""
01_download_data.py
--------------------
Downloads the three public source datasets used by this project via
GitHub mirrors (works without a Kaggle account / API key).

Run from the project root:
    python scripts/01_download_data.py
"""
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW_DIR = os.path.join(ROOT, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

FILES = {
    # Retail structured data — "Give Me Some Credit" (Kaggle 2011), 150k borrowers
    "cs-training.csv":
        "https://raw.githubusercontent.com/JLZml/Credit-Scoring-Data-Sets/master/"
        "3.%20Kaggle/Give%20Me%20Some%20Credit/cs-training.csv",

    # MSME/Corporate structured data — Taiwan Economic Journal bankruptcy dataset, 6819 firms x 95 ratios
    "company_bankruptcy.csv":
        "https://raw.githubusercontent.com/SayamAlt/Company-Bankruptcy-Prediction/main/data.csv",

    # Unstructured news sentiment — Financial PhraseBank (Malo et al., 2014), 4846 headlines
    "financial_news_sentiment_raw.csv":
        "https://raw.githubusercontent.com/isaaccs/sentiment-analysis-for-financial-news/master/all-data.csv",
}


def download(name, url):
    dest = os.path.join(RAW_DIR, name)
    print(f"Downloading {name} ...")
    urllib.request.urlretrieve(url, dest)
    size_kb = os.path.getsize(dest) / 1024
    print(f"  -> saved to {dest} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    for fname, url in FILES.items():
        download(fname, url)

    # financial_news_sentiment_raw.csv is Latin-1 encoded with no header — normalize it
    import pandas as pd
    src = os.path.join(RAW_DIR, "financial_news_sentiment_raw.csv")
    df = pd.read_csv(src, encoding="ISO-8859-1", header=None, names=["sentiment", "headline"])
    df.to_csv(os.path.join(RAW_DIR, "financial_news_sentiment.csv"), index=False)
    os.remove(src)
    print("\nAll raw datasets downloaded to data/raw/")
