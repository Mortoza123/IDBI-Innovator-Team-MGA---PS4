# -*- coding: utf-8 -*-
"""
02_generate_synthetic_unstructured.py
---------------------------------------
Real GST filing text and bank call-center logs are not publicly available
(regulated/proprietary). This script generates REALISTIC, risk-correlated
SYNTHETIC text linked to every MSME borrower in company_bankruptcy.csv, so
the multi-modal fusion pipeline can be built and demoed end-to-end.

Borrowers already flagged 'Bankrupt' in the structured data are more
likely to get delayed-filing / stress-toned text — this creates a
realistic correlation for the model to learn from, mirroring what a real
GST/call-center feed would carry.

Run from the project root:
    python scripts/02_generate_synthetic_unstructured.py
"""
import os
import random
import pandas as pd

random.seed(42)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW_DIR = os.path.join(ROOT, "data", "raw")
PROC_DIR = os.path.join(ROOT, "data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)

GST_ONTIME_TEMPLATES = [
    "GSTR-3B filed on schedule for {month}. Turnover consistent with prior quarter.",
    "Regular GST compliance maintained. No late filing flags in {month}.",
    "GST return for {month} filed within due date. ITC claims reconciled.",
    "Steady monthly turnover reported in GST filings; no anomalies for {month}.",
]
GST_DELAY_TEMPLATES = [
    "GSTR-3B filing delayed by {days} days for {month}. Late fee applicable.",
    "Turnover declared in {month} filing dropped sharply versus prior period.",
    "GST return for {month} filed after due date; ITC mismatch flagged.",
    "Repeated late filing pattern observed for {month}; compliance risk noted.",
    "Nil/near-nil turnover reported in {month} GST return, inconsistent with loan ticket size.",
]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

CALL_NEUTRAL_TEMPLATES = [
    "Routine EMI reminder call placed. Borrower confirmed payment scheduled.",
    "Borrower reachable, no dispute raised, account in good standing.",
    "Courtesy call completed. No stress indicators noted.",
]
CALL_STRESS_TEMPLATES = [
    "Borrower requested EMI deferment citing cash flow stress.",
    "Call escalated - borrower reported delay in receivables from customers.",
    "Borrower unresponsive to last two collection calls.",
    "Complaint logged: borrower disputes penalty charges amid repayment difficulty.",
    "Field visit report notes shutdown/reduced operations at business premises.",
    "Borrower requested restructuring due to declining order book.",
]


def gen_gst_text(is_risky):
    m = random.choice(MONTHS)
    if is_risky and random.random() < 0.75:
        return random.choice(GST_DELAY_TEMPLATES).format(month=m, days=random.randint(8, 45))
    return random.choice(GST_ONTIME_TEMPLATES).format(month=m)


def gen_call_text(is_risky):
    if is_risky and random.random() < 0.7:
        return random.choice(CALL_STRESS_TEMPLATES)
    return random.choice(CALL_NEUTRAL_TEMPLATES)


def main():
    corp = pd.read_csv(os.path.join(RAW_DIR, "company_bankruptcy.csv"))
    corp.columns = [c.strip() for c in corp.columns]
    corp = corp.reset_index().rename(columns={"index": "borrower_id"})
    corp["borrower_id"] = "MSME" + corp["borrower_id"].astype(str).str.zfill(5)

    records = []
    for _, row in corp.iterrows():
        is_risky = bool(row["Bankrupt?"] == 1) or (random.random() < 0.08)  # small noise for realism
        n_gst_entries = random.randint(2, 4)
        n_call_entries = random.randint(1, 3)
        gst_texts = [gen_gst_text(is_risky) for _ in range(n_gst_entries)]
        call_texts = [gen_call_text(is_risky) for _ in range(n_call_entries)]
        records.append({
            "borrower_id": row["borrower_id"],
            "segment": "MSME",
            "gst_filing_notes": " || ".join(gst_texts),
            "collection_call_notes": " || ".join(call_texts),
            "label_bankrupt": int(row["Bankrupt?"]),
        })

    unstructured_df = pd.DataFrame(records)
    out_path = os.path.join(PROC_DIR, "msme_unstructured_text.csv")
    unstructured_df.to_csv(out_path, index=False)
    print(f"Synthetic unstructured dataset created: {unstructured_df.shape} -> {out_path}")

    # sanity check: risky borrowers should show more delay/stress language
    def flag_rate(col, keyword):
        return unstructured_df[unstructured_df[col].str.contains(keyword, case=False)]["label_bankrupt"].mean()

    print("Sanity check - P(bankrupt=1 | GST delay mentioned):",
          round(flag_rate("gst_filing_notes", "delay"), 3))
    print("Sanity check - P(bankrupt=1 | GST on-time language):",
          round(flag_rate("gst_filing_notes", "on schedule|within due date"), 3))


if __name__ == "__main__":
    main()
