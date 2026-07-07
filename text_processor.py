"""
TextProcessor — enriches raw unstructured data extracted from reviews,
call transcripts, and emails into machine-learning-ready features.

Produces:
  - sentiment score (-1 to +1) and label (POSITIVE / NEUTRAL / NEGATIVE)
  - keyword flags per product category
  - customer_unstructured_features.parquet  (one row per customer_id)
  - product_voice_of_customer.parquet       (one row per product_id)
"""
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log = get_logger("text_processor")

# Simple lexicon-based sentiment
_POSITIVE = {
    "love", "excellent", "great", "fantastic", "outstanding", "perfect",
    "superb", "amazing", "brilliant", "wonderful", "happy", "delighted",
    "impressed", "satisfied", "recommend", "reorder", "fast", "quality",
}
_NEGATIVE = {
    "disappointed", "terrible", "awful", "poor", "damaged", "broke",
    "unhappy", "returned", "worst", "bad", "horrible", "angry", "issue",
    "problem", "complaint", "delay", "missing", "refund", "wrong", "off",
}

_STOPWORDS = {
    "the", "a", "an", "is", "it", "in", "to", "and", "of", "for", "my",
    "i", "me", "this", "that", "with", "was", "at", "on", "not", "but",
    "have", "as", "be", "by", "we", "are", "has", "will", "can", "its",
    "very", "so", "up", "do", "if", "all", "or", "had", "from", "our",
    "their", "they", "would", "could", "should", "been", "also", "just",
}

_CATEGORY_KEYWORDS = {
    "beverages":  {"drink", "tea", "coffee", "juice", "bottle", "water", "flavour", "taste", "refreshing"},
    "snacks":     {"crisps", "biscuit", "chocolate", "snack", "sweet", "crispy", "crunch", "pack"},
    "dairy":      {"milk", "cheese", "yogurt", "cream", "fresh", "butter", "dairy"},
    "produce":    {"fruit", "vegetable", "fresh", "organic", "salad", "ripe", "juicy"},
    "household":  {"clean", "spray", "soap", "wash", "product", "easy", "effective", "smell"},
}


def _tokenise(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r"[a-zA-Z]+", text) if w.lower() not in _STOPWORDS]


def _sentiment_score(tokens: list[str]) -> float:
    pos = sum(1 for t in tokens if t in _POSITIVE)
    neg = sum(1 for t in tokens if t in _NEGATIVE)
    total = pos + neg
    if total == 0:
        return 0.0
    return round((pos - neg) / total, 4)


def _sentiment_label(score: float) -> str:
    if score > 0.1:
        return "POSITIVE"
    if score < -0.1:
        return "NEGATIVE"
    return "NEUTRAL"


def _category_flags(tokens: list[str]) -> dict[str, int]:
    return {cat: int(bool(kws & set(tokens))) for cat, kws in _CATEGORY_KEYWORDS.items()}


def process_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Add sentiment score/label and category keyword flags to reviews DataFrame."""
    if df.empty:
        return df
    df = df.copy()
    tokens_col       = df["review_text"].fillna("").apply(_tokenise)
    df["sent_score"] = tokens_col.apply(_sentiment_score)
    df["sentiment"]  = df["sent_score"].apply(_sentiment_label)
    for cat in _CATEGORY_KEYWORDS:
        df[f"kw_{cat}"] = tokens_col.apply(lambda t: int(bool(_CATEGORY_KEYWORDS[cat] & set(t))))
    log.info(f"Reviews sentiment: {df['sentiment'].value_counts().to_dict()}")
    return df


def build_customer_features(reviews: pd.DataFrame,
                             transcripts: pd.DataFrame,
                             emails: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate unstructured signals per customer_id:
      - avg_review_sentiment, n_reviews, pct_negative_reviews
      - n_support_calls, n_negative_calls
      - n_support_emails, n_negative_emails
      - overall_satisfaction_score (composite -1..1)
    """
    records: dict[str, dict] = defaultdict(lambda: {
        "n_reviews": 0, "sum_sentiment": 0.0, "n_neg_reviews": 0,
        "n_calls": 0,   "n_neg_calls": 0,
        "n_emails": 0,  "n_neg_emails": 0,
    })

    if not reviews.empty and "sent_score" in reviews.columns:
        for _, row in reviews.iterrows():
            cid = row["customer_id"]
            records[cid]["n_reviews"]    += 1
            records[cid]["sum_sentiment"] += row.get("sent_score", 0)
            if row.get("sentiment") == "NEGATIVE":
                records[cid]["n_neg_reviews"] += 1

    if not transcripts.empty and "sentiment" in transcripts.columns:
        for _, row in transcripts.iterrows():
            cid = row["customer_id"]
            records[cid]["n_calls"] += 1
            if row.get("sentiment") == "NEGATIVE":
                records[cid]["n_neg_calls"] += 1

    if not emails.empty and "sentiment" in emails.columns:
        for _, row in emails.iterrows():
            cid = row["customer_id"]
            records[cid]["n_emails"] += 1
            if row.get("sentiment") == "NEGATIVE":
                records[cid]["n_neg_emails"] += 1

    rows = []
    for cid, r in records.items():
        avg_sent = r["sum_sentiment"] / r["n_reviews"] if r["n_reviews"] > 0 else 0.0
        neg_pct  = (r["n_neg_reviews"] / r["n_reviews"]) if r["n_reviews"] > 0 else 0.0
        call_neg = (r["n_neg_calls"] / r["n_calls"]) if r["n_calls"] > 0 else 0.0
        total_interactions = r["n_reviews"] + r["n_calls"] + r["n_emails"]
        satisfaction = round(avg_sent * 0.5 - neg_pct * 0.3 - call_neg * 0.2, 4)
        rows.append({
            "customer_id":             cid,
            "n_reviews":               r["n_reviews"],
            "avg_review_sentiment":    round(avg_sent, 4),
            "pct_negative_reviews":    round(neg_pct, 4),
            "n_support_calls":         r["n_calls"],
            "n_negative_calls":        r["n_neg_calls"],
            "n_support_emails":        r["n_emails"],
            "n_negative_emails":       r["n_neg_emails"],
            "total_interactions":      total_interactions,
            "satisfaction_score":      satisfaction,
        })
    df = pd.DataFrame(rows)
    log.info(f"Customer unstructured features: {len(df)} customers")
    return df


def build_product_voice_of_customer(reviews: pd.DataFrame,
                                     transcripts: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate unstructured signals per product_id:
      - avg_rating, avg_sentiment, n_mentions, n_complaints
      - category keyword prevalence
    """
    voc: dict[str, dict] = defaultdict(lambda: {
        "n_reviews": 0, "sum_rating": 0, "sum_sent": 0.0, "n_complaints": 0,
        **{f"kw_{c}": 0 for c in _CATEGORY_KEYWORDS},
    })

    if not reviews.empty:
        for _, row in reviews.iterrows():
            pid = row.get("product_id", "")
            if not pid:
                continue
            voc[pid]["n_reviews"]  += 1
            voc[pid]["sum_rating"] += row.get("rating", 3)
            voc[pid]["sum_sent"]   += row.get("sent_score", 0.0)
            if row.get("sentiment") == "NEGATIVE":
                voc[pid]["n_complaints"] += 1
            for cat in _CATEGORY_KEYWORDS:
                voc[pid][f"kw_{cat}"] += row.get(f"kw_{cat}", 0)

    if not transcripts.empty:
        for _, row in transcripts.iterrows():
            pid = row.get("product_id", "")
            if not pid:
                continue
            voc[pid]["n_reviews"] += 1  # count transcript mentions too
            if row.get("sentiment") == "NEGATIVE":
                voc[pid]["n_complaints"] += 1

    rows = []
    for pid, r in voc.items():
        n = r["n_reviews"] or 1
        rows.append({
            "product_id":       pid,
            "n_mentions":       r["n_reviews"],
            "avg_rating":       round(r["sum_rating"] / n, 3),
            "avg_sentiment":    round(r["sum_sent"] / n, 4),
            "n_complaints":     r["n_complaints"],
            "complaint_rate":   round(r["n_complaints"] / n, 4),
            **{f"kw_{c}": round(r[f"kw_{c}"] / n, 4) for c in _CATEGORY_KEYWORDS},
        })
    df = pd.DataFrame(rows)
    log.info(f"Product VoC features: {len(df)} products")
    return df
