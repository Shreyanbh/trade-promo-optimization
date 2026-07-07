import pandas as pd
import numpy as np
from src.utils.logger import get_logger

log = get_logger(__name__)


def compute_promo_sensitivity(
    transactions: pd.DataFrame, promos: pd.DataFrame
) -> pd.DataFrame:
    """Computes a per-customer promo sensitivity score (0–1)."""
    txn = transactions.copy()
    promo = promos.copy()
    txn["date"] = pd.to_datetime(txn["date"], errors="coerce")
    promo["start_date"] = pd.to_datetime(promo["start_date"], errors="coerce")
    promo["end_date"] = pd.to_datetime(promo["end_date"], errors="coerce")

    # tag each transaction as promo or non-promo
    promo_product_dates = {}
    for _, row in promo.iterrows():
        key = (row["product_id"], row["start_date"], row["end_date"])
        promo_product_dates[row["product_id"]] = (row["start_date"], row["end_date"])

    def is_promo(row):
        pid = row["product_id"]
        if pid not in promo_product_dates:
            return 0
        start, end = promo_product_dates[pid]
        return 1 if start <= row["date"] <= end else 0

    txn["is_promo_purchase"] = txn.apply(is_promo, axis=1)

    agg = txn.groupby("customer_id").agg(
        total_purchases=("transaction_id", "count"),
        promo_purchases=("is_promo_purchase", "sum"),
    ).reset_index()

    agg["promo_sensitivity_score"] = agg["promo_purchases"] / agg["total_purchases"].replace(0, 1)
    log.info(f"Promo sensitivity computed for {len(agg)} customers")
    return agg[["customer_id", "promo_purchases", "total_purchases", "promo_sensitivity_score"]]
