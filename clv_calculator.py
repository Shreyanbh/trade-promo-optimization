import pandas as pd
import numpy as np
from src.utils.logger import get_logger

log = get_logger(__name__)


def compute_clv(transactions: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """Simple BG/NBD-inspired CLV approximation using historical transaction data."""
    txn = transactions.copy()
    txn["date"] = pd.to_datetime(txn["date"], errors="coerce")
    snapshot = txn["date"].max()

    rfm = txn.groupby("customer_id").agg(
        last_purchase=("date", "max"),
        frequency=("transaction_id", "count"),
        monetary=("amount", "sum"),
    ).reset_index()

    rfm["recency_days"] = (snapshot - rfm["last_purchase"]).dt.days
    rfm["avg_order_value"] = rfm["monetary"] / rfm["frequency"]

    # Simplified CLV: monetary * purchase_rate * expected_lifespan_weight
    rfm["purchase_rate"] = rfm["frequency"] / rfm["recency_days"].replace(0, 1)
    rfm["clv_score"] = (rfm["avg_order_value"] * rfm["frequency"] *
                        np.log1p(rfm["purchase_rate"]))
    rfm["clv_score"] = (rfm["clv_score"] - rfm["clv_score"].min()) / (
        rfm["clv_score"].max() - rfm["clv_score"].min() + 1e-9
    )

    log.info(f"CLV computed for {len(rfm)} customers")
    return rfm[["customer_id", "recency_days", "frequency", "monetary", "avg_order_value", "clv_score"]]
