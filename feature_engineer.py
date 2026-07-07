import pandas as pd
import numpy as np
from src.phase3.clv_calculator import compute_clv
from src.phase3.promo_sensitivity import compute_promo_sensitivity
from src.utils.logger import get_logger

log = get_logger(__name__)


class FeatureEngineer:
    def build_feature_matrix(
        self,
        customers: pd.DataFrame,
        transactions: pd.DataFrame,
        products: pd.DataFrame,
        promos: pd.DataFrame,
    ) -> pd.DataFrame:
        log.info("Building feature matrix …")

        clv_df = compute_clv(transactions, customers)
        promo_df = compute_promo_sensitivity(transactions, promos)

        txn = transactions.copy()
        txn["date"] = pd.to_datetime(txn["date"], errors="coerce")

        basket = txn.groupby("customer_id").agg(
            avg_basket_size=("amount", "mean"),
            purchase_frequency=("date", lambda x: x.nunique()),
        ).reset_index()

        category_spend = (
            txn.merge(products[["product_id", "category"]], on="product_id", how="left")
            .groupby(["customer_id", "category"])["amount"]
            .sum()
            .unstack(fill_value=0)
            .reset_index()
        )
        category_spend.columns = [
            f"spend_{c}" if c != "customer_id" else c
            for c in category_spend.columns
        ]

        feat = (
            clv_df
            .merge(promo_df, on="customer_id", how="left")
            .merge(basket, on="customer_id", how="left")
            .merge(customers[["customer_id", "age", "region"]], on="customer_id", how="left")
            .merge(category_spend, on="customer_id", how="left")
        )

        feat["region"] = feat["region"].astype("category").cat.codes
        feat = feat.fillna(0)
        log.info(f"Feature matrix: {feat.shape[0]} customers x {feat.shape[1]} features")
        return feat
