import pandas as pd
from src.utils.logger import get_logger

log = get_logger(__name__)


class DataTransformer:
    def parse_dates(self, df: pd.DataFrame, date_cols: list[str]) -> pd.DataFrame:
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return df

    def encode_categoricals(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        for col in cols:
            if col in df.columns:
                df[col] = df[col].astype("category").cat.codes
        return df

    def normalize(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        for col in cols:
            if col in df.columns:
                mn, mx = df[col].min(), df[col].max()
                if mx > mn:
                    df[col] = (df[col] - mn) / (mx - mn)
        log.info(f"Normalized columns: {cols}")
        return df

    def merge_transactions_customers(
        self, transactions: pd.DataFrame, customers: pd.DataFrame
    ) -> pd.DataFrame:
        merged = transactions.merge(customers, on="customer_id", how="left")
        log.info(f"Merged transactions+customers: {len(merged)} rows")
        return merged
