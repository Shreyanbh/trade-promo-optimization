import pandas as pd
from src.utils.logger import get_logger

log = get_logger(__name__)


class DataCleaner:
    def clean(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        original = len(df)
        df = df.drop_duplicates()
        df = df.dropna(how="all")

        num_cols = df.select_dtypes(include="number").columns.tolist()
        for col in num_cols:
            df[col] = df[col].fillna(df[col].median())

        cat_cols = df.select_dtypes(include="object").columns.tolist()
        for col in cat_cols:
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "unknown")

        log.info(f"[{dataset_name}] cleaned: {original} -> {len(df)} rows")
        return df
