import os
import pandas as pd
from pathlib import Path
from src.utils.file_helpers import read_data, write_data, ensure_dirs
from src.utils.logger import get_logger
from src.config.settings import PATHS

log = get_logger(__name__)


class DataIngestor:
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or PATHS["processed_data"]
        ensure_dirs(self.output_dir)

    def ingest(self, source_path: str, dataset_name: str) -> tuple[pd.DataFrame, dict]:
        log.info(f"Ingesting {dataset_name} from {source_path}")
        df = read_data(source_path)
        meta = {
            "dataset_name": dataset_name,
            "source_path":  source_path,
            "rows":         len(df),
            "columns":      list(df.columns),
            "dtypes":       df.dtypes.astype(str).to_dict(),
            "null_counts":  df.isnull().sum().to_dict(),
        }
        log.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        return df, meta

    def save_processed(self, df: pd.DataFrame, dataset_name: str) -> str:
        out_path = os.path.join(self.output_dir, f"{dataset_name}.parquet")
        write_data(df, out_path, fmt="parquet")
        return out_path

    def generate_synthetic_data(self) -> dict[str, pd.DataFrame]:
        import numpy as np
        np.random.seed(42)
        n_customers = 500
        n_products = 50
        n_transactions = 5000

        customer_ids = [f"C{i:04d}" for i in range(n_customers)]
        product_ids  = [f"P{i:03d}" for i in range(n_products)]
        categories   = ["beverages", "snacks", "dairy", "produce", "household"]

        customers = pd.DataFrame({
            "customer_id": customer_ids,
            "age":         np.random.randint(18, 70, n_customers),
            "region":      np.random.choice(["north", "south", "east", "west"], n_customers),
            "signup_date": pd.date_range("2020-01-01", periods=n_customers, freq="D").strftime("%Y-%m-%d"),
        })

        products = pd.DataFrame({
            "product_id":   product_ids,
            "product_name": [f"Product_{i}" for i in range(n_products)],
            "category":     np.random.choice(categories, n_products),
            "price":        np.round(np.random.uniform(1.0, 50.0, n_products), 2),
        })

        transactions = pd.DataFrame({
            "transaction_id": [f"T{i:06d}" for i in range(n_transactions)],
            "customer_id":    np.random.choice(customer_ids, n_transactions),
            "product_id":     np.random.choice(product_ids, n_transactions),
            "date":           pd.date_range("2023-01-01", periods=n_transactions, freq="1h").strftime("%Y-%m-%d"),
            "amount":         np.round(np.random.uniform(1.0, 200.0, n_transactions), 2),
            "quantity":       np.random.randint(1, 10, n_transactions),
        })

        promos = pd.DataFrame({
            "promo_id":     [f"PR{i:03d}" for i in range(20)],
            "product_id":   np.random.choice(product_ids, 20),
            "discount_pct": np.round(np.random.uniform(0.05, 0.40, 20), 2),
            "start_date":   pd.date_range("2023-03-01", periods=20, freq="15D").strftime("%Y-%m-%d"),
            "end_date":     pd.date_range("2023-03-15", periods=20, freq="15D").strftime("%Y-%m-%d"),
        })

        log.info("Generated synthetic dataset")
        return {
            "customers":    customers,
            "products":     products,
            "transactions": transactions,
            "promos":       promos,
        }
