import json
import os
import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger

log = get_logger(__name__)


def read_data(path: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    readers = {
        ".csv":     pd.read_csv,
        ".parquet": pd.read_parquet,
        ".xlsx":    pd.read_excel,
        ".xls":     pd.read_excel,
        ".json":    pd.read_json,
    }
    if ext not in readers:
        raise ValueError(f"Unsupported file type: {ext}")
    log.info(f"Reading {path}")
    return readers[ext](path)


def write_data(df: pd.DataFrame, path: str, fmt: str = "parquet") -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    elif fmt == "csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported write format: {fmt}")
    log.info(f"Saved {len(df)} rows -> {path}")
    return path


def read_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def write_json(data: dict | list, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)
