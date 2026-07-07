import os
import re

import pandas as pd

from src.security.validators import SQLValidator, scrub_secrets, SecurityError


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [
        re.sub(r"[^a-z0-9]+", "_", col.strip().lower()).strip("_")
        for col in df.columns
    ]
    return df


def _interpolate_env(s: str) -> str:
    def replace(match):
        var = match.group(1)
        if ":-" in var:
            name, default = var.split(":-", 1)
            return os.environ.get(name.strip(), default)
        return os.environ.get(var.strip(), match.group(0))

    return re.sub(r"\$\{([^}]+)\}", replace, s)


def _check_dialect_driver(uri: str) -> None:
    lower = uri.lower()

    if lower.startswith("postgresql://") or lower.startswith("postgres://"):
        try:
            import psycopg2  # noqa: F401
        except ImportError:
            raise ImportError("pip install psycopg2-binary")

    elif lower.startswith("mysql://") or lower.startswith("mysql+pymysql://"):
        try:
            import pymysql  # noqa: F401
        except ImportError:
            raise ImportError("pip install pymysql")

    elif lower.startswith("snowflake://"):
        try:
            import snowflake.sqlalchemy  # noqa: F401
        except ImportError:
            raise ImportError("pip install snowflake-sqlalchemy")

    elif lower.startswith("bigquery://"):
        try:
            import sqlalchemy_bigquery  # noqa: F401
        except ImportError:
            raise ImportError("pip install sqlalchemy-bigquery")

    elif lower.startswith("redshift+redshift_connector://"):
        try:
            import redshift_connector  # noqa: F401
        except ImportError:
            raise ImportError("pip install redshift-connector sqlalchemy-redshift")

    elif lower.startswith("mssql+pyodbc://"):
        try:
            import pyodbc  # noqa: F401
        except ImportError:
            raise ImportError("pip install pyodbc")


class DatabaseConnector:
    def load(self, source_cfg: dict) -> pd.DataFrame:
        try:
            from sqlalchemy import create_engine, text
        except ImportError:
            raise ImportError("pip install sqlalchemy")

        raw_connection = source_cfg["connection"]
        connection_str = _interpolate_env(raw_connection)

        _check_dialect_driver(connection_str)

        query = source_cfg.get("query")
        table = source_cfg.get("table")
        chunksize = source_cfg.get("chunksize")
        params = source_cfg.get("params")

        if not query:
            if not table:
                raise ValueError("source_cfg must have either 'query' or 'table'")
            safe_table = SQLValidator.sanitize_identifier(table)
            query = f"SELECT * FROM {safe_table}"

        SQLValidator.assert_readonly(query)

        try:
            engine = create_engine(connection_str)
        except Exception as e:
            raise RuntimeError(f"Could not create database engine: {scrub_secrets(str(e))}")

        read_kwargs = {}
        if params:
            read_kwargs["params"] = params
        if chunksize:
            read_kwargs["chunksize"] = chunksize

        with engine.connect() as conn:
            result = pd.read_sql(text(query), conn, **read_kwargs)

        if chunksize:
            frames = list(result)
            if not frames:
                return _normalize_columns(pd.DataFrame())
            df = pd.concat(frames, ignore_index=True)
        else:
            df = result

        return _normalize_columns(df)
