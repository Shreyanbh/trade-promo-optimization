import io
import re
import json
import warnings
import yaml
import pandas as pd
import numpy as np
from dataclasses import dataclass, field

PIPELINE_TABLES = ["customers", "transactions", "products", "promotions"]

TYPE_MAP = {
    "string":   "object",
    "str":      "object",
    "text":     "object",
    "varchar":  "object",
    "integer":  "Int64",
    "int":      "Int64",
    "bigint":   "Int64",
    "float":    "float64",
    "double":   "float64",
    "decimal":  "float64",
    "numeric":  "float64",
    "date":     "datetime64[ns]",
    "datetime": "datetime64[ns]",
    "timestamp":"datetime64[ns]",
    "boolean":  "bool",
    "bool":     "bool",
}

PANDAS_TO_SCHEMA = {
    "object":          "string",
    "string":          "string",
    "int64":           "integer",
    "Int64":           "integer",
    "int32":           "integer",
    "float64":         "float",
    "float32":         "float",
    "bool":            "boolean",
    "datetime64[ns]":  "datetime",
    "category":        "string",
}


@dataclass
class ColumnSchema:
    name: str
    type: str = "string"
    required: bool = False
    description: str = ""
    is_primary_key: bool = False
    maps_to: str = ""    # pipeline column name this maps to


@dataclass
class TableSchema:
    name: str
    description: str = ""
    maps_to: str = ""    # pipeline table (customers / transactions / products / promotions / "")
    primary_key: str = ""
    columns: list = field(default_factory=list)


@dataclass
class UserSchema:
    version: str = "1.0"
    description: str = ""
    tables: list = field(default_factory=list)


def _col_from_dict(d: dict) -> ColumnSchema:
    return ColumnSchema(
        name=str(d.get("name", "")),
        type=str(d.get("type", "string")).lower(),
        required=bool(d.get("required", False)),
        description=str(d.get("description", "")),
        is_primary_key=bool(d.get("is_primary_key", False)),
        maps_to=str(d.get("maps_to", "")),
    )


def _table_from_dict(d: dict) -> TableSchema:
    cols = [_col_from_dict(c) for c in d.get("columns", [])]
    return TableSchema(
        name=str(d.get("name", "table")),
        description=str(d.get("description", "")),
        maps_to=str(d.get("maps_to", "")),
        primary_key=str(d.get("primary_key", "")),
        columns=cols,
    )


def load_schema(source) -> UserSchema:
    if isinstance(source, (str, bytes, bytearray)):
        if isinstance(source, str):
            raw = source
        else:
            raw = source.decode("utf-8", errors="replace")
    elif hasattr(source, "read"):
        raw = source.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
    else:
        raise TypeError(f"Cannot read schema from {type(source)}")

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError:
        data = json.loads(raw)

    tables = [_table_from_dict(t) for t in data.get("tables", [])]
    return UserSchema(
        version=str(data.get("version", "1.0")),
        description=str(data.get("description", "")),
        tables=tables,
    )


def save_schema(schema: UserSchema, path: str):
    data = {
        "version": schema.version,
        "description": schema.description,
        "tables": [
            {
                "name": t.name,
                "description": t.description,
                "maps_to": t.maps_to,
                "primary_key": t.primary_key,
                "columns": [
                    {
                        "name": c.name,
                        "type": c.type,
                        "required": c.required,
                        "description": c.description,
                        "is_primary_key": c.is_primary_key,
                        "maps_to": c.maps_to,
                    }
                    for c in t.columns
                ],
            }
            for t in schema.tables
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def infer_schema(df: pd.DataFrame, table_name: str, maps_to: str = "") -> TableSchema:
    cols = []
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        schema_type = PANDAS_TO_SCHEMA.get(dtype_str, "string")
        is_pk = col.lower() in ("id", f"{table_name}_id", f"{table_name.rstrip('s')}_id")
        cols.append(ColumnSchema(
            name=col,
            type=schema_type,
            required=df[col].isnull().sum() == 0,
            is_primary_key=is_pk,
        ))
    return TableSchema(
        name=table_name,
        maps_to=maps_to,
        columns=cols,
    )


def validate_against_schema(df: pd.DataFrame, table: TableSchema) -> list[str]:
    errors = []
    if df.empty:
        errors.append(f"Table '{table.name}' is empty.")
        return errors

    for col in table.columns:
        if col.name not in df.columns:
            if col.required:
                errors.append(f"Required column '{col.name}' is missing from '{table.name}'.")
            continue

        series = df[col.name]

        if col.required and series.isnull().all():
            errors.append(f"Required column '{col.name}' in '{table.name}' is entirely null.")

        target = TYPE_MAP.get(col.type)
        if target in ("Int64", "float64"):
            non_null = series.dropna()
            if len(non_null) > 0:
                try:
                    pd.to_numeric(non_null, errors="raise")
                except (ValueError, TypeError):
                    errors.append(
                        f"Column '{col.name}' in '{table.name}' declared as {col.type} "
                        f"but contains non-numeric values."
                    )
        elif target == "datetime64[ns]":
            non_null = series.dropna()
            if len(non_null) > 0 and not pd.api.types.is_datetime64_any_dtype(series):
                try:
                    pd.to_datetime(non_null, errors="raise", infer_datetime_format=True)
                except (ValueError, TypeError):
                    errors.append(
                        f"Column '{col.name}' in '{table.name}' declared as {col.type} "
                        f"but cannot be parsed as dates."
                    )

    return errors


def coerce_types(df: pd.DataFrame, table: TableSchema) -> pd.DataFrame:
    df = df.copy()
    for col in table.columns:
        if col.name not in df.columns:
            continue
        target = TYPE_MAP.get(col.type)
        if target is None:
            continue
        try:
            if target in ("Int64", "float64"):
                df[col.name] = pd.to_numeric(df[col.name], errors="coerce").astype(target)
            elif target == "datetime64[ns]":
                df[col.name] = pd.to_datetime(df[col.name], errors="coerce", infer_datetime_format=True)
            elif target == "bool":
                df[col.name] = df[col.name].astype(bool)
        except Exception:
            pass
    return df


def schema_to_sources_yaml(schema: UserSchema, uploads_dir: str) -> str:
    entries = []
    for t in schema.tables:
        path = f"{uploads_dir}/{t.name}.parquet".replace("\\", "/")
        col_map = {c.name: c.maps_to for c in t.columns if c.maps_to and c.maps_to != c.name}
        entry = {
            "name": f"upload_{t.name}",
            "type": "file",
            "path": path,
            "format": "parquet",
            "maps_to": t.maps_to if t.maps_to else t.name,
        }
        if col_map:
            entry["column_map"] = col_map
        else:
            entry["column_map"] = {}
        entries.append(entry)

    data = {"sources": entries}
    return yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False)


def blank_schema_yaml() -> str:
    return """\
version: "1.0"
description: "My organisation's trade promo data"
tables:
  - name: customers
    description: "Customer master — one row per customer"
    maps_to: customers
    primary_key: customer_id
    columns:
      - name: customer_id
        type: string
        required: true
        is_primary_key: true
      - name: region
        type: string
        required: false
      - name: age
        type: integer
        required: false

  - name: transactions
    description: "Purchase history — one row per transaction"
    maps_to: transactions
    primary_key: transaction_id
    columns:
      - name: transaction_id
        type: string
        required: true
        is_primary_key: true
      - name: customer_id
        type: string
        required: true
      - name: product_id
        type: string
        required: false
      - name: date
        type: date
        required: true
      - name: amount
        type: float
        required: true
      - name: channel
        type: string
        required: false

  - name: products
    description: "Product catalogue — one row per SKU"
    maps_to: products
    primary_key: product_id
    columns:
      - name: product_id
        type: string
        required: true
        is_primary_key: true
      - name: product_name
        type: string
        required: false
      - name: category
        type: string
        required: false
      - name: price
        type: float
        required: false

  - name: promotions
    description: "Promotions calendar — one row per promo event"
    maps_to: promotions
    primary_key: promo_id
    columns:
      - name: promo_id
        type: string
        required: true
        is_primary_key: true
      - name: product_id
        type: string
        required: false
      - name: discount_pct
        type: float
        required: false
      - name: start_date
        type: date
        required: false
      - name: end_date
        type: date
        required: false
      - name: promo_type
        type: string
        required: false
"""
