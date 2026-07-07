import os
import re
import glob as _glob
import warnings
import yaml
import pandas as pd

from src.ingestion.schema_mapper import SchemaMapper, REQUIRED_SCHEMAS
from src.ingestion.data_validator import DataValidator

_RE_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")

def _interpolate(value: str) -> str:
    if not isinstance(value, str):
        return value
    def _sub(m):
        val = os.environ.get(m.group(1))
        if val is None:
            if m.group(2) is not None:
                return m.group(2)
            raise EnvironmentError(f"Required env var ${{{m.group(1)}}} is not set")
        return val
    return _RE_VAR.sub(_sub, value)


def _load_sources_yaml(path: str = None) -> list[dict]:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "..", "sources.yaml")
    path = os.path.abspath(path)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw.get("sources", []) if raw else []


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [
        re.sub(r"[^a-z0-9]+", "_", str(c).lower()).strip("_")
        for c in df.columns
    ]
    return df


def _apply_column_map(df: pd.DataFrame, col_map: dict | None) -> pd.DataFrame:
    if not col_map:
        return df
    renamed = {k: v for k, v in col_map.items() if k in df.columns}
    return df.rename(columns=renamed)


def _load_source(source: dict) -> pd.DataFrame | None:
    src_type = source.get("type", "file").lower()
    name     = source.get("name", "unnamed")

    try:
        if src_type == "file":
            from src.ingestion.connectors.file_connector import FileConnector
            df = FileConnector().load(source)
        elif src_type == "database":
            from src.ingestion.connectors.database_connector import DatabaseConnector
            df = DatabaseConnector().load(source)
        elif src_type == "api":
            from src.ingestion.connectors.api_connector import APIConnector
            df = APIConnector().load(source)
        else:
            warnings.warn(f"[source_loader] Unknown source type '{src_type}' for '{name}'")
            return None

        df = _norm_cols(df)
        df = _apply_column_map(df, source.get("column_map"))
        print(f"  [source_loader] Loaded '{name}' → {len(df):,} rows × {len(df.columns)} cols")
        return df

    except Exception as e:
        warnings.warn(f"[source_loader] Failed to load source '{name}': {e}")
        return None


def load_all_sources(
    sources_yaml_path: str = None,
    auto_map: bool = True,
    validate: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Reads sources.yaml, loads every source, auto-maps columns (fuzzy),
    unions sources that share the same maps_to target, validates, and
    returns a dict keyed by target table name.
    """
    entries = _load_sources_yaml(sources_yaml_path)
    if not entries:
        return {}

    mapper    = SchemaMapper()
    validator = DataValidator()

    buckets: dict[str, list[pd.DataFrame]] = {
        "customers": [], "transactions": [], "products": [], "promotions": []
    }

    for src in entries:
        target = src.get("maps_to", "")
        if target not in buckets:
            warnings.warn(f"[source_loader] Unknown maps_to='{target}' in source '{src.get('name')}' — skipping")
            continue

        df = _load_source(src)
        if df is None or df.empty:
            continue

        if auto_map and not src.get("column_map"):
            auto = mapper.auto_map(df, target)
            df   = mapper.apply_mapping(df, auto)

        buckets[target].append(df)

    result: dict[str, pd.DataFrame] = {}
    for target, frames in buckets.items():
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

        if validate:
            validate_fn = getattr(validator, f"validate_{target}", None)
            if validate_fn:
                vr = validate_fn(combined)
                for err in vr.errors:
                    warnings.warn(f"[source_loader] {target}: {err}")
                for warn in vr.warnings:
                    print(f"  [source_loader] WARN {target}: {warn}")

        result[target] = combined
        print(f"  [source_loader] Final '{target}': {len(combined):,} rows")

    if "customers" in result and "transactions" in result:
        ri = validator.referential_integrity(result["customers"], result["transactions"])
        for w in ri:
            print(f"  [source_loader] WARN referential_integrity: {w}")

    return result
