import difflib
import pandas as pd
from datetime import datetime

REQUIRED_SCHEMAS = {
    "customers": {
        "customer_id":    {"type": "str",   "required": True,  "aliases": ["cust_id","id","customer","client_id","account_id","member_id"]},
        "region":         {"type": "str",   "required": False, "aliases": ["territory","area","zone","market","geography","state","country"]},
        "age":            {"type": "int",   "required": False, "aliases": ["age_years","customer_age","age_group"]},
        "loyalty_tier":   {"type": "str",   "required": False, "aliases": ["tier","loyalty_level","member_tier","status","membership"]},
        "income_bracket": {"type": "str",   "required": False, "aliases": ["income","income_range","income_level","salary_band"]},
        "signup_date":    {"type": "date",  "required": False, "aliases": ["join_date","registration_date","created_date","start_date","onboard_date"]},
        "email":          {"type": "str",   "required": False, "aliases": ["email_address","mail","contact_email"]},
    },
    "transactions": {
        "transaction_id": {"type": "str",   "required": True,  "aliases": ["txn_id","order_id","purchase_id","sale_id","invoice_id","receipt_id"]},
        "customer_id":    {"type": "str",   "required": True,  "aliases": ["cust_id","client_id","account_id","member_id","buyer_id"]},
        "product_id":     {"type": "str",   "required": True,  "aliases": ["prod_id","item_id","sku","article_id","product_code"]},
        "date":           {"type": "date",  "required": True,  "aliases": ["transaction_date","purchase_date","order_date","sale_date","txn_date","timestamp"]},
        "amount":         {"type": "float", "required": True,  "aliases": ["total","price","revenue","spend","sale_amount","net_amount","value","gross_amount"]},
        "quantity":       {"type": "int",   "required": False, "aliases": ["qty","units","count","num_items","quantity_sold"]},
        "channel":        {"type": "str",   "required": False, "aliases": ["sales_channel","purchase_channel","medium","store_type","channel_type"]},
        "promo_code":     {"type": "str",   "required": False, "aliases": ["promotion_code","discount_code","offer_code","coupon","voucher_code"]},
    },
    "products": {
        "product_id":     {"type": "str",   "required": True,  "aliases": ["prod_id","item_id","sku","article_id","product_code"]},
        "product_name":   {"type": "str",   "required": False, "aliases": ["name","item_name","description","product_description","title"]},
        "category":       {"type": "str",   "required": False, "aliases": ["category_name","dept","department","product_type","product_category","segment"]},
        "price":          {"type": "float", "required": False, "aliases": ["unit_price","list_price","retail_price","msrp","cost","base_price"]},
        "brand":          {"type": "str",   "required": False, "aliases": ["brand_name","manufacturer","vendor","supplier","make"]},
    },
    "promotions": {
        "promo_id":       {"type": "str",   "required": True,  "aliases": ["promotion_id","offer_id","campaign_id","deal_id","discount_id"]},
        "product_id":     {"type": "str",   "required": False, "aliases": ["prod_id","item_id","sku","applies_to"]},
        "discount_pct":   {"type": "float", "required": False, "aliases": ["discount","pct_off","percent_off","discount_percent","reduction","rebate"]},
        "start_date":     {"type": "date",  "required": False, "aliases": ["promo_start","valid_from","effective_date","begin_date","from_date"]},
        "end_date":       {"type": "date",  "required": False, "aliases": ["promo_end","valid_to","expiry_date","expiration_date","to_date"]},
        "promo_type":     {"type": "str",   "required": False, "aliases": ["type","promotion_type","offer_type","deal_type","mechanic"]},
    },
}


def _normalize(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _score_candidate(pipeline_col: str, user_col: str, aliases: list[str]) -> float:
    norm_user = _normalize(user_col)
    norm_pipe = _normalize(pipeline_col)

    if norm_user == norm_pipe:
        return 1.0

    if norm_user in [_normalize(a) for a in aliases]:
        return 0.95

    if norm_pipe in norm_user or norm_user in norm_pipe:
        return 0.7

    for alias in aliases:
        norm_alias = _normalize(alias)
        if norm_alias in norm_user or norm_user in norm_alias:
            return 0.6

    best_ratio = difflib.SequenceMatcher(None, norm_pipe, norm_user).ratio()
    for alias in aliases:
        ratio = difflib.SequenceMatcher(None, _normalize(alias), norm_user).ratio()
        if ratio > best_ratio:
            best_ratio = ratio

    return best_ratio * 0.5


def _coerce_column(series: pd.Series, target_type: str) -> pd.Series:
    if target_type == "str":
        return series.astype(str).where(series.notna(), other=None)
    elif target_type == "int":
        return pd.to_numeric(series, errors="coerce").astype("Int64")
    elif target_type == "float":
        return pd.to_numeric(series, errors="coerce").astype(float)
    elif target_type == "date":
        return pd.to_datetime(series, errors="coerce")
    return series


class SchemaMapper:

    def auto_map(self, df: pd.DataFrame, table: str) -> dict[str, str | None]:
        schema = REQUIRED_SCHEMAS.get(table, {})
        user_cols = list(df.columns)
        mapping: dict[str, str | None] = {}

        for pipeline_col, meta in schema.items():
            aliases = meta.get("aliases", [])
            best_col = None
            best_score = 0.0

            for user_col in user_cols:
                score = _score_candidate(pipeline_col, user_col, aliases)
                if score > best_score:
                    best_score = score
                    best_col = user_col

            mapping[pipeline_col] = best_col if best_score > 0.3 else None

        return mapping

    def confidence(self, df: pd.DataFrame, table: str, mapping: dict[str, str | None]) -> dict[str, float]:
        schema = REQUIRED_SCHEMAS.get(table, {})
        scores: dict[str, float] = {}

        for pipeline_col, user_col in mapping.items():
            if user_col is None:
                scores[pipeline_col] = 0.0
                continue
            aliases = schema.get(pipeline_col, {}).get("aliases", [])
            scores[pipeline_col] = _score_candidate(pipeline_col, user_col, aliases)

        return scores

    def apply_mapping(self, df: pd.DataFrame, mapping: dict[str, str | None]) -> pd.DataFrame:
        reverse: dict[str, str] = {v: k for k, v in mapping.items() if v is not None}
        mapped_user_cols = [c for c in df.columns if c in reverse]
        result = df[mapped_user_cols].rename(columns=reverse).copy()

        table_schemas = {}
        for schema in REQUIRED_SCHEMAS.values():
            table_schemas.update(schema)

        for pipeline_col in result.columns:
            if pipeline_col in table_schemas:
                target_type = table_schemas[pipeline_col]["type"]
                result[pipeline_col] = _coerce_column(result[pipeline_col], target_type)

        for pipeline_col, user_col in mapping.items():
            if user_col is None and pipeline_col not in result.columns:
                result[pipeline_col] = None

        return result

    def validate(self, df: pd.DataFrame, table: str) -> list[str]:
        schema = REQUIRED_SCHEMAS.get(table, {})
        errors: list[str] = []

        for pipeline_col, meta in schema.items():
            if not meta["required"]:
                continue
            if pipeline_col not in df.columns:
                errors.append(f"Missing required column: '{pipeline_col}'")
                continue
            if df[pipeline_col].isna().all():
                errors.append(f"Required column '{pipeline_col}' is entirely null")
                continue
            expected_type = meta["type"]
            col = df[pipeline_col]
            if expected_type in ("int", "float"):
                non_null = col.dropna()
                if len(non_null) > 0 and not pd.api.types.is_numeric_dtype(non_null):
                    try:
                        pd.to_numeric(non_null, errors="raise")
                    except (ValueError, TypeError):
                        errors.append(f"Column '{pipeline_col}' has non-numeric values, expected {expected_type}")
            elif expected_type == "date":
                non_null = col.dropna()
                if len(non_null) > 0 and not pd.api.types.is_datetime64_any_dtype(non_null):
                    try:
                        pd.to_datetime(non_null, errors="raise")
                    except (ValueError, TypeError):
                        errors.append(f"Column '{pipeline_col}' has unparseable date values")

        return errors

    def summary(self, df: pd.DataFrame, table: str, mapping: dict[str, str | None]) -> dict:
        schema = REQUIRED_SCHEMAS.get(table, {})
        mapped_count = sum(1 for v in mapping.values() if v is not None)
        unmapped_required = [
            col for col, meta in schema.items()
            if meta["required"] and mapping.get(col) is None
        ]
        unmapped_optional = [
            col for col, meta in schema.items()
            if not meta["required"] and mapping.get(col) is None
        ]

        sample_values: dict[str, list] = {}
        for pipeline_col, user_col in mapping.items():
            if user_col is not None and user_col in df.columns:
                non_null = df[user_col].dropna()
                sample_values[pipeline_col] = non_null.head(3).tolist()

        return {
            "mapped": mapped_count,
            "unmapped_required": unmapped_required,
            "unmapped_optional": unmapped_optional,
            "sample_values": sample_values,
        }
