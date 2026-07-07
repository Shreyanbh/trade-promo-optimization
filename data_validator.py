from __future__ import annotations

import pandas as pd
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def _null_pct(df: pd.DataFrame) -> dict[str, float]:
    if len(df) == 0:
        return {col: 1.0 for col in df.columns}
    return {col: round(df[col].isna().mean(), 4) for col in df.columns}


def _duplicate_pct(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    n_dupes = series.duplicated().sum()
    return round(n_dupes / len(series), 4)


def _try_parse_dates(series: pd.Series) -> bool:
    try:
        pd.to_datetime(series.dropna(), errors="raise")
        return True
    except (ValueError, TypeError):
        return False


def _is_numeric_positive(series: pd.Series) -> tuple[bool, bool]:
    numeric = pd.to_numeric(series.dropna(), errors="coerce")
    all_numeric = numeric.notna().all()
    all_positive = (numeric > 0).all() if all_numeric and len(numeric) > 0 else False
    return all_numeric, all_positive


class DataValidator:

    def validate_customers(self, df: pd.DataFrame) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []

        if len(df) == 0:
            errors.append("customers DataFrame has 0 rows")
            return ValidationResult(ok=False, errors=errors, warnings=warnings, stats={})

        required = ["customer_id"]
        for col in required:
            if col not in df.columns:
                errors.append(f"Missing required column: '{col}'")
            elif df[col].isna().all():
                errors.append(f"Required column '{col}' is entirely null")

        stats = {
            "row_count": len(df),
            "null_pct": _null_pct(df),
            "duplicate_pct": {},
        }

        if "customer_id" in df.columns and not df["customer_id"].isna().all():
            dup_pct = _duplicate_pct(df["customer_id"])
            stats["duplicate_pct"]["customer_id"] = dup_pct
            if dup_pct > 0.05:
                warnings.append(
                    f"customer_id has {dup_pct:.1%} duplicate values (threshold: 5%)"
                )

        date_cols = ["signup_date"]
        for col in date_cols:
            if col in df.columns and not df[col].isna().all():
                if not _try_parse_dates(df[col]):
                    errors.append(f"Column '{col}' contains unparseable date values")

        return ValidationResult(
            ok=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            stats=stats,
        )

    def validate_transactions(self, df: pd.DataFrame) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []

        if len(df) == 0:
            errors.append("transactions DataFrame has 0 rows")
            return ValidationResult(ok=False, errors=errors, warnings=warnings, stats={})

        required = ["transaction_id", "customer_id", "product_id", "date", "amount"]
        for col in required:
            if col not in df.columns:
                errors.append(f"Missing required column: '{col}'")
            elif df[col].isna().all():
                errors.append(f"Required column '{col}' is entirely null")

        stats = {
            "row_count": len(df),
            "null_pct": _null_pct(df),
            "duplicate_pct": {},
        }

        for id_col in ["transaction_id", "customer_id"]:
            if id_col in df.columns and not df[id_col].isna().all():
                dup_pct = _duplicate_pct(df[id_col])
                stats["duplicate_pct"][id_col] = dup_pct
                if id_col == "transaction_id" and dup_pct > 0.05:
                    warnings.append(
                        f"transaction_id has {dup_pct:.1%} duplicate values (threshold: 5%)"
                    )

        if "date" in df.columns and not df["date"].isna().all():
            if not _try_parse_dates(df["date"]):
                errors.append("Column 'date' contains unparseable date values")

        if "amount" in df.columns and not df["amount"].isna().all():
            all_numeric, all_positive = _is_numeric_positive(df["amount"])
            if not all_numeric:
                errors.append("Column 'amount' contains non-numeric values")
            elif not all_positive:
                warnings.append("Column 'amount' contains zero or negative values")

        return ValidationResult(
            ok=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            stats=stats,
        )

    def validate_products(self, df: pd.DataFrame) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []

        if len(df) == 0:
            errors.append("products DataFrame has 0 rows")
            return ValidationResult(ok=False, errors=errors, warnings=warnings, stats={})

        required = ["product_id"]
        for col in required:
            if col not in df.columns:
                errors.append(f"Missing required column: '{col}'")
            elif df[col].isna().all():
                errors.append(f"Required column '{col}' is entirely null")

        stats = {
            "row_count": len(df),
            "null_pct": _null_pct(df),
            "duplicate_pct": {},
        }

        if "product_id" in df.columns and not df["product_id"].isna().all():
            dup_pct = _duplicate_pct(df["product_id"])
            stats["duplicate_pct"]["product_id"] = dup_pct
            if dup_pct > 0.05:
                warnings.append(
                    f"product_id has {dup_pct:.1%} duplicate values (threshold: 5%)"
                )

        if "price" in df.columns and not df["price"].isna().all():
            all_numeric, all_positive = _is_numeric_positive(df["price"])
            if not all_numeric:
                errors.append("Column 'price' contains non-numeric values")
            elif not all_positive:
                warnings.append("Column 'price' contains zero or negative values")

        return ValidationResult(
            ok=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            stats=stats,
        )

    def validate_promotions(self, df: pd.DataFrame) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []

        if len(df) == 0:
            errors.append("promotions DataFrame has 0 rows")
            return ValidationResult(ok=False, errors=errors, warnings=warnings, stats={})

        required = ["promo_id"]
        for col in required:
            if col not in df.columns:
                errors.append(f"Missing required column: '{col}'")
            elif df[col].isna().all():
                errors.append(f"Required column '{col}' is entirely null")

        stats = {
            "row_count": len(df),
            "null_pct": _null_pct(df),
            "duplicate_pct": {},
        }

        if "promo_id" in df.columns and not df["promo_id"].isna().all():
            dup_pct = _duplicate_pct(df["promo_id"])
            stats["duplicate_pct"]["promo_id"] = dup_pct
            if dup_pct > 0.05:
                warnings.append(
                    f"promo_id has {dup_pct:.1%} duplicate values (threshold: 5%)"
                )

        for date_col in ["start_date", "end_date"]:
            if date_col in df.columns and not df[date_col].isna().all():
                if not _try_parse_dates(df[date_col]):
                    errors.append(f"Column '{date_col}' contains unparseable date values")

        if "discount_pct" in df.columns and not df["discount_pct"].isna().all():
            all_numeric, all_positive = _is_numeric_positive(df["discount_pct"])
            if not all_numeric:
                errors.append("Column 'discount_pct' contains non-numeric values")
            elif not all_positive:
                warnings.append("Column 'discount_pct' contains zero or negative values")

        return ValidationResult(
            ok=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            stats=stats,
        )

    def validate_all(
        self,
        customers_df: pd.DataFrame,
        transactions_df: pd.DataFrame,
        products_df: pd.DataFrame | None = None,
        promotions_df: pd.DataFrame | None = None,
    ) -> dict[str, ValidationResult]:
        results: dict[str, ValidationResult] = {
            "customers": self.validate_customers(customers_df),
            "transactions": self.validate_transactions(transactions_df),
        }
        if products_df is not None:
            results["products"] = self.validate_products(products_df)
        if promotions_df is not None:
            results["promotions"] = self.validate_promotions(promotions_df)
        return results

    def referential_integrity(
        self,
        customers_df: pd.DataFrame,
        transactions_df: pd.DataFrame,
    ) -> list[str]:
        warnings: list[str] = []

        if "customer_id" not in customers_df.columns:
            warnings.append("customers DataFrame missing 'customer_id'; cannot check referential integrity")
            return warnings

        if "customer_id" not in transactions_df.columns:
            warnings.append("transactions DataFrame missing 'customer_id'; cannot check referential integrity")
            return warnings

        known_ids = set(customers_df["customer_id"].dropna().astype(str))
        txn_ids = transactions_df["customer_id"].dropna().astype(str)

        if len(txn_ids) == 0:
            warnings.append("transactions DataFrame has no non-null customer_id values to check")
            return warnings

        orphan_mask = ~txn_ids.isin(known_ids)
        orphan_count = orphan_mask.sum()
        orphan_pct = orphan_count / len(txn_ids)

        if orphan_count > 0:
            warnings.append(
                f"{orphan_count} transaction rows ({orphan_pct:.1%}) have customer_id values "
                f"not found in the customers table"
            )

        return warnings
