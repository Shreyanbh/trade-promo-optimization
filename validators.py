from dataclasses import dataclass, field
import pandas as pd


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"Valid: {self.is_valid}"]
        if self.errors:
            lines += [f"  ERROR: {e}" for e in self.errors]
        if self.warnings:
            lines += [f"  WARN:  {w}" for w in self.warnings]
        return "\n".join(lines)


SCHEMAS = {
    "customer": {
        "required_cols": ["customer_id", "age", "region", "signup_date"],
        "dtypes": {"customer_id": "object"},
    },
    "transaction": {
        "required_cols": ["transaction_id", "customer_id", "product_id", "date", "amount", "quantity"],
        "dtypes": {"amount": "float64", "quantity": "int64"},
    },
    "product": {
        "required_cols": ["product_id", "product_name", "category", "price"],
        "dtypes": {"price": "float64"},
    },
    "promo": {
        "required_cols": ["promo_id", "product_id", "discount_pct", "start_date", "end_date"],
        "dtypes": {"discount_pct": "float64"},
    },
}


def validate_dataframe(df: pd.DataFrame, schema_name: str) -> ValidationResult:
    if schema_name not in SCHEMAS:
        return ValidationResult(False, errors=[f"Unknown schema: {schema_name}"])

    schema = SCHEMAS[schema_name]
    errors, warnings = [], []

    missing = [c for c in schema["required_cols"] if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")

    null_rates = df.isnull().mean()
    for col, rate in null_rates.items():
        if rate > 0.5:
            errors.append(f"Column '{col}' has {rate:.1%} nulls (threshold 50%)")
        elif rate > 0.1:
            warnings.append(f"Column '{col}' has {rate:.1%} nulls")

    if df.duplicated().sum() > 0:
        warnings.append(f"{df.duplicated().sum()} duplicate rows detected")

    return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)
