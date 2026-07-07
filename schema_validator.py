import pandas as pd
from src.utils.validators import validate_dataframe, ValidationResult
from src.utils.logger import get_logger

log = get_logger(__name__)


def validate(df: pd.DataFrame, schema_name: str) -> ValidationResult:
    result = validate_dataframe(df, schema_name)
    if result.is_valid:
        log.info(f"Schema '{schema_name}' validation PASSED")
    else:
        log.warning(f"Schema '{schema_name}' validation FAILED: {result.errors}")
    return result
