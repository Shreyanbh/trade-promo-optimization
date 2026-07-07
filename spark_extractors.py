"""
SparkExtractors — read multi-format source files with PySpark, write staging Parquet.

At 5M-customer scale, structured sources are Parquet; small sources (Excel, JSON) are
read with pandas and pushed into Spark DataFrames.  Unstructured TXT files are still
parsed with regex (same logic as extractors.py) — they remain a small dataset.
"""
import os
import re

import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

from src.config.settings import PATHS
from src.utils.logger import get_logger

log = get_logger("spark_extractors")

STAGING = PATHS["staging"]


def _stage_spark(df: DataFrame, name: str, spark_path: bool = True) -> str:
    """Write a Spark DataFrame to staging as Parquet; return the path."""
    path = os.path.join(STAGING, name)
    df.write.mode("overwrite").parquet(path)
    log.info(f"  Staged {name}: {path}/ ({df.count():,} rows)")
    return path


def _stage_pandas(df: pd.DataFrame, name: str) -> str:
    """Write a small pandas DataFrame to staging as Parquet; return the path."""
    path = os.path.join(STAGING, f"{name}.parquet")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")
    log.info(f"  Staged {name}: {path} ({len(df):,} rows, pandas)")
    return path


# ── Large Parquet sources (native PySpark read) ───────────────────────────────

class SparkParquetExtractor:
    """Reads partitioned Parquet directories produced by spark_data_generator."""

    def extract(self, spark: SparkSession, path: str) -> DataFrame:
        df = spark.read.parquet(path)
        log.info(f"  SparkParquetExtractor: {path}/ -> {df.count():,} rows, {len(df.columns)} cols")
        return df

    def extract_and_stage(self, spark: SparkSession, path: str, name: str) -> tuple:
        df = self.extract(spark, path)
        staged = _stage_spark(df, name)
        return df, staged


# ── Excel (small — pandas bridge) ────────────────────────────────────────────

class SparkExcelExtractor:
    """Reads Excel sheets via pandas, bridges to Spark DataFrame."""

    def extract_sheet(self, spark: SparkSession, path: str, sheet_name: str) -> DataFrame:
        pdf = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
        pdf.columns = [c.lower().replace(" ", "_") for c in pdf.columns]
        df = spark.createDataFrame(pdf)
        log.info(f"  SparkExcelExtractor [{sheet_name}]: {len(pdf):,} rows")
        return df

    def extract_all_sheets(self, spark: SparkSession, path: str) -> dict:
        xl = pd.ExcelFile(path, engine="openpyxl")
        return {s: self.extract_sheet(spark, path, s) for s in xl.sheet_names}

    def extract_and_stage(self, spark: SparkSession, path: str) -> dict:
        sheets = self.extract_all_sheets(spark, path)
        result = {}
        for name, df in sheets.items():
            key = name.lower()
            result[key] = (_stage_spark(df, f"products_{key}"), df)
        return result


# ── JSON (small — pandas bridge, then Spark) ─────────────────────────────────

class SparkJSONExtractor:
    """Reads JSON via pandas, flattens nested arrays, bridges to Spark DataFrame."""

    def extract(self, spark: SparkSession, path: str) -> DataFrame:
        with open(path, encoding="utf-8") as f:
            import json
            records = json.load(f)
        pdf = pd.json_normalize(records)
        pdf.columns = [c.lower().replace(".", "_").replace(" ", "_") for c in pdf.columns]
        df = spark.createDataFrame(pdf.astype(str))
        log.info(f"  SparkJSONExtractor: {path} -> {len(pdf):,} records")
        return df

    def extract_flat(self, spark: SparkSession, path: str) -> DataFrame:
        """Flatten top-level JSON — list fields joined to comma-separated strings."""
        with open(path, encoding="utf-8") as f:
            import json
            records = json.load(f)
        flat = []
        for r in records:
            row = {}
            for k, v in r.items():
                row[k] = ",".join(str(x) for x in v) if isinstance(v, list) else str(v)
            flat.append(row)
        pdf = pd.DataFrame(flat)
        df = spark.createDataFrame(pdf)
        log.info(f"  SparkJSONExtractor (flat): {path} -> {len(pdf):,} records")
        return df

    def extract_and_stage(self, spark: SparkSession, path: str, name: str) -> tuple:
        df = self.extract_flat(spark, path)
        staged = _stage_spark(df, name)
        return df, staged


# ── Unstructured TXT (pandas regex — same as before, small dataset) ──────────

_REVIEW_HDR  = re.compile(
    r"REVIEW \[(?P<id>R\d+)\] \| CUSTOMER: (?P<customer>C\d+) \| PRODUCT: (?P<product>P\d+)"
    r" \| DATE: (?P<date>[\d-]+) \| RATING: (?P<rating>\d)"
)
_REVIEW_BODY = re.compile(r'"(.+?)"')

_CALL_HDR   = re.compile(
    r"CALL \[(?P<id>T\d+)\] \| DATE: (?P<date>[\d-]+) \| AGENT: (?P<agent>\w+) \| CUSTOMER: (?P<customer>C\d+)"
)
_CALL_PROD  = re.compile(r"PRODUCT REFERENCED: (?P<product>P\d+)")
_CALL_RESOL = re.compile(r"RESOLUTION: (?P<resolution>.+)")
_CALL_SENTI = re.compile(r"SENTIMENT: (?P<sentiment>POSITIVE|NEGATIVE|NEUTRAL)")

_EMAIL_HDR  = re.compile(
    r"EMAIL \[(?P<id>E\d+)\] \| DATE: (?P<date>[\d-]+) \| CUSTOMER: (?P<customer>C\d+)"
    r" \| SUBJECT: (?P<subject>.+)"
)
_EMAIL_PROD  = re.compile(r"PRODUCT: (?P<product>P\d+)")
_EMAIL_SENTI = re.compile(r"SENTIMENT_LABEL: (?P<sentiment>POSITIVE|NEGATIVE|NEUTRAL)")


class SparkTextExtractor:
    """Parses structured fields from TXT files; produces pandas DFs bridged to Spark."""

    def extract_reviews(self, spark: SparkSession, path: str) -> tuple:
        rows, text, hdr = [], None, None
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()
                m = _REVIEW_HDR.match(line)
                if m:
                    hdr = m.groupdict()
                    continue
                if hdr:
                    mb = _REVIEW_BODY.match(line)
                    if mb:
                        rows.append({
                            "review_id":   hdr["id"],
                            "customer_id": hdr["customer"],
                            "product_id":  hdr["product"],
                            "date":        hdr["date"],
                            "rating":      int(hdr["rating"]),
                            "review_text": mb.group(1),
                        })
                        hdr = None
        pdf = pd.DataFrame(rows)
        df = spark.createDataFrame(pdf)
        log.info(f"  TextExtractor reviews: {len(pdf):,} records")
        return df, pdf

    def extract_transcripts(self, spark: SparkSession, path: str) -> tuple:
        rows, hdr = [], None
        buf: dict = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()
                m = _CALL_HDR.match(line)
                if m:
                    buf = m.groupdict(); continue
                mp = _CALL_PROD.match(line)
                if mp: buf["product"] = mp.group("product"); continue
                mr = _CALL_RESOL.match(line)
                if mr: buf["resolution"] = mr.group("resolution"); continue
                ms = _CALL_SENTI.match(line)
                if ms:
                    buf["sentiment"] = ms.group("sentiment")
                    rows.append({
                        "call_id":     buf.get("id",""),
                        "customer_id": buf.get("customer",""),
                        "date":        buf.get("date",""),
                        "agent":       buf.get("agent",""),
                        "product_id":  buf.get("product",""),
                        "sentiment":   buf.get("sentiment",""),
                        "resolution":  buf.get("resolution",""),
                    })
                    buf = {}
        pdf = pd.DataFrame(rows)
        df = spark.createDataFrame(pdf)
        log.info(f"  TextExtractor transcripts: {len(pdf):,} records")
        return df, pdf

    def extract_emails(self, spark: SparkSession, path: str) -> tuple:
        rows, buf = [], {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()
                m = _EMAIL_HDR.match(line)
                if m:
                    buf = m.groupdict(); continue
                mp = _EMAIL_PROD.match(line)
                if mp: buf["product"] = mp.group("product"); continue
                ms = _EMAIL_SENTI.match(line)
                if ms:
                    buf["sentiment"] = ms.group("sentiment")
                    rows.append({
                        "email_id":    buf.get("id",""),
                        "customer_id": buf.get("customer",""),
                        "date":        buf.get("date",""),
                        "subject":     buf.get("subject",""),
                        "product_id":  buf.get("product",""),
                        "sentiment":   buf.get("sentiment",""),
                    })
                    buf = {}
        pdf = pd.DataFrame(rows)
        df = spark.createDataFrame(pdf)
        log.info(f"  TextExtractor emails: {len(pdf):,} records")
        return df, pdf

    def extract_and_stage(self, spark: SparkSession,
                           reviews_path: str, transcripts_path: str,
                           emails_path: str) -> dict:
        r_df, r_pdf = self.extract_reviews(spark, reviews_path)
        t_df, t_pdf = self.extract_transcripts(spark, transcripts_path)
        e_df, e_pdf = self.extract_emails(spark, emails_path)
        return {
            "reviews":     (r_pdf, _stage_spark(r_df, "reviews_staged")),
            "transcripts": (t_pdf, _stage_spark(t_df, "transcripts_staged")),
            "emails":      (e_pdf, _stage_spark(e_df, "emails_staged")),
        }
