"""
Multi-source extractors for the ELT pipeline.

Each extractor pulls from a specific source format and normalises the result
into a pandas DataFrame (or list of dicts for nested sources like JSON).
Staged output is written to data/staging/ as Parquet before transformation.
"""
import json
import os
import re

import pandas as pd

from src.utils.logger import get_logger

log = get_logger("extractors")

STAGING = os.path.join("data", "staging")


def _stage(df: pd.DataFrame, name: str) -> str:
    os.makedirs(STAGING, exist_ok=True)
    path = os.path.join(STAGING, f"{name}.parquet")
    df.to_parquet(path, index=False)
    log.info(f"  Staged {name}: {len(df)} rows -> {path}")
    return path


# ── Structured extractors ─────────────────────────────────────────────────────

class CSVExtractor:
    """Extract one or more CSV files into a single DataFrame."""

    def extract(self, path: str, **read_kwargs) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV not found: {path}")
        df = pd.read_csv(path, **read_kwargs)
        log.info(f"CSV extracted: {path} ({len(df)} rows, {len(df.columns)} cols)")
        return df

    def extract_and_stage(self, path: str, name: str, **read_kwargs) -> tuple[pd.DataFrame, str]:
        df = self.extract(path, **read_kwargs)
        staged = _stage(df, name)
        return df, staged


class ExcelExtractor:
    """Extract one or all sheets from an Excel workbook."""

    def extract_sheet(self, path: str, sheet_name=0, **read_kwargs) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Excel not found: {path}")
        df = pd.read_excel(path, sheet_name=sheet_name, **read_kwargs)
        log.info(f"Excel extracted: {path} sheet={sheet_name!r} ({len(df)} rows)")
        return df

    def extract_all_sheets(self, path: str) -> dict[str, pd.DataFrame]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Excel not found: {path}")
        sheets = pd.read_excel(path, sheet_name=None)
        for name, df in sheets.items():
            log.info(f"  Sheet '{name}': {len(df)} rows x {len(df.columns)} cols")
        return sheets

    def extract_and_stage(self, path: str) -> dict[str, str]:
        sheets = self.extract_all_sheets(path)
        staged = {}
        for sheet_name, df in sheets.items():
            key = sheet_name.lower().replace(" ", "_")
            staged[key] = _stage(df, f"excel_{key}")
        return staged


class JSONExtractor:
    """Extract nested or flat JSON files."""

    def extract(self, path: str) -> list[dict]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"JSON not found: {path}")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        log.info(f"JSON extracted: {path} ({len(data)} records)")
        return data

    def extract_flat(self, path: str) -> pd.DataFrame:
        """Flatten top-level array of objects into a DataFrame. Nested lists become comma-joined strings."""
        records = self.extract(path)
        flat = []
        for r in records:
            row = {}
            for k, v in r.items():
                if isinstance(v, list):
                    row[k] = ",".join(str(x) for x in v)
                elif isinstance(v, dict):
                    row[k] = json.dumps(v)
                else:
                    row[k] = v
            flat.append(row)
        return pd.DataFrame(flat)

    def extract_and_stage(self, path: str, name: str) -> tuple[pd.DataFrame, str]:
        df = self.extract_flat(path)
        staged = _stage(df, name)
        return df, staged


# ── Unstructured extractors ───────────────────────────────────────────────────

class TextExtractor:
    """
    Parses delimited free-text files (reviews, call transcripts, support emails)
    into structured DataFrames using regex-based field extraction.
    """

    # header patterns
    _REVIEW_HDR  = re.compile(
        r"REVIEW \[(?P<review_id>R\d+)\].*?CUSTOMER:\s*(?P<customer_id>C\d+).*?PRODUCT:\s*(?P<product_id>P\d+).*?DATE:\s*(?P<date>[\d-]+).*?RATING:\s*(?P<rating>\d+)"
    )
    _CALL_HDR    = re.compile(
        r"CALL \[(?P<call_id>T\d+)\].*?DATE:\s*(?P<date>[\d-]+).*?AGENT:\s*(?P<agent>\w+).*?CUSTOMER:\s*(?P<customer_id>C\d+)"
    )
    _CALL_PROD   = re.compile(r"PRODUCT REFERENCED:\s*(?P<product_id>P\d+)")
    _CALL_SENTI  = re.compile(r"SENTIMENT:\s*(?P<sentiment>\w+)")
    _CALL_RESOL  = re.compile(r"RESOLUTION:\s*(?P<resolution>.+)")
    _EMAIL_HDR   = re.compile(
        r"EMAIL \[(?P<email_id>E\d+)\].*?DATE:\s*(?P<date>[\d-]+).*?CUSTOMER:\s*(?P<customer_id>C\d+).*?SUBJECT:\s*(?P<subject>.+)"
    )
    _EMAIL_PROD  = re.compile(r"PRODUCT:\s*(?P<product_id>P\d+)")
    _EMAIL_SENTI = re.compile(r"SENTIMENT_LABEL:\s*(?P<sentiment>\w+)")

    def _read_blocks(self, path: str, sep: str) -> list[str]:
        with open(path, encoding="utf-8") as f:
            content = f.read()
        return [b.strip() for b in content.split(sep) if b.strip()]

    def extract_reviews(self, path: str) -> pd.DataFrame:
        blocks = self._read_blocks(path, "---")
        rows = []
        for block in blocks:
            lines = block.splitlines()
            if not lines:
                continue
            hdr = self._REVIEW_HDR.search(lines[0])
            if not hdr:
                continue
            text = " ".join(l.strip('"').strip() for l in lines[1:] if l.strip())
            rows.append({
                "review_id":   hdr.group("review_id"),
                "customer_id": hdr.group("customer_id"),
                "product_id":  hdr.group("product_id"),
                "date":        hdr.group("date"),
                "rating":      int(hdr.group("rating")),
                "review_text": text,
                "source":      "customer_reviews",
            })
        df = pd.DataFrame(rows)
        log.info(f"Reviews extracted: {path} ({len(df)} reviews)")
        return df

    def extract_transcripts(self, path: str) -> pd.DataFrame:
        blocks = self._read_blocks(path, "---")
        rows = []
        for block in blocks:
            hdr  = self._CALL_HDR.search(block)
            prod = self._CALL_PROD.search(block)
            sent = self._CALL_SENTI.search(block)
            res  = self._CALL_RESOL.search(block)
            if not hdr:
                continue
            rows.append({
                "call_id":     hdr.group("call_id"),
                "customer_id": hdr.group("customer_id"),
                "date":        hdr.group("date"),
                "agent":       hdr.group("agent"),
                "product_id":  prod.group("product_id") if prod else "",
                "sentiment":   sent.group("sentiment")  if sent else "UNKNOWN",
                "resolution":  res.group("resolution").strip() if res else "",
                "source":      "call_transcripts",
            })
        df = pd.DataFrame(rows)
        log.info(f"Transcripts extracted: {path} ({len(df)} calls)")
        return df

    def extract_emails(self, path: str) -> pd.DataFrame:
        blocks = self._read_blocks(path, "===")
        rows = []
        for block in blocks:
            hdr  = self._EMAIL_HDR.search(block)
            prod = self._EMAIL_PROD.search(block)
            sent = self._EMAIL_SENTI.search(block)
            if not hdr:
                continue
            rows.append({
                "email_id":    hdr.group("email_id"),
                "customer_id": hdr.group("customer_id"),
                "date":        hdr.group("date"),
                "subject":     hdr.group("subject").strip(),
                "product_id":  prod.group("product_id") if prod else "",
                "sentiment":   sent.group("sentiment")  if sent else "UNKNOWN",
                "source":      "support_emails",
            })
        df = pd.DataFrame(rows)
        log.info(f"Emails extracted: {path} ({len(df)} emails)")
        return df

    def extract_and_stage(self, reviews_path: str, transcripts_path: str,
                          emails_path: str) -> dict[str, tuple[pd.DataFrame, str]]:
        result = {}
        result["reviews"]     = (df_r := self.extract_reviews(reviews_path),     _stage(df_r, "reviews"))
        result["transcripts"] = (df_t := self.extract_transcripts(transcripts_path), _stage(df_t, "transcripts"))
        result["emails"]      = (df_e := self.extract_emails(emails_path),        _stage(df_e, "emails"))
        return result
