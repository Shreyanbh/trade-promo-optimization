import glob
import io
import json
import logging
import os
import re
import warnings
from pathlib import Path

import pandas as pd

from src.security.validators import PathValidator, FileValidator, SecurityError

logger = logging.getLogger(__name__)

_MAX_FILE_MB = int(os.environ.get("MAX_FILE_MB", "500"))


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [
        re.sub(r"[^a-z0-9]+", "_", col.strip().lower()).strip("_")
        for col in df.columns
    ]
    return df


def _infer_format(path: str) -> str:
    lower = path.lower().split("?")[0]
    for ext, fmt in [
        (".parquet", "parquet"),
        (".xlsx", "excel"),
        (".xls", "excel"),
        (".json", "json"),
        (".xml", "xml"),
        (".avro", "avro"),
        (".csv", "csv"),
    ]:
        if lower.endswith(ext):
            return fmt
    return "csv"


def _read_single(file_like, fmt: str, options: dict, sheet_name, encoding: str) -> pd.DataFrame:
    if fmt == "csv":
        kwargs = {"low_memory": False, "encoding": encoding}
        kwargs.update(options)
        return pd.read_csv(file_like, **kwargs)

    elif fmt == "parquet":
        kwargs = {"engine": "pyarrow"}
        kwargs.update(options)
        return pd.read_parquet(file_like, **kwargs)

    elif fmt == "excel":
        kwargs = {"sheet_name": sheet_name, "engine": "openpyxl"}
        kwargs.update(options)
        return pd.read_excel(file_like, **kwargs)

    elif fmt == "json":
        if isinstance(file_like, (str, Path)):
            with open(file_like, "r", encoding=encoding) as f:
                raw = json.load(f)
        else:
            raw = json.load(io.TextIOWrapper(file_like, encoding=encoding))
        if isinstance(raw, list):
            return pd.json_normalize(raw)
        return pd.read_json(io.StringIO(json.dumps(raw)), **options)

    elif fmt == "xml":
        kwargs = {}
        kwargs.update(options)
        return pd.read_xml(file_like, **kwargs)

    elif fmt == "avro":
        try:
            import fastavro
        except ImportError:
            raise ImportError("pip install fastavro")
        if isinstance(file_like, (str, Path)):
            with open(file_like, "rb") as f:
                records = list(fastavro.reader(f))
        else:
            records = list(fastavro.reader(file_like))
        return pd.DataFrame(records)

    elif fmt == "delta":
        try:
            from deltalake import DeltaTable
        except ImportError:
            raise ImportError("pip install deltalake")
        path_str = str(file_like) if not isinstance(file_like, str) else file_like
        return DeltaTable(path_str).to_pandas()

    else:
        raise ValueError(f"Unsupported format: {fmt}")


def _list_s3_keys(bucket: str, prefix: str, pattern: str):
    try:
        import boto3
    except ImportError:
        raise ImportError("pip install boto3")
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    import fnmatch
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if fnmatch.fnmatch(key, pattern):
                keys.append(key)
    return keys


def _read_s3_file(bucket: str, key: str) -> io.BytesIO:
    try:
        import boto3
    except ImportError:
        raise ImportError("pip install boto3")
    s3 = boto3.client("s3")
    buf = io.BytesIO()
    s3.download_fileobj(bucket, key, buf)
    buf.seek(0)
    return buf


def _list_azure_blobs(account_url: str, container: str, prefix: str, pattern: str):
    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError:
        raise ImportError("pip install azure-storage-blob")
    import fnmatch
    client = BlobServiceClient(account_url=account_url)
    container_client = client.get_container_client(container)
    blobs = []
    for blob in container_client.list_blobs(name_starts_with=prefix):
        if fnmatch.fnmatch(blob.name, pattern):
            blobs.append(blob.name)
    return blobs


def _read_azure_blob(account_url: str, container: str, blob_name: str) -> io.BytesIO:
    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError:
        raise ImportError("pip install azure-storage-blob")
    client = BlobServiceClient(account_url=account_url)
    blob_client = client.get_blob_client(container=container, blob=blob_name)
    buf = io.BytesIO()
    blob_client.download_blob().readinto(buf)
    buf.seek(0)
    return buf


def _list_gcs_blobs(bucket_name: str, prefix: str, pattern: str):
    try:
        from google.cloud import storage
    except ImportError:
        raise ImportError("pip install google-cloud-storage")
    import fnmatch
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = []
    for blob in client.list_blobs(bucket, prefix=prefix):
        if fnmatch.fnmatch(blob.name, pattern):
            blobs.append(blob.name)
    return blobs


def _read_gcs_blob(bucket_name: str, blob_name: str) -> io.BytesIO:
    try:
        from google.cloud import storage
    except ImportError:
        raise ImportError("pip install google-cloud-storage")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    buf = io.BytesIO()
    blob.download_to_file(buf)
    buf.seek(0)
    return buf


class FileConnector:
    def load(self, source_cfg: dict) -> pd.DataFrame:
        path = source_cfg["path"]
        fmt = source_cfg.get("format") or _infer_format(path)
        options = source_cfg.get("options") or {}
        sheet_name = source_cfg.get("sheet_name", 0)
        encoding = source_cfg.get("encoding", "utf-8")

        frames = []

        if fmt == "delta":
            df = _read_single(path, "delta", options, sheet_name, encoding)
            return _normalize_columns(df)

        if path.startswith("s3://"):
            rest = path[5:]
            bucket, _, blob_prefix_pattern = rest.partition("/")
            prefix_dir = "/".join(blob_prefix_pattern.split("/")[:-1])
            pattern = blob_prefix_pattern
            keys = _list_s3_keys(bucket, prefix_dir + "/" if prefix_dir else "", pattern)
            if not keys:
                warnings.warn(f"No S3 files matched: {path}")
                return pd.DataFrame()
            for key in keys:
                buf = _read_s3_file(bucket, key)
                file_fmt = fmt if source_cfg.get("format") else _infer_format(key)
                frames.append(_read_single(buf, file_fmt, options, sheet_name, encoding))

        elif path.startswith("abfs://") or (
            path.startswith("https://") and ".blob.core.windows.net" in path
        ):
            if path.startswith("abfs://"):
                rest = path[7:]
                container, _, blob_pattern = rest.partition("@")[0], rest.partition("@")[1], rest.partition("/")[2]
                container = rest.split("@")[0]
                host_and_path = rest.split("@")[1] if "@" in rest else rest
                account_url = "https://" + host_and_path.split("/")[0]
                blob_pattern = "/".join(host_and_path.split("/")[1:])
            else:
                parts = path[8:].split("/", 2)
                account_url = "https://" + parts[0]
                container = parts[1] if len(parts) > 1 else ""
                blob_pattern = parts[2] if len(parts) > 2 else ""
            prefix = "/".join(blob_pattern.split("/")[:-1])
            blob_names = _list_azure_blobs(account_url, container, prefix + "/" if prefix else "", blob_pattern)
            if not blob_names:
                warnings.warn(f"No Azure blobs matched: {path}")
                return pd.DataFrame()
            for blob_name in blob_names:
                buf = _read_azure_blob(account_url, container, blob_name)
                file_fmt = fmt if source_cfg.get("format") else _infer_format(blob_name)
                frames.append(_read_single(buf, file_fmt, options, sheet_name, encoding))

        elif path.startswith("gs://"):
            rest = path[5:]
            bucket_name, _, blob_pattern = rest.partition("/")
            prefix = "/".join(blob_pattern.split("/")[:-1])
            blob_names = _list_gcs_blobs(bucket_name, prefix + "/" if prefix else "", blob_pattern)
            if not blob_names:
                warnings.warn(f"No GCS blobs matched: {path}")
                return pd.DataFrame()
            for blob_name in blob_names:
                buf = _read_gcs_blob(bucket_name, blob_name)
                file_fmt = fmt if source_cfg.get("format") else _infer_format(blob_name)
                frames.append(_read_single(buf, file_fmt, options, sheet_name, encoding))

        else:
            # Path traversal check before any glob expansion
            PathValidator.assert_safe_local(path)
            matched = glob.glob(path, recursive=True)
            if not matched:
                warnings.warn(f"No local files matched: {path}")
                return pd.DataFrame()
            for local_path in matched:
                PathValidator.assert_safe_local(local_path)
                size_mb = os.path.getsize(local_path) / (1024 * 1024)
                if size_mb > _MAX_FILE_MB:
                    raise SecurityError(
                        f"File '{local_path}' is {size_mb:.1f} MB — limit is {_MAX_FILE_MB} MB."
                    )
                file_fmt = fmt if source_cfg.get("format") else _infer_format(local_path)
                frames.append(_read_single(local_path, file_fmt, options, sheet_name, encoding))

        if not frames:
            warnings.warn(f"No data loaded from: {path}")
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        return _normalize_columns(result)
