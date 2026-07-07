from __future__ import annotations

import io
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import pandas as pd


class StorageAdapter(ABC):
    @abstractmethod
    def read_parquet(self, path: str) -> pd.DataFrame: ...

    @abstractmethod
    def write_parquet(self, df: pd.DataFrame, path: str) -> None: ...

    @abstractmethod
    def read_csv(self, path: str, **kwargs) -> pd.DataFrame: ...

    @abstractmethod
    def write_csv(self, df: pd.DataFrame, path: str) -> None: ...

    @abstractmethod
    def read_json(self, path: str) -> Union[list, dict]: ...

    @abstractmethod
    def write_json(self, data: Union[list, dict], path: str) -> None: ...

    @abstractmethod
    def read_bytes(self, path: str) -> bytes: ...

    @abstractmethod
    def write_bytes(self, data: bytes, path: str) -> None: ...

    @abstractmethod
    def exists(self, path: str) -> bool: ...

    @abstractmethod
    def list_files(self, prefix: str, suffix: str = None) -> list[str]: ...

    @abstractmethod
    def makedirs(self, path: str) -> None: ...

    @abstractmethod
    def join(self, *parts: str) -> str: ...


class LocalStorage(StorageAdapter):
    def __init__(self, base_path: str):
        self.base_path = base_path

    def _resolve(self, path: str) -> Path:
        if os.path.isabs(path):
            return Path(path)
        return Path(self.base_path) / path

    def read_parquet(self, path: str) -> pd.DataFrame:
        return pd.read_parquet(self._resolve(path), engine="pyarrow")

    def write_parquet(self, df: pd.DataFrame, path: str) -> None:
        resolved = self._resolve(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(resolved, engine="pyarrow", index=False)

    def read_csv(self, path: str, **kwargs) -> pd.DataFrame:
        return pd.read_csv(self._resolve(path), **kwargs)

    def write_csv(self, df: pd.DataFrame, path: str) -> None:
        resolved = self._resolve(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(resolved, index=False)

    def read_json(self, path: str) -> Union[list, dict]:
        with open(self._resolve(path), "r", encoding="utf-8") as f:
            return json.load(f)

    def write_json(self, data: Union[list, dict], path: str) -> None:
        resolved = self._resolve(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with open(resolved, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def read_bytes(self, path: str) -> bytes:
        with open(self._resolve(path), "rb") as f:
            return f.read()

    def write_bytes(self, data: bytes, path: str) -> None:
        resolved = self._resolve(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with open(resolved, "wb") as f:
            f.write(data)

    def exists(self, path: str) -> bool:
        return self._resolve(path).exists()

    def list_files(self, prefix: str, suffix: str = None) -> list[str]:
        base = self._resolve(prefix)
        if not base.exists():
            return []
        results = [str(p) for p in base.rglob("*") if p.is_file()]
        if suffix:
            results = [p for p in results if p.endswith(suffix)]
        return results

    def makedirs(self, path: str) -> None:
        self._resolve(path).mkdir(parents=True, exist_ok=True)

    def join(self, *parts: str) -> str:
        return os.path.join(*parts)


class S3Storage(StorageAdapter):
    def __init__(self, bucket: str, prefix: str, region: str, profile: str = None):
        try:
            import boto3
        except ImportError:
            raise ImportError("boto3 is required for S3Storage: pip install boto3")
        import boto3

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.region = region
        session = boto3.Session(profile_name=profile) if profile else boto3.Session()
        self.s3 = session.client("s3", region_name=region)

    def _key(self, path: str) -> str:
        path = path.lstrip("/")
        return f"{self.prefix}/{path}" if self.prefix else path

    def _ensure_prefix(self, key: str) -> None:
        # S3 has no real directories; uploading to a key implicitly creates the path
        pass

    def read_parquet(self, path: str) -> pd.DataFrame:
        buf = io.BytesIO(self.read_bytes(path))
        return pd.read_parquet(buf, engine="pyarrow")

    def write_parquet(self, df: pd.DataFrame, path: str) -> None:
        buf = io.BytesIO()
        df.to_parquet(buf, engine="pyarrow", index=False)
        buf.seek(0)
        self.s3.put_object(Bucket=self.bucket, Key=self._key(path), Body=buf.read())

    def read_csv(self, path: str, **kwargs) -> pd.DataFrame:
        buf = io.BytesIO(self.read_bytes(path))
        return pd.read_csv(buf, **kwargs)

    def write_csv(self, df: pd.DataFrame, path: str) -> None:
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        self.s3.put_object(
            Bucket=self.bucket,
            Key=self._key(path),
            Body=buf.getvalue().encode("utf-8"),
        )

    def read_json(self, path: str) -> Union[list, dict]:
        return json.loads(self.read_bytes(path).decode("utf-8"))

    def write_json(self, data: Union[list, dict], path: str) -> None:
        self.s3.put_object(
            Bucket=self.bucket,
            Key=self._key(path),
            Body=json.dumps(data, indent=2).encode("utf-8"),
        )

    def read_bytes(self, path: str) -> bytes:
        resp = self.s3.get_object(Bucket=self.bucket, Key=self._key(path))
        return resp["Body"].read()

    def write_bytes(self, data: bytes, path: str) -> None:
        self.s3.put_object(Bucket=self.bucket, Key=self._key(path), Body=data)

    def exists(self, path: str) -> bool:
        try:
            import botocore
        except ImportError:
            raise ImportError("boto3/botocore is required: pip install boto3")
        import botocore.exceptions

        try:
            self.s3.head_object(Bucket=self.bucket, Key=self._key(path))
            return True
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

    def list_files(self, prefix: str, suffix: str = None) -> list[str]:
        full_prefix = self._key(prefix)
        paginator = self.s3.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if suffix is None or key.endswith(suffix):
                    keys.append(key)
        return keys

    def makedirs(self, path: str) -> None:
        # S3 is a flat key-value store; no-op
        pass

    def join(self, *parts: str) -> str:
        return "/".join(p.strip("/") for p in parts)


class AzureStorage(StorageAdapter):
    def __init__(
        self,
        account_name: str,
        container: str,
        prefix: str,
        credential=None,
    ):
        try:
            from azure.storage.filedatalake import DataLakeServiceClient
        except ImportError:
            raise ImportError(
                "azure-storage-file-datalake is required for AzureStorage: "
                "pip install azure-storage-file-datalake"
            )
        from azure.storage.filedatalake import DataLakeServiceClient

        self.container = container
        self.prefix = prefix.strip("/")

        if credential is None:
            try:
                from azure.identity import DefaultAzureCredential
            except ImportError:
                raise ImportError(
                    "azure-identity is required when no credential is provided: "
                    "pip install azure-identity"
                )
            credential = DefaultAzureCredential()

        # credential may be a connection string, SAS token string, or a credential object
        if isinstance(credential, str) and credential.startswith("DefaultEndpointsProtocol"):
            self._client = DataLakeServiceClient.from_connection_string(credential)
        elif isinstance(credential, str):
            # treat as SAS token
            self._client = DataLakeServiceClient(
                account_url=f"https://{account_name}.dfs.core.windows.net",
                credential=credential,
            )
        else:
            self._client = DataLakeServiceClient(
                account_url=f"https://{account_name}.dfs.core.windows.net",
                credential=credential,
            )

        self._fs = self._client.get_file_system_client(container)

    def _path(self, path: str) -> str:
        path = path.lstrip("/")
        return f"{self.prefix}/{path}" if self.prefix else path

    def _file_client(self, path: str):
        return self._fs.get_file_client(self._path(path))

    def _ensure_parent(self, path: str) -> None:
        parent = "/".join(self._path(path).split("/")[:-1])
        if parent:
            try:
                self._fs.create_directory(parent)
            except Exception:
                pass

    def read_parquet(self, path: str) -> pd.DataFrame:
        buf = io.BytesIO(self.read_bytes(path))
        return pd.read_parquet(buf, engine="pyarrow")

    def write_parquet(self, df: pd.DataFrame, path: str) -> None:
        buf = io.BytesIO()
        df.to_parquet(buf, engine="pyarrow", index=False)
        self.write_bytes(buf.getvalue(), path)

    def read_csv(self, path: str, **kwargs) -> pd.DataFrame:
        buf = io.BytesIO(self.read_bytes(path))
        return pd.read_csv(buf, **kwargs)

    def write_csv(self, df: pd.DataFrame, path: str) -> None:
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        self.write_bytes(buf.getvalue().encode("utf-8"), path)

    def read_json(self, path: str) -> Union[list, dict]:
        return json.loads(self.read_bytes(path).decode("utf-8"))

    def write_json(self, data: Union[list, dict], path: str) -> None:
        self.write_bytes(json.dumps(data, indent=2).encode("utf-8"), path)

    def read_bytes(self, path: str) -> bytes:
        fc = self._file_client(path)
        download = fc.download_file()
        return download.readall()

    def write_bytes(self, data: bytes, path: str) -> None:
        self._ensure_parent(path)
        fc = self._file_client(path)
        fc.upload_data(data, overwrite=True)

    def exists(self, path: str) -> bool:
        try:
            self._file_client(path).get_file_properties()
            return True
        except Exception:
            return False

    def list_files(self, prefix: str, suffix: str = None) -> list[str]:
        full_prefix = self._path(prefix)
        paths = self._fs.get_paths(path=full_prefix, recursive=True)
        results = []
        for p in paths:
            if not p.is_directory:
                name = p.name
                if suffix is None or name.endswith(suffix):
                    results.append(name)
        return results

    def makedirs(self, path: str) -> None:
        self._fs.create_directory(self._path(path))

    def join(self, *parts: str) -> str:
        return "/".join(p.strip("/") for p in parts)


class GCSStorage(StorageAdapter):
    def __init__(self, bucket: str, prefix: str, project: str = None):
        try:
            from google.cloud import storage as gcs
        except ImportError:
            raise ImportError(
                "google-cloud-storage is required for GCSStorage: "
                "pip install google-cloud-storage"
            )
        from google.cloud import storage as gcs

        self.bucket_name = bucket
        self.prefix = prefix.strip("/")
        client = gcs.Client(project=project)
        self._bucket = client.bucket(bucket)

    def _blob_name(self, path: str) -> str:
        path = path.lstrip("/")
        return f"{self.prefix}/{path}" if self.prefix else path

    def _blob(self, path: str):
        return self._bucket.blob(self._blob_name(path))

    def read_parquet(self, path: str) -> pd.DataFrame:
        buf = io.BytesIO(self.read_bytes(path))
        return pd.read_parquet(buf, engine="pyarrow")

    def write_parquet(self, df: pd.DataFrame, path: str) -> None:
        buf = io.BytesIO()
        df.to_parquet(buf, engine="pyarrow", index=False)
        self.write_bytes(buf.getvalue(), path)

    def read_csv(self, path: str, **kwargs) -> pd.DataFrame:
        buf = io.BytesIO(self.read_bytes(path))
        return pd.read_csv(buf, **kwargs)

    def write_csv(self, df: pd.DataFrame, path: str) -> None:
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        self.write_bytes(buf.getvalue().encode("utf-8"), path)

    def read_json(self, path: str) -> Union[list, dict]:
        return json.loads(self.read_bytes(path).decode("utf-8"))

    def write_json(self, data: Union[list, dict], path: str) -> None:
        self.write_bytes(json.dumps(data, indent=2).encode("utf-8"), path)

    def read_bytes(self, path: str) -> bytes:
        return self._blob(path).download_as_bytes()

    def write_bytes(self, data: bytes, path: str) -> None:
        self._blob(path).upload_from_string(data)

    def exists(self, path: str) -> bool:
        return self._blob(path).exists()

    def list_files(self, prefix: str, suffix: str = None) -> list[str]:
        full_prefix = self._blob_name(prefix)
        blobs = self._bucket.list_blobs(prefix=full_prefix)
        results = []
        for b in blobs:
            if suffix is None or b.name.endswith(suffix):
                results.append(b.name)
        return results

    def makedirs(self, path: str) -> None:
        # GCS is a flat namespace; no-op
        pass

    def join(self, *parts: str) -> str:
        return "/".join(p.strip("/") for p in parts)


def get_storage(config: dict) -> StorageAdapter:
    provider = config["storage"]["provider"].lower()
    cfg = config["storage"].get(provider, {})

    if provider == "local":
        return LocalStorage(base_path=cfg["base_path"])

    if provider == "s3":
        return S3Storage(
            bucket=cfg["bucket"],
            prefix=cfg.get("prefix", ""),
            region=cfg["region"],
            profile=cfg.get("profile"),
        )

    if provider == "azure":
        return AzureStorage(
            account_name=cfg["account_name"],
            container=cfg["container"],
            prefix=cfg.get("prefix", ""),
            credential=cfg.get("credential"),
        )

    if provider == "gcs":
        return GCSStorage(
            bucket=cfg["bucket"],
            prefix=cfg.get("prefix", ""),
            project=cfg.get("project"),
        )

    raise ValueError(f"Unknown storage provider: {provider!r}. Must be one of: local, s3, azure, gcs")
