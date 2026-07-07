import os
import re

import pandas as pd

from src.security.validators import URLValidator, SecurityError


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [
        re.sub(r"[^a-z0-9]+", "_", col.strip().lower()).strip("_")
        for col in df.columns
    ]
    return df


def _interpolate_env(value):
    if not isinstance(value, str):
        return value

    def replace(match):
        var = match.group(1)
        if ":-" in var:
            name, default = var.split(":-", 1)
            return os.environ.get(name.strip(), default)
        return os.environ.get(var.strip(), match.group(0))

    return re.sub(r"\$\{([^}]+)\}", replace, value)


def _interpolate_dict(d: dict) -> dict:
    return {k: _interpolate_env(v) for k, v in d.items()}


def _extract_data(response_json, data_path: str):
    if not data_path:
        if isinstance(response_json, list):
            return response_json
        return [response_json]

    node = response_json
    for key in data_path.split("."):
        if isinstance(node, dict):
            node = node.get(key)
        else:
            return []
        if node is None:
            return []

    if isinstance(node, list):
        return node
    return [node]


def _build_auth(auth_cfg: dict, headers: dict, params: dict):
    auth_type = auth_cfg.get("type", "").lower()

    if auth_type == "basic":
        try:
            from requests.auth import HTTPBasicAuth
        except ImportError:
            raise ImportError("pip install requests")
        username = _interpolate_env(auth_cfg.get("username", ""))
        password = _interpolate_env(auth_cfg.get("password", ""))
        return HTTPBasicAuth(username, password), headers, params

    elif auth_type == "bearer":
        token = _interpolate_env(auth_cfg.get("token", ""))
        headers = dict(headers)
        headers["Authorization"] = f"Bearer {token}"
        return None, headers, params

    elif auth_type == "api_key":
        key_value = _interpolate_env(auth_cfg.get("key", ""))
        key_name = auth_cfg.get("key_name", "api_key")
        location = auth_cfg.get("in", "header")
        if location == "header":
            headers = dict(headers)
            headers[key_name] = key_value
        else:
            params = dict(params)
            params[key_name] = key_value
        return None, headers, params

    return None, headers, params


class APIConnector:
    def load(self, source_cfg: dict) -> pd.DataFrame:
        try:
            import requests
        except ImportError:
            raise ImportError("pip install requests")

        url = source_cfg["url"]
        URLValidator.assert_ssrf_safe(url)
        method = source_cfg.get("method", "GET").upper()
        if method not in ("GET", "POST"):
            raise SecurityError(f"HTTP method '{method}' is not allowed. Only GET and POST are permitted.")
        raw_headers = source_cfg.get("headers") or {}
        raw_params = source_cfg.get("params") or {}
        body = source_cfg.get("body")
        auth_cfg = source_cfg.get("auth") or {}
        pagination_cfg = source_cfg.get("pagination") or {}
        data_path = source_cfg.get("data_path", "")
        timeout = source_cfg.get("timeout", 30)

        headers = _interpolate_dict(raw_headers)
        params = _interpolate_dict(raw_params)

        auth_obj = None
        if auth_cfg:
            auth_obj, headers, params = _build_auth(auth_cfg, headers, params)

        pagination_type = pagination_cfg.get("type", "none")
        page_param = pagination_cfg.get("page_param", "page")
        size_param = pagination_cfg.get("size_param", "per_page")
        size = pagination_cfg.get("size", 100)
        cursor_field = pagination_cfg.get("cursor_field", "next_cursor")
        cursor_param = pagination_cfg.get("cursor_param", "cursor")
        max_pages = pagination_cfg.get("max_pages", 100)

        all_records = []

        def do_request(req_params, req_body=None, req_url=None):
            target_url = req_url or url
            kwargs = {
                "headers": headers,
                "params": req_params,
                "timeout": timeout,
                "auth": auth_obj,
            }
            if method == "POST":
                kwargs["json"] = req_body or body
            return requests.request(method, target_url, **kwargs)

        if pagination_type == "none":
            resp = do_request(params)
            resp.raise_for_status()
            records = _extract_data(resp.json(), data_path)
            all_records.extend(records)

        elif pagination_type == "page":
            current_page = 1
            current_params = dict(params)
            current_params[size_param] = size
            for _ in range(max_pages):
                current_params[page_param] = current_page
                resp = do_request(current_params)
                resp.raise_for_status()
                records = _extract_data(resp.json(), data_path)
                if not records:
                    break
                all_records.extend(records)
                if len(records) < size:
                    break
                current_page += 1

        elif pagination_type == "offset":
            offset = 0
            current_params = dict(params)
            current_params[size_param] = size
            for _ in range(max_pages):
                current_params["offset"] = offset
                resp = do_request(current_params)
                resp.raise_for_status()
                records = _extract_data(resp.json(), data_path)
                if not records:
                    break
                all_records.extend(records)
                if len(records) < size:
                    break
                offset += size

        elif pagination_type == "cursor":
            current_params = dict(params)
            current_params[size_param] = size
            for _ in range(max_pages):
                resp = do_request(current_params)
                resp.raise_for_status()
                data = resp.json()
                records = _extract_data(data, data_path)
                if not records:
                    break
                all_records.extend(records)
                next_cursor = None
                node = data
                for key in cursor_field.split("."):
                    if isinstance(node, dict):
                        node = node.get(key)
                    else:
                        node = None
                        break
                next_cursor = node
                if not next_cursor:
                    break
                current_params[cursor_param] = next_cursor

        elif pagination_type == "link":
            current_url = url
            current_params = dict(params)
            for _ in range(max_pages):
                resp = do_request(current_params, req_url=current_url)
                resp.raise_for_status()
                records = _extract_data(resp.json(), data_path)
                if not records:
                    break
                all_records.extend(records)
                link_header = resp.headers.get("Link", "")
                next_url = None
                for part in link_header.split(","):
                    part = part.strip()
                    if 'rel="next"' in part:
                        match = re.search(r"<([^>]+)>", part)
                        if match:
                            next_url = match.group(1)
                            break
                if not next_url:
                    break
                URLValidator.assert_ssrf_safe(next_url)
                current_url = next_url
                current_params = {}

        if not all_records:
            return pd.DataFrame()

        df = pd.json_normalize(all_records)
        return _normalize_columns(df)
