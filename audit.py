"""
Security audit log — append-only JSONL file.
Every security-relevant action (uploads, config changes, violations, pipeline runs)
is logged with a timestamp so the organisation has a traceable history.
"""

import json
import os
import datetime
import traceback
import threading

_LOCK = threading.Lock()
_LOG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "outputs", "reports", "security_audit.jsonl"
)


def _write(entry: dict):
    os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
    entry["ts"] = datetime.datetime.utcnow().isoformat() + "Z"
    line = json.dumps(entry, ensure_ascii=False, default=str)
    with _LOCK:
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def log_upload(table_name: str, filename: str, rows: int, size_mb: float):
    _write({
        "event": "DATA_UPLOAD",
        "table": table_name,
        "filename": filename,
        "rows": rows,
        "size_mb": round(size_mb, 2),
    })


def log_schema_upload(tables: list[str], source: str):
    _write({
        "event": "SCHEMA_UPLOAD",
        "tables": tables,
        "source": source,
    })


def log_config_change(section: str, provider: str, source: str):
    _write({
        "event": "CONFIG_CHANGE",
        "section": section,
        "provider": provider,
        "source": source,
    })


def log_env_write(keys_set: list[str], source: str):
    _write({
        "event": "ENV_WRITE",
        "keys": keys_set,
        "source": source,
    })


def log_pipeline_run(triggered_by: str = "dashboard"):
    _write({
        "event": "PIPELINE_RUN",
        "triggered_by": triggered_by,
    })


def log_violation(violation_type: str, detail: str, source: str = "unknown"):
    _write({
        "event":    "SECURITY_VIOLATION",
        "type":     violation_type,
        "detail":   detail[:500],
        "source":   source,
    })


def log_auth(success: bool, method: str = "password"):
    _write({
        "event":   "AUTH",
        "success": success,
        "method":  method,
    })


def recent_violations(n: int = 20) -> list[dict]:
    if not os.path.exists(_LOG_PATH):
        return []
    entries = []
    with open(_LOG_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
                if e.get("event") == "SECURITY_VIOLATION":
                    entries.append(e)
            except json.JSONDecodeError:
                continue
    return entries[-n:]


def recent_events(n: int = 50) -> list[dict]:
    if not os.path.exists(_LOG_PATH):
        return []
    lines = []
    with open(_LOG_PATH, encoding="utf-8") as f:
        for line in f:
            lines.append(line)
    entries = []
    for line in lines[-n:]:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return list(reversed(entries))
