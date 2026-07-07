"""
Load config.yaml with ${ENV_VAR} interpolation.
Falls back to settings.py values when config.yaml is missing.
"""
import os
import re
from pathlib import Path

try:
    import yaml
    _YAML_OK = True
except ImportError:
    _YAML_OK = False

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config.yaml"

_ENVAR_RE = re.compile(r"\$\{([^}]+)\}")


def _interpolate(value: str) -> str:
    """Replace ${VAR} and ${VAR:-default} with environment values."""
    def _replace(m):
        spec = m.group(1)
        if ":-" in spec:
            var, default = spec.split(":-", 1)
        else:
            var, default = spec, ""
        return os.environ.get(var.strip(), default)
    return _ENVAR_RE.sub(_replace, value)


def _walk(obj):
    """Recursively interpolate all string values in a dict/list."""
    if isinstance(obj, dict):
        return {k: _walk(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk(i) for i in obj]
    if isinstance(obj, str):
        return _interpolate(obj)
    return obj


def load_config(path: str | Path = None) -> dict:
    """Return the fully-interpolated config dict."""
    target = Path(path) if path else CONFIG_PATH
    if not target.exists():
        return _default_config()
    if not _YAML_OK:
        raise ImportError("pyyaml is required to load config.yaml. Run: pip install pyyaml")
    with open(target, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _walk(raw or {})


def _default_config() -> dict:
    """Minimal config built from env vars when config.yaml is absent."""
    return {
        "storage":  {"provider": "local", "local": {"base_path": str(ROOT / "data")}},
        "compute":  {"provider": "local", "local": {"driver_memory": "4g", "executor_memory": "4g",
                                                      "shuffle_partitions": 50, "parallelism": 8}},
        "llm":      {"provider": "anthropic", "max_tokens": 2048,
                     "anthropic": {"api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
                                   "model": "claude-sonnet-4-6"}},
        "pipeline": {"n_customers": 5_000_000, "n_transactions": 25_000_000,
                     "ml_sample_size": 200_000, "max_segments": 6, "random_seed": 42},
        "paths":    {"raw": "raw", "staging": "staging", "processed": "processed",
                     "models": "models", "reports": "reports", "uploads": "uploads"},
    }


# Module-level singleton — call load_config() once at startup
_CONFIG: dict | None = None


def get_config() -> dict:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_config()
    return _CONFIG


def reload_config() -> dict:
    global _CONFIG
    _CONFIG = load_config()
    return _CONFIG
