"""Config loader — reads config.yaml from the package directory.

Priority: config.yaml > hardcoded defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config() -> dict[str, Any]:
    """Load the YAML config file, returning a nested dict."""
    if not _CONFIG_PATH.exists():
        return {}
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}


def get_backend_url() -> str:
    """Resolve the backend URL.

    Priority: config.yaml ``backend.host:port`` > ``http://localhost:8090``.
    """
    cfg = load_config()
    be = cfg.get("backend", {})
    host = be.get("host", "localhost")
    port = be.get("port", 8090)
    return f"http://{host}:{port}"


def get_default_timeout() -> int:
    cfg = load_config()
    return cfg.get("defaults", {}).get("exec_timeout", 300)


def get_default_max_mem_quota() -> int:
    cfg = load_config()
    return cfg.get("defaults", {}).get("max_mem_quota", 3000)
