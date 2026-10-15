"""Config loader — reads config.yaml from the package directory.

Priority: env vars (``SEARCH_URL`` / ``SEARCH_HOST`` / ``SEARCH_PORT``) >
config.yaml > hardcoded defaults.
"""

from __future__ import annotations

import os
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


def get_search_url() -> str:
    """Resolve the server URL (client side).

    Priority: ``SEARCH_URL`` env var > config.yaml ``backend.host:port`` >
    ``http://localhost:8991``.
    """
    env_url = os.getenv("SEARCH_URL", "").strip().rstrip("/")
    if env_url:
        return env_url
    cfg = load_config()
    be = cfg.get("backend", {})
    host = os.getenv("SEARCH_HOST", "") or be.get("host", "localhost")
    port = os.getenv("SEARCH_PORT", "") or be.get("port", 8991)
    return f"http://{host}:{port}"


def get_probe_timeout() -> float:
    """Health-check probe timeout in seconds."""
    cfg = load_config()
    return float(cfg.get("timeouts", {}).get("probe", 3.0))


def get_request_timeout() -> float:
    """Search request timeout in seconds."""
    cfg = load_config()
    return float(cfg.get("timeouts", {}).get("request", 120.0))


def get_default_max_results() -> int:
    """Default max results for search verbs that support it."""
    cfg = load_config()
    return int(cfg.get("defaults", {}).get("max_results", 10))
