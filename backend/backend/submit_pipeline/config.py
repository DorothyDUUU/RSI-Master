"""Runtime config for the submit-pipeline CLI (values not baked into constants).

``config.yaml`` lives next to this file.  It carries only the few knobs that
depend on *deployment* (where gpu_backend lives, how long to poll), never the
pipeline structure itself (that is ``constants.py``).

``get_config()`` merges the YAML over built-in defaults; a missing/invalid file
degrades to defaults (best-effort, never raises — the pipeline must still submit
on an unconfigured node).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_DEFAULTS: dict[str, Any] = {
    "gpu_backend": {
        "url": "",
        "submit_timeout_sec": 60,
    },
    "eval": {
        "poll_interval_sec": 30,
        "script_timeout_sec": 7200,
    },
}

_CONFIG_YAML = Path(__file__).with_name("config.yaml")


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_config() -> dict[str, Any]:
    """Return merged config (YAML over defaults).  Never raises."""
    if not _CONFIG_YAML.is_file():
        return dict(_DEFAULTS)
    try:
        loaded = yaml.safe_load(_CONFIG_YAML.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("submit_pipeline config parse failed: %s; using defaults", exc)
        return dict(_DEFAULTS)
    if not isinstance(loaded, dict):
        return dict(_DEFAULTS)
    return _deep_merge(_DEFAULTS, loaded)


def get_gpu_backend_url() -> str:
    """gpu_backend URL from pipeline config, else the rollout gpu_backend config."""
    cfg = get_config()
    url = str(cfg.get("gpu_backend", {}).get("url", "") or "").strip()
    if url:
        return url
    # Lazy import keeps config.py importable even when gpu_backend is absent.
    try:
        from backend.gpu_backend.config import get_backend_url

        return get_backend_url()
    except Exception as exc:  # noqa: BLE001
        logger.warning("get_gpu_backend_url fell back empty: %s", exc)
        return ""


__all__ = ["get_config", "get_gpu_backend_url"]
