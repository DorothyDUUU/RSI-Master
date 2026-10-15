from __future__ import annotations

from copy import deepcopy


SECRET_KEYS = {"openai_api_token", "anthropic_api_token"}


def deep_merge(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def redact(config: dict) -> dict:
    result = deepcopy(config)
    general = result.setdefault("general", {})
    for key in SECRET_KEYS:
        if key in general:
            general[key] = f"<{key.upper()}>"
    return result
