"""Environment loading helpers for RSI-Master entrypoints."""
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Iterable

try:
    from dotenv import dotenv_values
except ImportError:
    dotenv_values = None  # type: ignore[misc, assignment]


def _optional_path(env_var: str) -> Path | None:
    """Read an opt-in path from the environment, or ``None`` when unset.

    Deployment-specific env files are never hardcoded, so a fresh checkout
    loads only its own ``.env`` and nothing else.
    """
    raw = os.environ.get(env_var, "").strip()
    return Path(raw).expanduser() if raw else None


# Optional shared .env, opt-in via EVOMASTER_SHARED_ENV.
SHARED_ENV_PATH: Path | None = _optional_path("EVOMASTER_SHARED_ENV")
# Optional legacy MCP config JSON, opt-in via EVOMASTER_LEGACY_MCP_CONFIG.
LEGACY_MCP_CONFIG_PATH: Path | None = _optional_path("EVOMASTER_LEGACY_MCP_CONFIG")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_env_paths() -> tuple[Path, ...]:
    return tuple(p for p in (project_root() / ".env", SHARED_ENV_PATH) if p is not None)


def default_json_env_paths() -> tuple[Path, ...]:
    return tuple(p for p in (LEGACY_MCP_CONFIG_PATH,) if p is not None)


def _parse_simple_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _iter_json_env_maps(value: object):
    if isinstance(value, dict):
        env = value.get("env")
        if isinstance(env, dict):
            yield env
        for child in value.values():
            yield from _iter_json_env_maps(child)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_json_env_maps(item)


def load_env_file(path: str | Path, *, override: bool = False) -> list[str]:
    """Load a dotenv-style file without overriding existing env by default."""
    env_path = Path(path).expanduser()
    if not env_path.exists() or not env_path.is_file():
        return []

    if dotenv_values is not None:
        parsed = {
            key: value
            for key, value in dotenv_values(env_path).items()
            if key and value is not None
        }
    else:
        parsed = _parse_simple_env(env_path)

    loaded: list[str] = []
    for key, value in parsed.items():
        if override or key not in os.environ:
            os.environ[key] = str(value)
            loaded.append(key)
    return loaded


def load_json_env_file(path: str | Path, *, override: bool = False) -> list[str]:
    """Load every nested JSON object named ``env`` into process env."""
    env_path = Path(path).expanduser()
    if not env_path.exists() or not env_path.is_file():
        return []

    payload = json.loads(env_path.read_text(encoding="utf-8"))
    loaded: list[str] = []
    for env_map in _iter_json_env_maps(payload):
        for key, value in env_map.items():
            if not key or value is None:
                continue
            if override or key not in os.environ:
                os.environ[str(key)] = str(value)
                loaded.append(str(key))
    return loaded


def load_evomaster_env(
    extra_paths: Iterable[str | Path] = (),
    *,
    override: bool = False,
) -> list[str]:
    """Load env files in priority order.

    Priority is: existing process environment, explicit extra paths, the repo's
    ``.env``, the optional shared .env (``EVOMASTER_SHARED_ENV``), then optional
    legacy MCP JSON env blocks (``EVOMASTER_LEGACY_MCP_CONFIG``).
    """
    loaded: list[str] = []
    seen: set[Path] = set()
    for path in [*extra_paths, *default_env_paths()]:
        env_path = Path(path).expanduser()
        try:
            resolved = env_path.resolve()
        except OSError:
            resolved = env_path
        if resolved in seen:
            continue
        seen.add(resolved)
        loaded.extend(load_env_file(env_path, override=override))
    for path in default_json_env_paths():
        loaded.extend(load_json_env_file(path, override=override))
    return loaded
