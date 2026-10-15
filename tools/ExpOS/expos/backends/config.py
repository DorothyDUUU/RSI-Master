"""YAML config loader for the ExpOS backend.

优先级:
    1. 调用方显式传入的 ``config_path``
    2. 包目录下 ``config.yaml``

不走环境变量 (per user directive)。想覆盖 URL 时, 拷贝 config.yaml 到别处
并把路径通过 ``load_backend_config(path=...)`` 传进来。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


@dataclass(frozen=True)
class BackendConfig:
    """Resolved backend configuration snapshot."""

    host: str
    port: int
    submit_timeout: float
    status_timeout: float
    connect_timeout: float
    trials_dir: str
    driver: str

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


def load_backend_config(path: str | Path | None = None) -> BackendConfig:
    """Load :class:`BackendConfig` from ``config.yaml``.

    Parameters
    ----------
    path:
        Explicit path to a YAML file. When ``None`` (default), use the
        packaged ``config.yaml`` next to this module.
    """
    cfg_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = {}
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    backend = raw.get("backend") or {}
    timeouts = raw.get("timeouts") or {}
    server = raw.get("server") or {}
    return BackendConfig(
        host=str(backend.get("host", "localhost")),
        port=int(backend.get("port", 8990)),
        submit_timeout=float(timeouts.get("submit", 7200)),
        status_timeout=float(timeouts.get("status", 30)),
        connect_timeout=float(timeouts.get("connect", 10)),
        trials_dir=str(server.get("trials_dir", "state/backend_trials")),
        driver=str(server.get("driver", "stub")),
    )
