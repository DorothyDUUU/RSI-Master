"""Shared CLI output helpers.

CLI 层的所有命令都通过这里输出, 保证:
    - 默认输出为可读文本
    - ``--json`` 时输出结构化 JSON (方便 agent bash 调用后解析)
    - 错误统一走 :func:`emit_error`, 返回非零退出码
"""

from __future__ import annotations

import json
import sys
from typing import Any


def emit_json(payload: Any) -> None:
    """Print ``payload`` as pretty JSON to stdout."""
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def emit_yaml_like(payload: Any, indent: int = 0) -> None:
    """Render a nested dict/list as a light YAML-ish tree.

    仅用于 CLI 的默认可读输出。不追求严格 YAML 规范, 只为看起来分层清晰。
    """
    prefix = "  " * indent
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, (dict, list)) and value:
                print(f"{prefix}{key}:")
                emit_yaml_like(value, indent + 1)
            else:
                print(f"{prefix}{key}: {_scalar(value)}")
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, (dict, list)):
                print(f"{prefix}-")
                emit_yaml_like(item, indent + 1)
            else:
                print(f"{prefix}- {_scalar(item)}")
    else:
        print(f"{prefix}{_scalar(payload)}")


def _scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value if value else '""'
    return str(value)


def emit_error(message: str, *, code: int = 1) -> int:
    """Print ``message`` to stderr and return the exit code (for main()).

    CLI 处理器不 :func:`sys.exit`, 由 :mod:`expos.cli.main` 统一 return, 便于测试。
    """
    print(f"[expos] error: {message}", file=sys.stderr)
    return code
