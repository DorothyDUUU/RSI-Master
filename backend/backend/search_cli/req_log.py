"""Request-level JSONL logging for the search backend (traj_analysis T8).

落点由 ``SEARCH_REQ_LOG`` env 控制 —— run 脚本里 ``SEARCH_REQ_LOG=${POD_LOG}/search_requests.jsonl``
即可让分析侧 (``utils_tools/traj_analysis/scan_run.py`` G5) 拿到请求级延迟。
两条路径共用:

- server 代理路径: ``server.py`` 的 ``@app.middleware("http")`` 记每个非 /health POST;
- CLI 本地回退路径: ``cli.py::_dispatch`` 在 health 探测失败、CLI 进程内直接 ``ops.execute`` 时记一行
  (``via=local``), 否则 server 不可达时搜索请求会完全无记录。

日志写入永不影响搜索本身 —— 文件/权限/JSON 出错一律吞掉。
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any

_LOCK = threading.Lock()


def req_log_path() -> str:
    """SEARCH_REQ_LOG env; 空串 = 关闭日志 (向后兼容)。"""
    return os.getenv("SEARCH_REQ_LOG", "").strip()


def log_request_record(rec: dict[str, Any]) -> None:
    """Append one request record; no-op when SEARCH_REQ_LOG is unset."""
    path = req_log_path()
    if not path:
        return
    try:
        with _LOCK:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
    except Exception:  # noqa: BLE001 — 日志失败绝不影响搜索
        pass
