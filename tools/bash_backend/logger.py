"""Execute 请求/响应的 JSONL 持久化日志。

每次 execute 调用记录两行：
1. 请求日志（调用时立刻写入）
2. 响应日志（执行完成后写入）
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any


class ExecuteLogger:
    """Execute 请求/响应的 JSONL 日志记录器。"""

    def __init__(self, data_dir: str = ""):
        """
        Args:
            data_dir: 日志存储目录。为空时使用 tools/bash_backend/data/。
        """
        if not data_dir:
            data_dir = str(Path(__file__).parent / "data")
        self._data_dir = data_dir
        self._request_file = os.path.join(data_dir, "execute_requests.jsonl")
        self._response_file = os.path.join(data_dir, "execute_responses.jsonl")
        self._lock = threading.Lock()

        os.makedirs(data_dir, exist_ok=True)

    def log_request(
        self,
        process_id: str,
        command: str,
        agent_workspace: str,
        timeout: float,
        meta_data: dict[str, Any] | None = None,
    ) -> None:
        """记录 execute 请求（调用时立刻写入）。"""
        record = {
            "process_id": process_id,
            "timestamp": datetime.now().isoformat(),
            "epoch": time.time(),
            "command": command,
            "agent_workspace": agent_workspace,
            "timeout": timeout,
            "meta_data": meta_data,
        }
        self._append(self._request_file, record)

    def log_response(
        self,
        process_id: str,
        stdout: str,
        stderr: str,
        exit_code: int,
        working_dir: str,
        filtered: bool,
        filter_reason: str | None,
        duration_seconds: float,
        command: str = "",
    ) -> None:
        """记录 execute 响应（执行完成后写入）。"""
        record = {
            "process_id": process_id,
            "timestamp": datetime.now().isoformat(),
            "epoch": time.time(),
            "command": command,
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
            "working_dir": working_dir,
            "filtered": filtered,
            "filter_reason": filter_reason,
            "duration_seconds": duration_seconds,
            "stdout_length": len(stdout),
            "stderr_length": len(stderr),
        }
        self._append(self._response_file, record)

    def _append(self, filepath: str, record: dict) -> None:
        """线程安全地追加一行 JSON 到文件。"""
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
