"""进程跟踪器。

跟踪所有通过 server 执行的进程，支持查询和 kill。
"""

from __future__ import annotations

import os
import signal
import threading
import time
from dataclasses import dataclass
from enum import Enum


class ProcessStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    KILLED = "killed"
    FILTERED = "filtered"


@dataclass
class ProcessInfo:
    """运行中进程的信息。"""
    process_id: str
    pgid: int | None
    pid: int | None
    command: str
    agent_workspace: str
    start_time: float
    end_time: float | None = None
    status: ProcessStatus = ProcessStatus.RUNNING
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    filter_reason: str | None = None


class ProcessTracker:
    """进程跟踪器。

    跟踪所有通过 server 执行的进程，支持查询和 kill。
    """

    def __init__(self):
        self._processes: dict[str, ProcessInfo] = {}
        self._lock = threading.Lock()

    def register(self, info: ProcessInfo) -> None:
        """注册一个新进程。"""
        with self._lock:
            self._processes[info.process_id] = info

    def update(
        self,
        process_id: str,
        *,
        status: ProcessStatus | None = None,
        exit_code: int | None = None,
        stdout: str | None = None,
        stderr: str | None = None,
        pgid: int | None = None,
        pid: int | None = None,
        end_time: float | None = None,
        filter_reason: str | None = None,
    ) -> None:
        """更新进程信息。"""
        with self._lock:
            info = self._processes.get(process_id)
            if info is None:
                return
            if status is not None:
                info.status = status
            if exit_code is not None:
                info.exit_code = exit_code
            if stdout is not None:
                info.stdout = stdout
            if stderr is not None:
                info.stderr = stderr
            if pgid is not None:
                info.pgid = pgid
            if pid is not None:
                info.pid = pid
            if end_time is not None:
                info.end_time = end_time
            if filter_reason is not None:
                info.filter_reason = filter_reason

    def get(self, process_id: str) -> ProcessInfo | None:
        """获取进程信息。"""
        with self._lock:
            return self._processes.get(process_id)

    def get_all(self, status_filter: ProcessStatus | None = None) -> list[ProcessInfo]:
        """获取所有进程信息。"""
        with self._lock:
            if status_filter:
                return [p for p in self._processes.values() if p.status == status_filter]
            return list(self._processes.values())

    def get_active(self) -> list[ProcessInfo]:
        """获取所有活跃（运行中）的进程。"""
        return self.get_all(ProcessStatus.RUNNING)

    def kill_process(self, process_id: str) -> tuple[bool, str]:
        """Kill 指定进程。

        Returns:
            (success, message)
        """
        with self._lock:
            info = self._processes.get(process_id)
            if info is None:
                return False, f"进程 {process_id} 不存在"

            if info.status != ProcessStatus.RUNNING:
                return False, f"进程 {process_id} 已结束 (状态: {info.status.value})"

            if info.pgid is not None:
                try:
                    os.killpg(info.pgid, signal.SIGKILL)
                    info.status = ProcessStatus.KILLED
                    info.end_time = time.time()
                    return True, f"已 kill 进程组 {info.pgid}"
                except (ProcessLookupError, PermissionError) as e:
                    return False, f"kill 失败: {e}"
            elif info.pid is not None:
                try:
                    os.kill(info.pid, signal.SIGKILL)
                    info.status = ProcessStatus.KILLED
                    info.end_time = time.time()
                    return True, f"已 kill 进程 {info.pid}"
                except (ProcessLookupError, PermissionError) as e:
                    return False, f"kill 失败: {e}"
            else:
                return False, "进程没有 pgid 或 pid 信息"

    def kill_all(self) -> tuple[int, list[str]]:
        """Kill 所有活跃进程。

        Returns:
            (killed_count, messages)
        """
        messages = []
        killed = 0
        active = self.get_active()
        for info in active:
            ok, msg = self.kill_process(info.process_id)
            if ok:
                killed += 1
            messages.append(f"[{info.process_id}] {msg}")
        return killed, messages

    def cleanup_old(self, max_age_seconds: float = 3600) -> int:
        """清理已完成的旧进程记录。

        Returns:
            清理的记录数
        """
        now = time.time()
        to_remove = []
        with self._lock:
            for pid, info in self._processes.items():
                if info.status != ProcessStatus.RUNNING:
                    if info.end_time and (now - info.end_time) > max_age_seconds:
                        to_remove.append(pid)
            for pid in to_remove:
                del self._processes[pid]
        return len(to_remove)
