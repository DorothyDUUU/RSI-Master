"""命令执行器。

封装 subprocess.Popen，支持资源限制和进程组管理。
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
from typing import Callable


def parse_memory_limit_to_bytes(limit_str: str | None) -> int | None:
    """将人类可读的内存限制字符串转换为字节数。"""
    if not limit_str:
        return None
    s = str(limit_str).strip().lower()
    multipliers = {"k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}
    if s and s[-1] in multipliers:
        return int(float(s[:-1]) * multipliers[s[-1]])
    try:
        return int(s)
    except ValueError:
        return None


class CommandExecutor:
    """命令执行器。

    封装 subprocess.Popen，支持资源限制和进程组管理。
    """

    def __init__(
        self,
        memory_limit: str | None = None,
        cpu_limit: float = 0.0,
        cpu_devices: str | None = None,
    ):
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.cpu_devices = cpu_devices

    def execute(
        self,
        command: str,
        timeout: float,
        workdir: str,
        env: dict[str, str] | None = None,
        on_started: Callable[[subprocess.Popen, int], None] | None = None,
    ) -> tuple[subprocess.Popen | None, int | None, str, str, int]:
        """执行命令并返回结果。

        Args:
            command: 要执行的命令
            timeout: 超时秒数
            workdir: 工作目录
            env: 额外环境变量
            on_started: 回调，在进程启动后立刻调用 (proc, pgid)

        Returns:
            (proc, pgid, stdout, stderr, exit_code)
            pgid 为 None 表示进程未成功启动
        """
        # 构建环境变量
        exec_env = os.environ.copy()
        if env:
            exec_env.update(env)

        # 构建最终命令（包装资源限制）
        final_command = command

        # CPU 亲和性
        if self.cpu_devices and os.name != "nt":
            final_command = f"taskset -c {self.cpu_devices} sh -c {shlex.quote(final_command)}"

        # prlimit 资源限制
        if os.name != "nt":
            prlimit_parts = ["prlimit"]
            if self.memory_limit:
                mem_bytes = parse_memory_limit_to_bytes(self.memory_limit)
                if mem_bytes and mem_bytes > 0:
                    prlimit_parts.append(f"--as={mem_bytes}")
            if self.cpu_limit and self.cpu_limit > 0:
                prlimit_parts.append(f"--cpu={int(self.cpu_limit)}")
            if len(prlimit_parts) > 1:
                final_command = f"{' '.join(prlimit_parts)} -- {final_command}"

        proc = None
        pgid = None
        stdout = ""
        stderr = ""
        exit_code = -1

        try:
            proc = subprocess.Popen(
                final_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=workdir if workdir and os.path.isdir(workdir) else None,
                env=exec_env,
                start_new_session=True,  # 创建新进程组
            )
            pgid = os.getpgid(proc.pid)

            # 通知调用方进程已启动（可用于立刻更新进程跟踪器）
            if on_started is not None:
                on_started(proc, pgid)

            stdout, stderr = proc.communicate(timeout=timeout)
            exit_code = proc.returncode

        except subprocess.TimeoutExpired:
            if pgid is not None:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            stderr = f"命令超时 ({timeout}s)"
            exit_code = -1

        except Exception as e:
            if pgid is not None and proc is not None and proc.poll() is None:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            stderr = str(e)
            exit_code = -1

        return proc, pgid, stdout, stderr, exit_code
