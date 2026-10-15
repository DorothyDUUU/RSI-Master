"""EvoMaster Bash tool.

Provides the ability to execute Bash commands in the environment.
Supports optional Bash Backend Server for sandbox filtering and process management.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import httpx
from pydantic import Field

from ..base import BaseTool, BaseToolParams, ToolError

if TYPE_CHECKING:
    from evomaster.agent.session import BaseSession

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bash Backend Server 配置
# ---------------------------------------------------------------------------

# 后端服务地址，为空则禁用后端（直接走 session.exec_bash）
BASH_BACKEND_URL: str = os.environ.get("BASH_BACKEND_URL", "http://localhost:40059").rstrip("/")

# 后端健康检查超时（秒）
_HEALTH_CHECK_TIMEOUT: float = 2.0

# 后端执行超时（秒），仅用于 HTTP 请求本身的超时
_BACKEND_REQUEST_TIMEOUT: float = 600.0

# ---------------------------------------------------------------------------
# Fallback 安全拦截 (后端不可用时的兜底)
#
# 后端挂掉时 BashTool 会 fallback 到 session.exec_bash → local_exec → 裸
# subprocess.run(shell=True)。这段兜底在落入该路径前先跑共享拦截模块
# (tools/bash_security, 与后端过滤器同源), 命中 → 不执行、返回过滤结果。
# ---------------------------------------------------------------------------
try:
    from tools.bash_security import ESCAPE_GUIDANCE, check_command, check_paths
    _SECURITY_AVAILABLE = True
except ImportError:  # repo root 不在 sys.path —— 保持现状,不 double-take-down
    _SECURITY_AVAILABLE = False
    check_command = None  # type: ignore[assignment]
    check_paths = None  # type: ignore[assignment]
    ESCAPE_GUIDANCE = None  # type: ignore[assignment]
    logger.warning(
        "tools.bash_security 不可用 —— Bash 兜底路径将不执行安全拦截。"
        "请确认 repo root 在 PYTHONPATH 上。"
    )

try:
    # 兜底白名单与线上服务端共用同一份真相。bash_backend 包会 import fastapi,
    # agent 运行环境可能没有 —— 失败时退回内联副本。
    from tools.bash_backend.constants import DEFAULT_WHITELIST as _BACKEND_DEFAULT_WHITELIST
except Exception:
    _BACKEND_DEFAULT_WHITELIST = None

# 与 tools/bash_backend/constants.py 保持一致的内联副本
_FALLBACK_DEFAULT_WHITELIST = [
    "/root",
    "/tmp",
    "/var/tmp",
]


def _get_fallback_allowed_dirs(workspace: str) -> list[str]:
    """兜底白名单 = [workspace] + DEFAULT_WHITELIST + env BASH_BACKEND_WHITELIST。

    BASH_BACKEND_WHITELIST 用来补齐部署侧额外放行的路径, 与启动服务端时
    传给 --whitelist 的值保持一致, 避免后端挂掉时 agent 读这些路径被误拦。
    """
    dirs: list[str] = [workspace]
    if _BACKEND_DEFAULT_WHITELIST:
        dirs.extend(_BACKEND_DEFAULT_WHITELIST)
    else:
        dirs.extend(_FALLBACK_DEFAULT_WHITELIST)
    env_wl = os.environ.get("BASH_BACKEND_WHITELIST", "")
    if env_wl:
        dirs.extend(p for p in env_wl.split() if p)
    return [d for d in dirs if d]


def _blocked_by_security(command: str, workspace: str) -> str | None:
    """后端不可用时对命令做兜底拦截。命中返回 reason,放行返回 None。"""
    if not _SECURITY_AVAILABLE:
        return None
    r = check_command(command)
    if not r.allowed:
        return f"[{r.check_id}] {r.reason}"
    dirs = _get_fallback_allowed_dirs(workspace)
    r = check_paths(command, cwd=workspace or ".", allowed_dirs=dirs)
    if not r.allowed:
        return f"[{r.check_id}] {r.reason}"
    return None


def configure_backend(url: str) -> None:
    """配置 Bash Backend Server 地址。

    Args:
        url: 后端服务地址，如 "http://localhost:40058"。为空则禁用。
    """
    global BASH_BACKEND_URL
    BASH_BACKEND_URL = url.rstrip("/")


def _check_backend_health(base_url: str) -> bool:
    """检查后端服务是否健康。"""
    try:
        resp = httpx.get(
            f"{base_url}/health",
            timeout=_HEALTH_CHECK_TIMEOUT,
        )
        return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError):
        return False


def _execute_via_backend(
    base_url: str,
    command: str,
    workspace: str,
    timeout: float,
) -> dict[str, Any] | None:
    """通过后端服务执行命令。

    白名单由服务端权威配置 (DEFAULT_WHITELIST + 启动 --whitelist), 客户端
    不携带白名单 —— 避免客户端擅自缩小/扩大沙箱范围。

    Returns:
        后端响应 dict，失败时返回 None。
    """
    try:
        resp = httpx.post(
            f"{base_url}/execute",
            json={
                "command": command,
                "agent_workspace": workspace,
                "timeout": timeout,
            },
            timeout=timeout + 30,  # HTTP 超时略大于命令超时
        )
        resp.raise_for_status()
        return resp.json()
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError) as e:
        logger.warning(f"Bash backend request failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class BashToolParams(BaseToolParams):
    """Execute a bash command in the terminal within a persistent shell session.

    ### Command Execution
    * One command at a time: You can only execute one bash command at a time. If you need to run multiple commands sequentially, use `&&` or `;` to chain them together.
    * Persistent session: Commands execute in a persistent shell session where environment variables, virtual environments, and working directory persist between commands.
    * Soft timeout: Commands have a soft timeout of 10 seconds, once that's reached, you have the option to continue or interrupt the command.

    ### Long-running Commands
    * For commands that may run indefinitely, run them in the background and redirect output to a file, e.g. `python3 app.py > server.log 2>&1 &`.
    * For commands that may run for a long time, you should set the "timeout" parameter to an appropriate value.
    * If a bash command returns exit code `-1`, this means the process hit the soft timeout and is not yet finished. By setting `is_input` to `true`, you can:
      - Send empty `command` to retrieve additional logs
      - Send text (set `command` to the text) to STDIN of the running process
      - Send control commands like `C-c` (Ctrl+C), `C-z` (Ctrl+Z) to interrupt the process

    ### Best Practices
    * Directory verification: Before creating new directories or files, first verify the parent directory exists and is the correct location.
    * Directory management: Try to maintain working directory by using absolute paths and avoiding excessive use of `cd`.

    ### Output Handling
    * Output truncation: If the output exceeds a maximum length, it will be truncated.
    """

    name: ClassVar[str] = "execute_bash"

    command: str = Field(
        description="The bash command to execute. Can be empty string to view additional logs when previous exit code is `-1`. Can be `C-c` (Ctrl+C) to interrupt the currently running process."
    )
    is_input: Literal["true", "false"] = Field(
        default="false",
        description="If True, the command is an input to the running process. If False, the command is a bash command to be executed in the terminal. Default is False.",
    )
    timeout: float = Field(
        default=-1,
        description="Optional. Sets a hard timeout in seconds for the command execution. If not provided, the command will use the default soft timeout behavior.",
    )


class BashTool(BaseTool):
    """Bash command execution tool."""

    name: ClassVar[str] = "execute_bash"
    params_class: ClassVar[type[BaseToolParams]] = BashToolParams

    def execute(self, session: BaseSession, args_json: str) -> tuple[str, dict[str, Any]]:
        """Execute a Bash command.

        优先使用 Bash Backend Server（如果配置了且健康），否则 fallback 到 session.exec_bash。
        """
        try:
            params = self.parse_params(args_json)
        except Exception as e:
            return f"Parameter validation error: {str(e)}", {"error": str(e)}

        assert isinstance(params, BashToolParams)

        timeout = int(params.timeout) if params.timeout > 0 else None
        is_input = params.is_input == "true"
        command = params.command

        # 获取 workspace
        workspace = session.get_workspace_path() or session.config.workspace_path

        # ------------------------------------------------------------------
        # 尝试通过 Bash Backend Server 执行
        # ------------------------------------------------------------------
        if BASH_BACKEND_URL and not is_input:
            backend_timeout = timeout or session.config.timeout or 300

            if _check_backend_health(BASH_BACKEND_URL):
                backend_result = _execute_via_backend(
                    base_url=BASH_BACKEND_URL,
                    command=command,
                    workspace=workspace,
                    timeout=backend_timeout,
                )

                if backend_result is not None:
                    # 后端返回成功，构建与 session.exec_bash 兼容的输出
                    output = backend_result.get("stdout", "")
                    stderr = backend_result.get("stderr", "")
                    exit_code = backend_result.get("exit_code", -1)
                    working_dir = backend_result.get("working_dir", "")
                    filtered = backend_result.get("filtered", False)

                    # 合并输出（与 local_exec 行为一致）
                    if stderr:
                        output = (output + stderr) if output else stderr

                    # 如果被过滤，追加过滤原因
                    if filtered:
                        filter_reason = backend_result.get("filter_reason", "")
                        if filter_reason:
                            output += f"\n[Filtered: {filter_reason}]"

                    result = {
                        "stdout": backend_result.get("stdout", ""),
                        "stderr": stderr,
                        "exit_code": exit_code,
                        "working_dir": working_dir,
                        "output": output,
                    }
                    return self._format_output(result)

        # ------------------------------------------------------------------
        # Fallback: 通过 session.exec_bash 执行
        # ------------------------------------------------------------------
        if not is_input:
            blocked = _blocked_by_security(command, workspace)
            if blocked:
                guidance = f"\n{ESCAPE_GUIDANCE}" if ESCAPE_GUIDANCE else ""
                reason = f"[BLOCKED by security] {blocked}{guidance}"
                logger.warning(f"Bash fallback security block: {reason}")
                result = {
                    "stdout": "",
                    "stderr": reason,
                    "exit_code": -1,
                    "working_dir": workspace or "",
                    "output": reason,
                }
                return self._format_output(result)

        result = session.exec_bash(
            command=command,
            timeout=timeout,
            is_input=is_input,
        )
        return self._format_output(result)

    @staticmethod
    def _format_output(result: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """格式化执行结果为 (observation, info) 元组。"""
        output = result.get("output", "") or result.get("stdout", "")
        exit_code = result.get("exit_code", -1)
        working_dir = result.get("working_dir", "")

        working_dir_abs = str(Path(working_dir).absolute()) if working_dir else ""

        obs = output
        if working_dir_abs:
            obs += f"\n[Current working directory: {working_dir_abs}]"
        if exit_code != -1:
            obs += f"\n[Command finished with exit code {exit_code}]"

        info = {
            "exit_code": exit_code,
            "working_dir": working_dir_abs,
        }

        return obs, info
