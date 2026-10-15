"""API 请求/响应的 Pydantic 模型。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .constants import DEFAULT_TIMEOUT


class ExecuteRequest(BaseModel):
    """执行请求。"""
    command: str = Field(description="要执行的 bash 命令")
    agent_workspace: str = Field(
        default="",
        description="agent 的工作目录（绝对路径）",
    )
    timeout: float = Field(
        default=DEFAULT_TIMEOUT,
        description="超时时间（秒）",
        gt=0,
    )
    meta_data: dict[str, Any] | None = Field(
        default=None,
        description="元数据（可包含 whitelist 等自定义信息）",
    )


class ExecuteResponse(BaseModel):
    """执行响应。"""
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    working_dir: str = ""
    filtered: bool = False
    filter_reason: str | None = None
    process_id: str | None = None
    duration_seconds: float = 0.0


class ProcessInfoResponse(BaseModel):
    """进程信息响应。"""
    process_id: str
    command: str
    agent_workspace: str
    status: str
    start_time: str
    end_time: str | None = None
    exit_code: int | None = None
    duration_seconds: float | None = None


class KillResponse(BaseModel):
    """Kill 操作响应。"""
    success: bool
    message: str


class KillAllResponse(BaseModel):
    """Kill All 响应。"""
    killed_count: int
    messages: list[str]


class FilterInfoResponse(BaseModel):
    """过滤器信息响应。"""
    name: str
    description: str
    type: str  # "pre_filter" or "post_check"
    enabled: bool


class SystemStatusResponse(BaseModel):
    """系统状态响应。"""
    cpu_usage_percent: float
    memory: dict[str, Any]
    active_processes: int
    total_tracked_processes: int


class ToggleFilterRequest(BaseModel):
    """切换过滤器状态请求。"""
    enabled: bool
