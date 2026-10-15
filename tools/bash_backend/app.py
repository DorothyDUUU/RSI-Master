"""FastAPI 应用和端点定义。"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

from .constants import DEFAULT_MAX_SLEEP_SECONDS, DEFAULT_WHITELIST, MAX_CONCURRENT_PROCESSES, MAX_OUTPUT_LENGTH
from .executor import CommandExecutor
from .logger import ExecuteLogger
from .filter_base import (
    FilterAction,
    FilterContext,
    _FILTER_REGISTRY,
    get_all_filters,
    get_post_checks,
    get_pre_filters,
)
from .filters import *  # noqa: F401,F403 — 触发过滤器注册
from .models import (
    ExecuteRequest,
    ExecuteResponse,
    FilterInfoResponse,
    KillAllResponse,
    KillResponse,
    ProcessInfoResponse,
    SystemStatusResponse,
    ToggleFilterRequest,
)
from .monitor import SystemMonitor
from .tracker import ProcessInfo, ProcessStatus, ProcessTracker

logger = logging.getLogger("bash_backend")


def create_app(
    default_workspace: str = "",
    whitelist: list[str] | None = None,
    memory_limit: str | None = None,
    cpu_limit: float = 0.0,
    cpu_devices: str | None = None,
    max_output_length: int = MAX_OUTPUT_LENGTH,
    max_concurrent_processes: int = MAX_CONCURRENT_PROCESSES,
    max_sleep_seconds: int = DEFAULT_MAX_SLEEP_SECONDS,
    data_dir: str = "",
) -> FastAPI:
    """创建 FastAPI 应用。

    Args:
        default_workspace: 默认工作目录
        whitelist: 白名单路径列表
        memory_limit: 内存限制（如 '4g'）
        cpu_limit: CPU 时间限制（秒）
        cpu_devices: CPU 核心绑定
        max_output_length: 最大输出长度（字符数）
        max_concurrent_processes: 最大并发进程数
        max_sleep_seconds: long_sleep 过滤器上限, 拦截 sleep 超过该秒数的命令
        data_dir: JSONL 日志存储目录，默认 tools/bash_backend/data/

    Returns:
        FastAPI 应用实例
    """
    app = FastAPI(
        title="Bash Backend Server",
        description="安全的 bash 命令执行服务，支持过滤、监控和进程管理",
        version="1.0.0",
    )

    # 配置
    _whitelist = list(DEFAULT_WHITELIST)
    if whitelist:
        _whitelist.extend(whitelist)
    _max_sleep_seconds = max_sleep_seconds

    executor = CommandExecutor(
        memory_limit=memory_limit,
        cpu_limit=cpu_limit,
        cpu_devices=cpu_devices,
    )

    # 全局进程跟踪器
    tracker = ProcessTracker()

    # JSONL 持久化日志
    exec_logger = ExecuteLogger(data_dir=data_dir)

    # 设置输出长度检查的最大值
    output_check = _FILTER_REGISTRY.get("output_length")
    if output_check and hasattr(output_check, "max_length"):
        output_check.max_length = max_output_length

    # -----------------------------------------------------------------------
    # Endpoints
    # -----------------------------------------------------------------------

    @app.get("/health")
    def health():
        """健康检查。"""
        return {"status": "ok", "timestamp": datetime.now().isoformat()}

    @app.post("/execute", response_model=ExecuteResponse)
    def execute_command(req: ExecuteRequest):
        """执行 bash 命令。

        流程: pre-filter → 执行 → post-check → 返回
        """
        # 定期清理已完成的旧进程记录（保留最近 10 分钟）
        tracker.cleanup_old(max_age_seconds=600)

        # 检查并发进程数限制
        active_count = len(tracker.get_active())
        if active_count >= max_concurrent_processes:
            raise HTTPException(
                status_code=429,
                detail=f"活跃进程数已达上限 ({active_count}/{max_concurrent_processes})，请稍后重试",
            )

        process_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        # 记录请求日志
        exec_logger.log_request(
            process_id=process_id,
            command=req.command,
            agent_workspace=req.agent_workspace,
            timeout=req.timeout,
            meta_data=req.meta_data,
        )

        # 解析 workspace
        workspace = req.agent_workspace or default_workspace
        resolved_workspace = str(Path(workspace).resolve()) if workspace else ""

        # 构建过滤上下文
        ctx = FilterContext(
            command=req.command,
            agent_workspace=req.agent_workspace,
            timeout=req.timeout,
            meta_data=req.meta_data or {},
            resolved_workspace=resolved_workspace,
        )
        # 白名单以服务端为准 (DEFAULT_WHITELIST + --whitelist), 覆盖任何客户端
        # 传入的白名单 —— 客户端可发送自己范围的 meta_data, 但不能缩小/扩大沙箱。
        ctx.meta_data["whitelist"] = _whitelist
        # long_sleep 过滤器上限同样以服务端为准, 客户端不可覆盖
        ctx.meta_data["max_sleep_seconds"] = _max_sleep_seconds

        # ---- Pre-filters ----
        for f in get_pre_filters():
            result = f.check(ctx)
            if result.action == FilterAction.REJECT:
                duration = time.time() - start_time
                tracker.register(ProcessInfo(
                    process_id=process_id,
                    pgid=None,
                    pid=None,
                    command=req.command,
                    agent_workspace=workspace,
                    start_time=start_time,
                    end_time=time.time(),
                    status=ProcessStatus.FILTERED,
                    filter_reason=result.reason,
                ))
                exec_logger.log_response(
                    process_id=process_id,
                    stdout="",
                    stderr=f"[FILTERED by {result.filter_name}] {result.reason}",
                    exit_code=-1,
                    working_dir=resolved_workspace,
                    filtered=True,
                    filter_reason=f"[{result.filter_name}] {result.reason}",
                    duration_seconds=duration,
                    command=req.command,
                )
                return ExecuteResponse(
                    stdout="",
                    stderr=f"[FILTERED by {result.filter_name}] {result.reason}",
                    exit_code=-1,
                    working_dir=resolved_workspace,
                    filtered=True,
                    filter_reason=f"[{result.filter_name}] {result.reason}",
                    process_id=process_id,
                    duration_seconds=duration,
                )
            elif result.action == FilterAction.WARN:
                logger.warning(f"Filter warning [{result.filter_name}]: {result.reason}")

        # ---- 执行命令 ----
        proc_info = ProcessInfo(
            process_id=process_id,
            pgid=None,
            pid=None,
            command=req.command,
            agent_workspace=workspace,
            start_time=start_time,
        )
        tracker.register(proc_info)

        proc, pgid, stdout, stderr, exit_code = executor.execute(
            command=req.command,
            timeout=req.timeout,
            workdir=resolved_workspace,
            on_started=lambda p, g: tracker.update(
                process_id, pgid=g, pid=p.pid,
            ),
        )

        # 更新进程信息
        end_time = time.time()
        tracker.update(
            process_id,
            pgid=pgid,
            pid=proc.pid if proc else None,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            end_time=end_time,
            status=ProcessStatus.COMPLETED if exit_code != -1 else ProcessStatus.TIMEOUT,
        )

        # ---- Post-checks ----
        combined_output = stdout + stderr
        for f in get_post_checks():
            result = f.check(ctx, output=combined_output)
            if result.action == FilterAction.REJECT:
                tracker.update(
                    process_id,
                    status=ProcessStatus.FILTERED,
                    filter_reason=result.reason,
                )
                post_duration = end_time - start_time
                exec_logger.log_response(
                    process_id=process_id,
                    stdout=stdout[:1000],
                    stderr=f"[POST-CHECK by {result.filter_name}] {result.reason}",
                    exit_code=exit_code,
                    working_dir=resolved_workspace,
                    filtered=True,
                    filter_reason=f"[{result.filter_name}] {result.reason}",
                    duration_seconds=post_duration,
                    command=req.command,
                )
                return ExecuteResponse(
                    stdout=stdout[:1000] + "\n... [TRUNCATED BY FILTER] ...",
                    stderr=f"[POST-CHECK by {result.filter_name}] {result.reason}",
                    exit_code=exit_code,
                    working_dir=resolved_workspace,
                    filtered=True,
                    filter_reason=f"[{result.filter_name}] {result.reason}",
                    process_id=process_id,
                    duration_seconds=post_duration,
                )

        duration = end_time - start_time
        exec_logger.log_response(
            process_id=process_id,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            working_dir=resolved_workspace,
            filtered=False,
            filter_reason=None,
            duration_seconds=round(duration, 3),
            command=req.command,
        )
        return ExecuteResponse(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            working_dir=resolved_workspace,
            filtered=False,
            process_id=process_id,
            duration_seconds=round(duration, 3),
        )

    @app.get("/processes", response_model=list[ProcessInfoResponse])
    def list_processes(status: str | None = None):
        """列出进程。

        Args:
            status: 可选的状态过滤 (running, completed, timeout, killed, filtered)
        """
        status_filter = None
        if status:
            try:
                status_filter = ProcessStatus(status)
            except ValueError:
                raise HTTPException(400, f"无效的状态值: {status}")

        processes = tracker.get_all(status_filter)
        now = time.time()
        return [
            ProcessInfoResponse(
                process_id=p.process_id,
                command=p.command[:200],  # 截断过长命令
                agent_workspace=p.agent_workspace,
                status=p.status.value,
                start_time=datetime.fromtimestamp(p.start_time).isoformat(),
                end_time=datetime.fromtimestamp(p.end_time).isoformat() if p.end_time else None,
                exit_code=p.exit_code,
                duration_seconds=round(
                    (p.end_time or now) - p.start_time, 2
                ),
            )
            for p in processes
        ]

    @app.post("/kill/{process_id}", response_model=KillResponse)
    def kill_process(process_id: str):
        """Kill 指定进程。"""
        ok, msg = tracker.kill_process(process_id)
        return KillResponse(success=ok, message=msg)

    @app.post("/kill_all", response_model=KillAllResponse)
    def kill_all_processes():
        """Kill 所有活跃进程。"""
        killed, messages = tracker.kill_all()
        return KillAllResponse(killed_count=killed, messages=messages)

    @app.get("/filters", response_model=list[FilterInfoResponse])
    def list_filters():
        """列出所有过滤器。"""
        return [
            FilterInfoResponse(
                name=f.name,
                description=f.description,
                type="pre_filter" if f.is_pre_filter else "post_check",
                enabled=f.enabled,
            )
            for f in _FILTER_REGISTRY.values()
        ]

    @app.post("/filters/{filter_name}/toggle", response_model=FilterInfoResponse)
    def toggle_filter(filter_name: str, req: ToggleFilterRequest):
        """启用/禁用指定过滤器。"""
        f = _FILTER_REGISTRY.get(filter_name)
        if f is None:
            raise HTTPException(404, f"过滤器 '{filter_name}' 不存在")
        f.enabled = req.enabled
        return FilterInfoResponse(
            name=f.name,
            description=f.description,
            type="pre_filter" if f.is_pre_filter else "post_check",
            enabled=f.enabled,
        )

    @app.get("/status", response_model=SystemStatusResponse)
    def system_status():
        """系统资源状态。"""
        return SystemStatusResponse(
            cpu_usage_percent=SystemMonitor.get_cpu_usage(),
            memory=SystemMonitor.get_memory_usage(),
            active_processes=len(tracker.get_active()),
            total_tracked_processes=len(tracker.get_all()),
        )

    return app
