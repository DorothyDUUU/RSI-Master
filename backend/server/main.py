"""FastAPI application for GPU backend server — session-based API."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .gpu_monitor import GPUMonitor
from .models import (
    BackendStatus,
    GPUInfo,
    ScriptLog,
    ScriptSubmitRequest,
    ScriptSubmitResponse,
    SessionExecRequest,
    SessionExecResponse,
    ShellSession,
)
from .session_manager import SessionManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting GPU Backend Server")
    # wrapper/task 日志目录默认 /tmp/gpu_backend_logs (pod 本地, 重启即丢)。
    # 启动时设 GPU_BACKEND_LOG_DIR=<run_dir>/gpu_backend_logs (或 --log-dir) 可把
    # gpu_script_*.log 等调度/task 日志落到 run-dir (NFS 持久, 随 run 归档)。
    app.state.session_manager = SessionManager(
        log_dir=os.environ.get("GPU_BACKEND_LOG_DIR")
    )
    app.state.gpu_monitor = GPUMonitor()
    # All slow external calls (nvidia-smi, tmux) live on dedicated daemon
    # threads that publish caches — the scanner thread, the tmux session-set
    # sampler, and the VRAM sampler.  Lock-taking endpoints only ever read
    # memory, so a wedged driver/tmux ("断联") can never freeze the server.
    app.state.session_manager.start_scanner()
    app.state.session_manager.start_tmux_sampler()
    app.state.gpu_monitor.start_sampler()
    yield
    logger.info("Shutting down GPU Backend Server")


app = FastAPI(
    title="GPU Backend Server",
    description="Session-based backend for GPU task scheduling and shell management",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _sm() -> SessionManager:
    from fastapi import Request  # noqa: F811
    import inspect as _inspect
    # We need the app reference — use the module-level one.
    return app.state.session_manager


def _gpu() -> GPUMonitor:
    return app.state.gpu_monitor


# ── Root / health ───────────────────────────────────────────────────────────

@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "GPU Backend Server", "version": "2.0.0"}


@app.get("/health")
def health_check() -> dict[str, Any]:
    """Cheap O(1) health probe.

    Reads only in-memory caches (GPU snapshot + sampled session set) — no
    subprocess, no lock held across I/O — so a wedged nvidia-smi/tmux (D-state)
    can never block the probe that the platform uses to keep this pod alive.
    """
    sm: SessionManager = app.state.session_manager
    gpu: GPUMonitor = app.state.gpu_monitor
    resource_status = gpu.check_resources()
    sessions = sm.list_sessions()
    return {
        "status": "healthy" if resource_status["healthy"] else "degraded",
        "resources": resource_status,
        "active_sessions": len(sessions),
        "gpu_snapshot_age_s": gpu.snapshot_age(),
    }


# ── Session endpoints ───────────────────────────────────────────────────────

@app.post("/sessions/exec", response_model=SessionExecResponse)
def session_exec(request: SessionExecRequest) -> SessionExecResponse:
    """Execute a command in a shell session (update_shell_session).

    When *session_id* is ``None`` a new tmux session is created and its id +
    working directory are returned alongside the output.
    """
    return app.state.session_manager.exec_in_session(request)


@app.get("/sessions", response_model=list[ShellSession])
def session_list() -> list[ShellSession]:
    """List all active shell sessions (list_shell_session).

    Returns both foreground interactive sessions and background script sessions.
    """
    return app.state.session_manager.list_sessions()


@app.post("/sessions/{session_id}/kill")
def session_kill(session_id: str) -> dict[str, Any]:
    """Kill a shell session and all its child processes (kill_shell_session)."""
    success = app.state.session_manager.kill_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return {"success": True, "message": f"Session {session_id} killed"}


# ── Script endpoints ────────────────────────────────────────────────────────

@app.post("/scripts/submit", response_model=ScriptSubmitResponse)
def script_submit(request: ScriptSubmitRequest) -> ScriptSubmitResponse:
    """Submit a bash script for execution (submit_shell_scripts).

    Supports foreground (blocking), background (tmux + log file), and
    resume-into-existing-session modes.
    """
    return app.state.session_manager.submit_script(request)


@app.get("/scripts/{script_id}")
def script_status(script_id: str) -> dict[str, Any]:
    """Get the current status of a submitted script."""
    info = app.state.session_manager.get_script_status(script_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Script {script_id} not found")
    return info.model_dump()


@app.get("/scripts/{script_id}/log", response_model=ScriptLog)
def script_log(script_id: str) -> ScriptLog:
    """Get the log output of a script."""
    log = app.state.session_manager.get_script_log(script_id)
    if log is None:
        raise HTTPException(status_code=404, detail=f"Script {script_id} not found")
    return log


@app.post("/scripts/{script_id}/kill")
def script_kill(script_id: str) -> dict[str, Any]:
    """Kill a running script."""
    success = app.state.session_manager.kill_script(script_id)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Script {script_id} not found or not running",
        )
    return {"success": True, "message": f"Script {script_id} killed"}


@app.get("/queue")
def queue_snapshot() -> dict[str, Any]:
    return app.state.session_manager.get_queue_snapshot()


@app.post("/scripts/{script_id}/cancel-queued")
def cancel_queued(script_id: str) -> dict[str, Any]:
    return app.state.session_manager.cancel_queued_script(script_id)


# ── Resource endpoints ──────────────────────────────────────────────────────

@app.get("/resources", response_model=GPUInfo)
def get_resources() -> GPUInfo:
    """Get GPU / CPU / memory resource information (get_info)."""
    return app.state.gpu_monitor.get_gpu_info()


@app.get("/status", response_model=BackendStatus)
def get_status() -> BackendStatus:
    """Get aggregated backend status (sessions + resources + queue)."""
    from .models import QueueItem
    sm: SessionManager = app.state.session_manager
    gpu: GPUMonitor = app.state.gpu_monitor
    queue_items: list[QueueItem] = []
    for sid in sm._queue:
        info = sm._scripts.get(sid)
        if info:
            queue_items.append(QueueItem(
                script_id=sid,
                priority=info.priority.value,
                gpu_nums=info.gpu_nums,
            ))
    return BackendStatus(
        sessions=sm.list_sessions(),
        gpu_info=gpu.get_gpu_info(),
        queue=queue_items,
    )


# ── Entrypoint ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8989)
