"""Synchronous HTTP client for the GPU backend server."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from ..server.models import (
    GPUInfo,
    ScriptLog,
    ScriptStatus,
    ScriptSubmitRequest,
    ScriptSubmitResponse,
    SessionExecRequest,
    SessionExecResponse,
    ShellSession,
)

logger = logging.getLogger(__name__)


class BackendError(Exception):
    """Raised when the backend returns an error response."""


class BackendClient:
    """HTTP client for the GPU backend server.

    Talks to the FastAPI server running on a GPU machine (possibly via SSH
    tunnel).  All methods are synchronous.
    """

    def __init__(self, base_url: str = "http://localhost:8989", timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    # -- context manager -------------------------------------------------------

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> BackendClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # -- low-level request -----------------------------------------------------

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            resp = self._client.request(method, url, **kwargs)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            logger.error("HTTP %s %s: %s", exc.response.status_code, url, exc.response.text)
            raise BackendError(exc.response.text) from exc
        except httpx.RequestError as exc:
            logger.error("Request error %s %s: %s", method, url, exc)
            raise BackendError(str(exc)) from exc

    # -- health ----------------------------------------------------------------

    def health_check(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    # -- shell sessions --------------------------------------------------------

    def exec_command(
        self,
        command: str,
        timeout: int = 300,
        session_id: str | None = None,
    ) -> SessionExecResponse:
        """Execute a command in a shell session.

        If *session_id* is ``None`` a new tmux session is created.
        """
        req = SessionExecRequest(command=command, timeout=timeout, session_id=session_id)
        data = self._request("POST", "/sessions/exec", json=req.model_dump())
        return SessionExecResponse(**data)

    def list_sessions(self) -> list[ShellSession]:
        data = self._request("GET", "/sessions")
        return [ShellSession(**s) for s in data]

    def kill_session(self, session_id: str) -> bool:
        try:
            data = self._request("POST", f"/sessions/{session_id}/kill")
            return data.get("success", False)
        except BackendError:
            return False

    # -- scripts ---------------------------------------------------------------

    def submit_script(
        self,
        command_path: str,
        timeout: int = 300,
        resume_session_id: str | None = None,
        run_in_background: bool = False,
        gpu_nums: int = 1,
        priority: str = "normal",
        max_mem_quota: int | None = None,
        need_exclusive_gpu: bool = True,
        queue_owner: str | None = None,
    ) -> ScriptSubmitResponse:
        """Submit a bash script for execution with GPU scheduling.

        *need_exclusive_gpu*: True by default (2026-08-29) — the task holds its
        allocated card(s) for the whole lifetime; pass False to allow sharing after
        the short exclusive lease window.
        """
        from ..server.models import ScriptPriority
        prio_map = {
            "low": ScriptPriority.LOW,
            "normal": ScriptPriority.NORMAL,
            "high": ScriptPriority.HIGH,
        }
        req = ScriptSubmitRequest(
            command_path=command_path,
            timeout=timeout,
            resume_session_id=resume_session_id,
            run_in_background=run_in_background,
            gpu_nums=gpu_nums,
            priority=prio_map.get(priority.lower(), ScriptPriority.NORMAL),
            max_mem_quota=max_mem_quota,
            need_exclusive_gpu=need_exclusive_gpu,
            queue_owner=queue_owner,
        )
        data = self._request("POST", "/scripts/submit", json=req.model_dump())
        return ScriptSubmitResponse(**data)

    def get_script_status(self, script_id: str) -> dict[str, Any] | None:
        try:
            return self._request("GET", f"/scripts/{script_id}")
        except BackendError:
            return None

    def get_script_log(self, script_id: str) -> ScriptLog | None:
        try:
            data = self._request("GET", f"/scripts/{script_id}/log")
            return ScriptLog(**data)
        except BackendError:
            return None

    def kill_script(self, script_id: str) -> bool:
        try:
            data = self._request("POST", f"/scripts/{script_id}/kill")
            return data.get("success", False)
        except BackendError:
            return False

    def get_queue(self) -> dict[str, Any]:
        return self._request("GET", "/queue")

    def cancel_queued_script(self, script_id: str) -> dict[str, Any]:
        # Never fall back to kill-script, even if an old server lacks the API.
        return self._request("POST", f"/scripts/{script_id}/cancel-queued")

    # -- resources -------------------------------------------------------------

    def get_resources(self) -> GPUInfo:
        data = self._request("GET", "/resources")
        return GPUInfo(**data)

    def get_status(self) -> dict[str, Any]:
        """Get combined sessions + resources snapshot."""
        return self._request("GET", "/status")
