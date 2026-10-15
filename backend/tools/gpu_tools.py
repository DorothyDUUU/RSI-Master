"""GPU tools for agent integration with backend server."""

from __future__ import annotations

import json
import logging
from typing import Any, ClassVar, Optional

from pydantic import BaseModel, Field

# Import base tool classes. Repo root is derived from this file's location
# (backend/tools/ -> <repo>) so no absolute checkout path is needed.
import sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evomaster.agent.tools.base import BaseTool, BaseToolParams, ToolError
from evomaster.agent.session import BaseSession

from ..client.client import BackendClient
from ..server.models import TaskPriority, TaskStatus


class GpuSubmitTaskParams(BaseToolParams):
    """Submit a task to the GPU backend for execution.

    This tool submits a bash script to the GPU backend server for execution.
    The backend will allocate GPU resources and run the script.
    """

    name: ClassVar[str] = "gpu_submit_task"

    bash_scripts_path: str = Field(
        ...,
        description="Path to the bash script file to execute"
    )
    run_backend: bool = Field(
        default=False,
        description="Whether to persist the task in tmux for long-running execution"
    )
    timeout: Optional[int] = Field(
        default=None,
        description="Timeout in seconds (ignored if run_backend=True)"
    )
    gpu_nums: int = Field(
        default=1,
        ge=1,
        le=8,
        description="Number of GPU devices required"
    )
    prior: str = Field(
        default="normal",
        description="Task priority: low, normal, high, or urgent"
    )
    require_instant_feedback: bool = Field(
        default=True,
        description="Whether to block and wait for task completion"
    )
    task_name: Optional[str] = Field(
        default=None,
        description="Optional name for the task"
    )
    env_vars: Optional[dict[str, str]] = Field(
        default=None,
        description="Optional environment variables as key-value pairs"
    )


class GpuSubmitTaskTool(BaseTool):
    """Tool for submitting GPU tasks."""

    name = "gpu_submit_task"
    params_class = GpuSubmitTaskParams

    def __init__(self, backend_url: str = "http://localhost:8000"):
        super().__init__()
        self.backend_url = backend_url

    def execute(
        self, session: BaseSession, args_json: str
    ) -> tuple[str, dict[str, Any]]:
        """Execute the tool.

        Args:
            session: Environment session.
            args_json: Parameter JSON string.

        Returns:
            (observation, info) tuple.
        """
        params: GpuSubmitTaskParams = self.parse_params(args_json)

        try:
            with BackendClient(self.backend_url) as client:
                # Map priority string to enum
                priority_map = {
                    "low": TaskPriority.LOW,
                    "normal": TaskPriority.NORMAL,
                    "high": TaskPriority.HIGH,
                    "urgent": TaskPriority.URGENT,
                }
                priority = priority_map.get(params.prior.lower(), TaskPriority.NORMAL)

                response = client.submit_task(
                    bash_scripts_path=params.bash_scripts_path,
                    run_backend=params.run_backend,
                    timeout=params.timeout,
                    gpu_nums=params.gpu_nums,
                    prior=priority,
                    require_instant_feedback=params.require_instant_feedback,
                    task_name=params.task_name,
                    env_vars=params.env_vars,
                )

                observation = f"Task submitted successfully. Task ID: {response.task_id}, Status: {response.status}"
                if response.gpu_devices:
                    observation += f", GPU Devices: {response.gpu_devices}"
                if response.tmux_session:
                    observation += f", Tmux Session: {response.tmux_session}"

                info = {
                    "task_id": response.task_id,
                    "status": response.status.value,
                    "gpu_devices": response.gpu_devices,
                    "tmux_session": response.tmux_session,
                    "pid": response.pid,
                }

                return observation, info

        except Exception as e:
            raise ToolError(f"Failed to submit task: {e}")


class GpuInspectRunningParams(BaseToolParams):
    """Inspect current GPU resources and running services.

    This tool shows GPU utilization, running tasks, tmux sessions,
    and queued tasks.
    """

    name: ClassVar[str] = "gpu_inspect_running"


class GpuInspectRunningTool(BaseTool):
    """Tool for inspecting GPU resources and running services."""

    name = "gpu_inspect_running"
    params_class = GpuInspectRunningParams

    def __init__(self, backend_url: str = "http://localhost:8000"):
        super().__init__()
        self.backend_url = backend_url

    def execute(
        self, session: BaseSession, args_json: str
    ) -> tuple[str, dict[str, Any]]:
        """Execute the tool.

        Args:
            session: Environment session.
            args_json: Parameter JSON string.

        Returns:
            (observation, info) tuple.
        """
        try:
            with BackendClient(self.backend_url) as client:
                services = client.get_running_services()

                # Format observation
                lines = ["=== GPU Status ==="]
                for device in services.gpu_info.devices:
                    lines.append(
                        f"GPU {device.device_id} ({device.name}): "
                        f"Memory {device.memory_utilization:.1f}%, "
                        f"Utilization {device.gpu_utilization:.1f}%"
                    )

                lines.append(f"\nAvailable devices: {services.gpu_info.available_devices}")
                lines.append(f"CPU: {services.gpu_info.cpu_percent:.1f}%")
                lines.append(
                    f"Memory: {services.gpu_info.memory_percent:.1f}% "
                    f"({services.gpu_info.memory_used_gb:.1f}/{services.gpu_info.memory_total_gb:.1f} GB)"
                )

                if services.gpu_tasks:
                    lines.append("\n=== Running Tasks ===")
                    for task in services.gpu_tasks:
                        lines.append(
                            f"- {task.task_id}: {task.task_name} "
                            f"(GPU {task.gpu_devices}, PID {task.pid})"
                        )

                if services.tmux_sessions:
                    lines.append("\n=== Tmux Sessions ===")
                    for session in services.tmux_sessions:
                        lines.append(f"- {session['name']}")

                if services.queued_tasks:
                    lines.append("\n=== Queued Tasks ===")
                    for task in services.queued_tasks:
                        lines.append(
                            f"- {task.task_id}: {task.task_name} "
                            f"(Priority: {task.priority.value})"
                        )

                observation = "\n".join(lines)

                info = {
                    "gpu_info": {
                        "total_devices": services.gpu_info.total_devices,
                        "available_devices": services.gpu_info.available_devices,
                        "cpu_percent": services.gpu_info.cpu_percent,
                        "memory_percent": services.gpu_info.memory_percent,
                    },
                    "running_tasks": len(services.gpu_tasks),
                    "queued_tasks": len(services.queued_tasks),
                    "tmux_sessions": len(services.tmux_sessions),
                }

                return observation, info

        except Exception as e:
            raise ToolError(f"Failed to inspect running services: {e}")


class GpuInspectTaskParams(BaseToolParams):
    """Get the status and logs of a specific task.

    This tool retrieves detailed information about a task,
    including its current status and output logs.
    """

    name: ClassVar[str] = "gpu_inspect_task"

    task_id: str = Field(
        ...,
        description="Task ID to inspect"
    )


class GpuInspectTaskTool(BaseTool):
    """Tool for inspecting a specific task."""

    name = "gpu_inspect_task"
    params_class = GpuInspectTaskParams

    def __init__(self, backend_url: str = "http://localhost:8000"):
        super().__init__()
        self.backend_url = backend_url

    def execute(
        self, session: BaseSession, args_json: str
    ) -> tuple[str, dict[str, Any]]:
        """Execute the tool.

        Args:
            session: Environment session.
            args_json: Parameter JSON string.

        Returns:
            (observation, info) tuple.
        """
        params: GpuInspectTaskParams = self.parse_params(args_json)

        try:
            with BackendClient(self.backend_url) as client:
                task = client.get_task_status(params.task_id)
                if not task:
                    raise ToolError(f"Task {params.task_id} not found")

                log = client.get_task_log(params.task_id)

                # Format observation
                lines = [f"=== Task {task.task_id} ==="]
                lines.append(f"Name: {task.task_name}")
                lines.append(f"Status: {task.status.value}")
                lines.append(f"Script: {task.bash_scripts_path}")
                lines.append(f"GPU Devices: {task.gpu_devices}")
                lines.append(f"Priority: {task.priority.value}")

                if task.pid:
                    lines.append(f"PID: {task.pid}")
                if task.tmux_session:
                    lines.append(f"Tmux Session: {task.tmux_session}")

                lines.append(f"Created: {task.created_at}")
                if task.started_at:
                    lines.append(f"Started: {task.started_at}")
                if task.completed_at:
                    lines.append(f"Completed: {task.completed_at}")
                if task.exit_code is not None:
                    lines.append(f"Exit Code: {task.exit_code}")

                if log:
                    if log.stdout:
                        lines.append(f"\n=== STDOUT ===\n{log.stdout[-2000:]}")
                    if log.stderr:
                        lines.append(f"\n=== STDERR ===\n{log.stderr[-2000:]}")

                observation = "\n".join(lines)

                info = {
                    "task_id": task.task_id,
                    "status": task.status.value,
                    "exit_code": task.exit_code,
                    "is_running": task.status == TaskStatus.RUNNING,
                }

                return observation, info

        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"Failed to inspect task: {e}")


class GpuKillTaskParams(BaseToolParams):
    """Kill a running GPU task.

    This tool terminates a running task and frees its GPU resources.
    """

    name: ClassVar[str] = "gpu_kill_task"

    task_id: str = Field(
        ...,
        description="Task ID to kill"
    )


class GpuKillTaskTool(BaseTool):
    """Tool for killing a running task."""

    name = "gpu_kill_task"
    params_class = GpuKillTaskParams

    def __init__(self, backend_url: str = "http://localhost:8000"):
        super().__init__()
        self.backend_url = backend_url

    def execute(
        self, session: BaseSession, args_json: str
    ) -> tuple[str, dict[str, Any]]:
        """Execute the tool.

        Args:
            session: Environment session.
            args_json: Parameter JSON string.

        Returns:
            (observation, info) tuple.
        """
        params: GpuKillTaskParams = self.parse_params(args_json)

        try:
            with BackendClient(self.backend_url) as client:
                success = client.kill_task(params.task_id)

                if success:
                    observation = f"Task {params.task_id} killed successfully"
                else:
                    observation = f"Failed to kill task {params.task_id} (not found or not running)"

                info = {
                    "task_id": params.task_id,
                    "success": success,
                }

                return observation, info

        except Exception as e:
            raise ToolError(f"Failed to kill task: {e}")


def create_gpu_tools(backend_url: str = "http://localhost:8000") -> list[BaseTool]:
    """Create all GPU tools.

    Args:
        backend_url: URL of the backend server.

    Returns:
        List of GPU tool instances.
    """
    return [
        GpuSubmitTaskTool(backend_url),
        GpuInspectRunningTool(backend_url),
        GpuInspectTaskTool(backend_url),
        GpuKillTaskTool(backend_url),
    ]
