"""Task scheduler for GPU backend."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .gpu_monitor import GPUMonitor
from .models import (
    TaskInfo,
    TaskLog,
    TaskPriority,
    TaskRequest,
    TaskResponse,
    TaskStatus,
)

logger = logging.getLogger(__name__)


class TaskScheduler:
    """Manages task scheduling and execution on GPU devices."""

    def __init__(self, log_dir: str = "/tmp/gpu_backend_logs"):
        self.gpu_monitor = GPUMonitor()
        self._tasks: dict[str, TaskInfo] = {}
        self._processes: dict[str, subprocess.Popen] = {}
        self._queue: list[str] = []  # task IDs in priority order
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

        # Task output buffers
        self._stdout_buffers: dict[str, list[str]] = defaultdict(list)
        self._stderr_buffers: dict[str, list[str]] = defaultdict(list)

    def _generate_task_id(self) -> str:
        """Generate a unique task ID."""
        return f"task_{uuid.uuid4().hex[:12]}"

    def _get_priority_weight(self, priority: TaskPriority) -> int:
        """Get numeric weight for priority (higher = more priority)."""
        weights = {
            TaskPriority.LOW: 0,
            TaskPriority.NORMAL: 1,
            TaskPriority.HIGH: 2,
            TaskPriority.URGENT: 3,
        }
        return weights.get(priority, 1)

    def _insert_into_queue(self, task_id: str, priority: TaskPriority) -> None:
        """Insert task into priority queue."""
        weight = self._get_priority_weight(priority)
        for i, queued_id in enumerate(self._queue):
            queued_task = self._tasks.get(queued_id)
            if queued_task and self._get_priority_weight(queued_task.priority) < weight:
                self._queue.insert(i, task_id)
                return
        self._queue.append(task_id)

    async def submit_task(self, request: TaskRequest) -> TaskResponse:
        """Submit a new task for execution.

        Args:
            request: Task submission request.

        Returns:
            TaskResponse with task ID and initial status.
        """
        async with self._lock:
            # Validate script path
            script_path = Path(request.bash_scripts_path)
            if not script_path.exists():
                return TaskResponse(
                    task_id="",
                    status=TaskStatus.FAILED,
                    message=f"Script not found: {request.bash_scripts_path}",
                )

            # Generate task ID
            task_id = self._generate_task_id()

            # Create task info
            task = TaskInfo(
                task_id=task_id,
                task_name=request.task_name or script_path.stem,
                status=TaskStatus.PENDING,
                bash_scripts_path=request.bash_scripts_path,
                gpu_devices=[],
                pid=None,
                tmux_session=None,
                created_at=datetime.now(),
                started_at=None,
                completed_at=None,
                exit_code=None,
                priority=request.prior,
                run_backend=request.run_backend,
                timeout=request.timeout,
            )

            self._tasks[task_id] = task

            # Try to run immediately or queue
            if request.require_instant_feedback:
                # Try to run immediately
                result = await self._try_run_task(task_id, request)
                return result
            else:
                # Add to queue
                self._insert_into_queue(task_id, request.prior)
                task.status = TaskStatus.PENDING
                return TaskResponse(
                    task_id=task_id,
                    status=TaskStatus.PENDING,
                    message="Task queued for execution",
                )

    async def _try_run_task(
        self, task_id: str, request: TaskRequest
    ) -> TaskResponse:
        """Try to run a task immediately.

        Args:
            task_id: Task ID to run.
            request: Original task request.

        Returns:
            TaskResponse with execution result.
        """
        task = self._tasks.get(task_id)
        if not task:
            return TaskResponse(
                task_id=task_id,
                status=TaskStatus.FAILED,
                message="Task not found",
            )

        # Get available GPU devices
        available_devices = self.gpu_monitor.get_available_devices(request.gpu_nums)
        if not available_devices:
            self._insert_into_queue(task_id, request.prior)
            return TaskResponse(
                task_id=task_id,
                status=TaskStatus.PENDING,
                message="No GPU devices available, task queued",
            )

        # Run the task
        return await self._run_task(task_id, request, available_devices)

    async def _run_task(
        self,
        task_id: str,
        request: TaskRequest,
        gpu_devices: list[int],
    ) -> TaskResponse:
        """Execute a task on specified GPU devices.

        Args:
            task_id: Task ID.
            request: Task request.
            gpu_devices: List of GPU device IDs to use.

        Returns:
            TaskResponse with execution result.
        """
        task = self._tasks[task_id]
        task.gpu_devices = gpu_devices
        task.started_at = datetime.now()
        task.status = TaskStatus.RUNNING

        # Set up environment
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in gpu_devices)
        if request.env_vars:
            env.update(request.env_vars)

        # Prepare command
        script_path = Path(request.bash_scripts_path)

        if request.run_backend:
            # Run in tmux session
            tmux_session = f"gpu_backend_{task_id}"
            task.tmux_session = tmux_session

            cmd = [
                "tmux", "new-session", "-d", "-s", tmux_session,
                f"cd {script_path.parent} && bash {script_path.name}"
            ]

            try:
                subprocess.run(cmd, check=True, timeout=10)
                return TaskResponse(
                    task_id=task_id,
                    status=TaskStatus.RUNNING,
                    message=f"Task running in tmux session: {tmux_session}",
                    gpu_devices=gpu_devices,
                    tmux_session=tmux_session,
                )
            except subprocess.SubprocessError as e:
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now()
                return TaskResponse(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    message=f"Failed to start tmux session: {e}",
                )
        else:
            # Run as subprocess
            try:
                process = subprocess.Popen(
                    ["bash", str(script_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=str(script_path.parent),
                    env=env,
                    text=True,
                )

                self._processes[task_id] = process
                task.pid = process.pid

                # Start background monitoring
                asyncio.create_task(
                    self._monitor_task(task_id, process, request.timeout)
                )

                if request.require_instant_feedback:
                    # Wait for completion
                    stdout, stderr = process.communicate(timeout=request.timeout)
                    task.exit_code = process.returncode
                    task.completed_at = datetime.now()
                    task.status = (
                        TaskStatus.COMPLETED if process.returncode == 0
                        else TaskStatus.FAILED
                    )
                    task.stdout = stdout
                    task.stderr = stderr

                    return TaskResponse(
                        task_id=task_id,
                        status=task.status,
                        message=f"Task completed with exit code {process.returncode}",
                        gpu_devices=gpu_devices,
                        pid=process.pid,
                    )
                else:
                    return TaskResponse(
                        task_id=task_id,
                        status=TaskStatus.RUNNING,
                        message="Task started in background",
                        gpu_devices=gpu_devices,
                        pid=process.pid,
                    )

            except Exception as e:
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now()
                return TaskResponse(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    message=f"Failed to start task: {e}",
                )

    async def _monitor_task(
        self,
        task_id: str,
        process: subprocess.Popen,
        timeout: Optional[int],
    ) -> None:
        """Monitor a running task in the background.

        Args:
            task_id: Task ID.
            process: Subprocess to monitor.
            timeout: Optional timeout in seconds.
        """
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            task = self._tasks.get(task_id)
            if task:
                task.exit_code = process.returncode
                task.completed_at = datetime.now()
                task.status = (
                    TaskStatus.COMPLETED if process.returncode == 0
                    else TaskStatus.FAILED
                )
                task.stdout = stdout
                task.stderr = stderr

                logger.info(
                    f"Task {task_id} completed with exit code {process.returncode}"
                )
        except subprocess.TimeoutExpired:
            # Kill the process on timeout
            process.kill()
            process.wait()
            task = self._tasks.get(task_id)
            if task:
                task.status = TaskStatus.KILLED
                task.completed_at = datetime.now()
                task.exit_code = -1
                task.stderr = "Task killed due to timeout"

            logger.warning(f"Task {task_id} killed due to timeout")
        except Exception as e:
            logger.error(f"Error monitoring task {task_id}: {e}")
        finally:
            self._processes.pop(task_id, None)
            # Try to run next queued task
            await self._process_queue()

    async def _process_queue(self) -> None:
        """Process the next task in the queue."""
        async with self._lock:
            if not self._queue:
                return

            # Get next task from queue
            task_id = self._queue[0]
            task = self._tasks.get(task_id)

            if not task:
                self._queue.pop(0)
                return

            # Check if we can run it
            # For queued tasks, we need to reconstruct the request
            # This is simplified - in production, store the request
            available_devices = self.gpu_monitor.get_available_devices(1)
            if available_devices:
                self._queue.pop(0)
                task.gpu_devices = available_devices
                task.started_at = datetime.now()
                task.status = TaskStatus.RUNNING

                logger.info(f"Starting queued task {task_id}")

    async def get_task_status(self, task_id: str) -> Optional[TaskInfo]:
        """Get the status of a task.

        Args:
            task_id: Task ID to check.

        Returns:
            TaskInfo if found, None otherwise.
        """
        return self._tasks.get(task_id)

    async def get_task_log(self, task_id: str) -> Optional[TaskLog]:
        """Get the log output of a task.

        Args:
            task_id: Task ID.

        Returns:
            TaskLog with stdout/stderr, or None if not found.
        """
        task = self._tasks.get(task_id)
        if not task:
            return None

        if task.run_backend and task.tmux_session:
            # Get tmux session output
            try:
                result = subprocess.run(
                    ["tmux", "capture-pane", "-t", task.tmux_session, "-p"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                return TaskLog(
                    task_id=task_id,
                    stdout=result.stdout,
                    stderr="",
                    exit_code=None,
                    is_running=True,
                )
            except subprocess.SubprocessError:
                pass

        return TaskLog(
            task_id=task_id,
            stdout=task.stdout or "",
            stderr=task.stderr or "",
            exit_code=task.exit_code,
            is_running=task.status == TaskStatus.RUNNING,
        )

    async def kill_task(self, task_id: str) -> bool:
        """Kill a running task.

        Args:
            task_id: Task ID to kill.

        Returns:
            True if task was killed, False otherwise.
        """
        task = self._tasks.get(task_id)
        if not task:
            return False

        if task.status != TaskStatus.RUNNING:
            return False

        # Kill subprocess
        process = self._processes.get(task_id)
        if process:
            try:
                process.kill()
                process.wait(timeout=5)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                pass

        # Kill tmux session
        if task.tmux_session:
            try:
                subprocess.run(
                    ["tmux", "kill-session", "-t", task.tmux_session],
                    check=True,
                    timeout=5,
                )
            except subprocess.SubprocessError:
                pass

        task.status = TaskStatus.KILLED
        task.completed_at = datetime.now()
        task.exit_code = -1

        return True

    async def get_running_tasks(self) -> list[TaskInfo]:
        """Get all running tasks."""
        return [
            task for task in self._tasks.values()
            if task.status == TaskStatus.RUNNING
        ]

    async def get_queued_tasks(self) -> list[TaskInfo]:
        """Get all queued tasks."""
        return [
            self._tasks[task_id]
            for task_id in self._queue
            if task_id in self._tasks
        ]

    async def get_tmux_sessions(self) -> list[dict[str, Any]]:
        """Get all tmux sessions started by the backend."""
        try:
            result = subprocess.run(
                ["tmux", "list-sessions"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            sessions = []
            for line in result.stdout.strip().split("\n"):
                if "gpu_backend_" in line:
                    parts = line.split(":")
                    if parts:
                        sessions.append({
                            "name": parts[0],
                            "details": line,
                        })
            return sessions
        except subprocess.SubprocessError:
            return []
