"""Data models for GPU backend server — session-based architecture."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, computed_field


# ── Session models ───────────────────────────────────────────────────────────

class SessionStatus(str, Enum):
    """Shell session lifecycle."""
    ACTIVE = "active"
    KILLED = "killed"


class ShellSession(BaseModel):
    """A persistent tmux shell session."""
    session_id: str = Field(..., description="Unique session ID, e.g. gpu_shell_<hash>")
    status: SessionStatus = Field(default=SessionStatus.ACTIVE)
    cwd: str = Field(default="/", description="Current working directory")
    created_at: datetime = Field(default_factory=datetime.now)
    last_command: str | None = Field(default=None, description="Last command executed in this session")
    is_background: bool = Field(default=False, description="Whether this is a background tmux session")
    log_path: str | None = Field(default=None, description="Log file path for background sessions")


class SessionExecRequest(BaseModel):
    """Execute a command inside a shell session.

    When *session_id* is ``None`` a fresh tmux session is created and its id /
    cwd are returned alongside the command output.
    """
    command: str = Field(..., description="Shell command to execute")
    timeout: int = Field(default=300, ge=0, description="Timeout in seconds; 0 = no timeout")
    session_id: str | None = Field(default=None, description="Target session; None → create new session")


class SessionExecResponse(BaseModel):
    """Result of executing a command in a shell session."""
    session_id: str = Field(..., description="Session ID the command ran in")
    return_code: int = Field(..., description="Command exit code")
    stdout: str = Field(default="")
    stderr: str = Field(default="")
    cwd: str | None = Field(default=None, description="Working directory (only populated for new sessions)")
    is_new: bool = Field(default=False, description="True if a new session was created")


# ── Script models ────────────────────────────────────────────────────────────

class ScriptPriority(str, Enum):
    """Script scheduling priority."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class ScriptStatus(str, Enum):
    """Script execution lifecycle."""
    QUEUED = "queued"          # waiting for GPU resources
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"
    REJECTED = "rejected"      # admission backpressure; never launched


class ScriptSubmitRequest(BaseModel):
    """Submit a bash script for execution.

    If *resume_session_id* is set the script is ``source``'d inside that
    session so it inherits environment variables.
    """
    command_path: str = Field(..., description="Absolute path to the bash script")
    queue_owner: str | None = Field(default=None, description="Run-scoped worker identity, supplied by ExpOS")
    timeout: int = Field(default=300, ge=0, description="Timeout in seconds; 0 = no timeout")
    resume_session_id: str | None = Field(default=None, description="If set, source the script in this session to inherit its env")
    run_in_background: bool = Field(default=False, description="Run in a background tmux session with log redirection")
    gpu_nums: int = Field(..., ge=0, description="Number of GPU devices required; 0 = no GPU")
    priority: ScriptPriority = Field(default=ScriptPriority.NORMAL, description="Scheduling priority: low, normal, high")
    max_mem_quota: int | None = Field(default=None, description="Minimum free GPU memory (MB) required on each device; None = no minimum")
    need_exclusive_gpu: bool = Field(default=True, description="Exclusively hold the allocated GPU(s) for the whole task lifetime (default True 2026-08-29); when False the card opens to other non-exclusive tasks after a short lease")


class ScriptInfo(BaseModel):
    """Information about a submitted script."""
    script_id: str = Field(..., description="Unique script ID, e.g. gpu_script_<hash>")
    command_path: str
    queue_owner: str | None = None
    status: ScriptStatus = Field(default=ScriptStatus.QUEUED)
    session_id: str | None = Field(default=None, description="Tmux session name if running")
    log_path: str | None = Field(default=None, description="Log file path for background scripts")
    return_code: int | None = Field(default=None)
    stdout: str = Field(default="")
    stderr: str = Field(default="")
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: datetime | None = Field(default=None)
    completed_at: datetime | None = Field(default=None)
    gpu_devices: list[int] = Field(default_factory=list, description="Allocated GPU device IDs")
    priority: ScriptPriority = Field(default=ScriptPriority.NORMAL)
    gpu_nums: int = Field(default=1)
    max_mem_quota: int | None = Field(default=None)
    need_exclusive_gpu: bool = Field(default=True, description="True → hold GPU(s) for the whole lifetime (default True 2026-08-29); False → card opens to other non-exclusive tasks after the lease window")


class ScriptSubmitResponse(BaseModel):
    """Result of submitting a script."""
    script_id: str
    session_id: str | None = Field(default=None)
    log_path: str | None = Field(default=None)
    status: ScriptStatus
    return_code: int | None = Field(default=None)
    stdout: str = Field(default="")
    stderr: str = Field(default="")
    message: str = Field(default="")
    reason_code: str = ""


class ScriptLog(BaseModel):
    """Log output of a script."""
    script_id: str
    stdout: str = Field(default="")
    stderr: str = Field(default="")
    return_code: int | None = Field(default=None)
    is_running: bool = Field(default=False)


# ── GPU / resource models ────────────────────────────────────────────────────

class GPUDevice(BaseModel):
    """GPU device information."""
    device_id: int
    name: str
    total_memory_mb: int
    used_memory_mb: int
    free_memory_mb: int
    gpu_utilization: float = Field(..., description="GPU core utilization percentage (0-100)")
    memory_utilization: float = Field(..., description="Memory bandwidth utilization percentage from nvidia-smi (0-100)")
    temperature: float | None = None
    processes: list[dict[str, Any]] = Field(default_factory=list)

    @computed_field
    @property
    def vram_usage_percent(self) -> float:
        """Actual VRAM usage percentage (used / total * 100)."""
        if self.total_memory_mb <= 0:
            return 0.0
        return (self.used_memory_mb / self.total_memory_mb) * 100.0


class GPUInfo(BaseModel):
    """Overall GPU and system information."""
    devices: list[GPUDevice]
    total_devices: int
    available_devices: list[int] = Field(default_factory=list, description="Device IDs available for new tasks")
    cpu_percent: float = Field(default=0.0)
    memory_percent: float = Field(default=0.0)
    memory_total_gb: float = Field(default=0.0)
    memory_used_gb: float = Field(default=0.0)


# ── Aggregate status models ──────────────────────────────────────────────────

class QueueItem(BaseModel):
    script_id: str
    priority: str
    gpu_nums: int

class BackendStatus(BaseModel):
    """Aggregated backend status (sessions + resources)."""
    sessions: list[ShellSession]
    gpu_info: GPUInfo
    queue: list[QueueItem] = Field(default_factory=list)
