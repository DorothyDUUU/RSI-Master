"""EvoMaster Docker Session implementation.

Docker-container-based Session. The heavy lifting (container lifecycle,
``docker exec``-based command execution, file I/O) lives in
:class:`evomaster.env.docker.DockerEnv`; this class is a thin Session
adapter on top.

The earlier version of this file relied on a long-running tmux session and
multi-line PS1 markers to track command boundaries. That approach was both
fragile (broke on alpine images, on quoting edge cases, and on commands that
echoed the marker text) and tightly coupled to a single thread of execution.
The current design simply forwards each call to ``DockerEnv.exec_bash_stateful``,
which runs one ``docker exec`` per command and persists the working directory
through a state file inside the container.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from evomaster.env.docker import DockerEnv, DockerEnvConfig

from .base import BaseSession, SessionConfig


class DockerSessionConfig(SessionConfig):
    """Docker Session configuration.

    Mirrors the YAML keys under ``session.docker``. Field-by-field:

    * ``image``: Docker image to run (must include ``bash``).
    * ``container_name``: name to give the new container; if it already
      exists it is reused. ``None`` -> auto-generated.
    * ``use_existing_container``: attach to an already-running container by
      name or id; takes precedence over ``container_name`` and disables our
      own lifecycle management for it.
    * ``working_dir``: cwd used inside the container, also exposed as
      ``workspace_path`` for compatibility with the Local session.
    * ``memory_limit`` / ``cpu_limit``: passed straight through to
      ``--memory`` / ``--cpus``.
    * ``gpu_devices``: ``"all"``, a single index ``"0"``, a comma list
      ``"0,1"``, or a Python list. ``None`` means no GPU.
    * ``network_mode``: ``"bridge"``, ``"host"``, ``"none"``, or any
      user-defined network.
    * ``volumes``: ``{host_path: container_path}``. Relative host paths are
      resolved against ``cwd`` at container-creation time.
    * ``env_vars``: extra environment variables for the container.
    * ``auto_remove``: when ``True``, ``docker run --rm`` is used and the
      container is stopped + removed at session close. When ``False`` we
      leave the container running so a subsequent ``open()`` can reuse it.
    * ``pull_image``: ``"missing"`` (default) / ``"always"`` / ``"never"``.
    * ``timeout``: default per-command timeout in seconds.
    """

    image: str = Field(
        default="python:3.11-slim",
        description="Docker image (must include bash)",
    )
    container_name: str | None = Field(
        default=None,
        description="Container name; if it already exists it is reused. None for auto-generated.",
    )
    working_dir: str = Field(
        default="/workspace",
        description="Working directory inside the container",
    )
    memory_limit: str | None = Field(
        default="4g",
        description="Memory limit (e.g. '4g'). None / empty disables the flag.",
    )
    cpu_limit: float = Field(
        default=2.0,
        description="CPU limit in cores; <= 0 disables the flag.",
    )
    cpu_devices: str | list[int] | None = Field(
        default=None,
        description=(
            "CPU set (pinning), passed as --cpuset-cpus. Restricts which "
            "physical CPUs the container sees (unlike --cpus, which only "
            "throttles total CPU time). Formats: '0-15', '0,2,4', or a list "
            "like [0, 1, 2, 3]. None means no pinning."
        ),
    )
    gpu_devices: str | list[str] | None = Field(
        default=None,
        description=(
            "GPU spec: 'all', a single id ('0'), a comma list ('0,1'), or a "
            "Python list (['0','1']). None / 'none' disables GPUs."
        ),
    )
    network_mode: str = Field(
        default="bridge",
        description="Network mode (bridge / host / none / custom)",
    )
    volumes: dict[str, str] = Field(
        default_factory=dict,
        description="Bind mounts as {host_path: container_path}; relative host paths are resolved.",
    )
    env_vars: dict[str, str] = Field(
        default_factory=dict,
        description="Extra environment variables for the container",
    )
    auto_remove: bool = Field(
        default=True,
        description="When True, stop+remove the container on session close (and pass --rm).",
    )
    use_existing_container: str | None = Field(
        default=None,
        description="Attach to an existing container by name or id; never created or removed.",
    )
    pull_image: str = Field(
        default="missing",
        description="Image pull policy: 'missing' (default), 'always', or 'never'.",
    )
    config_dir: str | None = Field(
        default=None,
        description=(
            "Directory of the YAML config file, used to resolve relative "
            "host paths in `volumes`. When set, relative paths are resolved "
            "against the project root (walked up from `config_dir` until a "
            "directory containing `evomaster/` is found); falls back to the "
            "current working directory otherwise."
        ),
    )


class DockerSession(BaseSession):
    """Docker-backed Session.

    Each ``exec_bash`` call is a separate ``docker exec``. Working directory
    persists across consecutive calls in the same thread; exported environment
    variables and background jobs do not (chain with ``&&`` if needed).
    """

    def __init__(self, config: DockerSessionConfig | None = None):
        """Initialize the Docker session.

        Args:
            config: Docker session configuration. ``None`` falls back to defaults.
        """
        super().__init__(config)
        self.config: DockerSessionConfig = config or DockerSessionConfig()
        env_config = DockerEnvConfig(session_config=self.config)
        self._env = DockerEnv(env_config)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def open(self) -> None:
        """Start (or attach to) the Docker container."""
        if self._is_open:
            self.logger.warning("Session already open")
            return
        if not self._env.is_ready:
            self._env.setup()
        self._is_open = True
        self.logger.info(
            f"Docker session opened "
            f"(container={self._env.container_name}, "
            f"id={(self._env.container_id or '')[:12]})"
        )

    def close(self) -> None:
        """Close the session.

        Honors ``auto_remove``: when ``True`` the container is stopped and
        removed; when ``False`` the container is left running so the next
        ``open()`` can reuse it.
        """
        if not self._is_open:
            return
        if self._env.is_ready:
            self._env.teardown()
        self._is_open = False
        self.logger.info("Session closed")

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #

    @property
    def container_id(self) -> str | None:
        """Active container id (full hash), or ``None`` if not yet opened."""
        return self._env.container_id

    @property
    def container_name(self) -> str | None:
        return self._env.container_name

    # ------------------------------------------------------------------ #
    # Command execution
    # ------------------------------------------------------------------ #

    def exec_bash(
        self,
        command: str,
        timeout: int | None = None,
        is_input: bool = False,
    ) -> dict[str, Any]:
        """Run a bash command inside the container.

        Working directory persists between calls (per thread). ``is_input``
        is not supported in this Session and returns a clear error message
        — same contract as :class:`LocalSession`.
        """
        if not self._is_open:
            raise RuntimeError("Session not open")

        timeout = timeout or self.config.timeout
        command = command.strip()

        if is_input:
            return {
                "stdout": (
                    "ERROR: Docker session does not support is_input mode. "
                    "Run interactive workflows via a single chained command "
                    "(e.g. with heredocs) instead."
                ),
                "stderr": "",
                "exit_code": 1,
                "working_dir": self.config.working_dir,
                "output": "",
            }

        if command == "":
            # Empty command: no-op. Mirrors the agent contract for "give me
            # more logs", which has no analog in this Session.
            return {
                "stdout": "",
                "stderr": "",
                "exit_code": 0,
                "working_dir": self.config.working_dir,
                "output": "",
            }

        return self._env.exec_bash_stateful(command, timeout=timeout)

    def to_session_path(self, host_path: str) -> str:
        """Translate a host path to the matching path inside the container.

        Looks up the DockerEnv volume mappings; if ``host_path`` falls under
        one of them, returns the in-container path. When no mount covers the
        path we fall back to the original host path — the caller will get a
        clear "file not found" error inside the container, which is easier
        to diagnose than a silent mistranslation.
        """
        container_path = self._env.host_to_container_path(host_path)
        if container_path is not None:
            return container_path
        return host_path

    # ------------------------------------------------------------------ #
    # File I/O — delegate to DockerEnv
    # ------------------------------------------------------------------ #

    def upload(self, local_path: str, remote_path: str) -> None:
        if not self._is_open:
            raise RuntimeError("Session not open")
        self._env.upload_file(local_path, remote_path)

    def read_file(self, remote_path: str, encoding: str = "utf-8") -> str:
        if not self._is_open:
            raise RuntimeError("Session not open")
        return self._env.read_file_content(remote_path, encoding)

    def write_file(self, remote_path: str, content: str, encoding: str = "utf-8") -> None:
        if not self._is_open:
            raise RuntimeError("Session not open")
        self._env.write_file_content(remote_path, content, encoding)

    def download(self, remote_path: str, timeout: int | None = None) -> bytes:
        if not self._is_open:
            raise RuntimeError("Session not open")
        return self._env.download_file(remote_path, timeout)

    def path_exists(self, remote_path: str) -> bool:
        if not self._is_open:
            raise RuntimeError("Session not open")
        return self._env.path_exists(remote_path)

    def is_file(self, remote_path: str) -> bool:
        if not self._is_open:
            raise RuntimeError("Session not open")
        return self._env.is_file(remote_path)

    def is_directory(self, remote_path: str) -> bool:
        if not self._is_open:
            raise RuntimeError("Session not open")
        return self._env.is_directory(remote_path)
