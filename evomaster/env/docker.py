"""Docker environment implementation.

Provides the low-level operations interface for Docker containers.

Design notes
============
The previous version of this module relied on a long-running tmux session
inside the container plus a multi-line PS1 marker scheme. That design was
fragile: it depended on apt-get availability, multi-line PS1 quoting through
``tmux send-keys``, and on the markers not appearing in user output. It also
silently broke when host volume paths were relative (Docker treats those as
named volumes), and when ``auto_remove=true`` was combined with the post-stop
``docker rm -f`` call.

The current implementation is much simpler: one ``docker exec`` per command,
with the per-thread working directory persisted to a state file inside the
container. Concrete consequences:

* No tmux dependency, no PS1 magic; works against any image with ``bash``.
* Working directory persists across consecutive ``exec_bash`` calls in the
  same thread (so ``cd`` inside one command is visible to the next), which
  matches the contract advertised by ``BashTool`` ("persistent shell").
* Exported environment variables and background processes do **not** persist
  across calls. Agents that need that should chain commands with ``&&``.
* Each thread keeps its own cwd state, so parallel exps that share a single
  container do not stomp on each other (used by the parallel playground).
* Volume host paths are resolved to absolute paths (and created on the host)
  before being passed to ``docker run`` so relative paths in YAML work.
* ``is_input`` (interactive STDIN to a still-running command) is not
  supported; the session returns a clear error message, mirroring the
  behavior of ``LocalSession``.
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import Field

from .base import BaseEnv, EnvConfig
from evomaster.agent.session.base import SessionConfig


class DockerEnvConfig(EnvConfig):
    """Docker environment configuration."""
    session_config: SessionConfig = Field(
        ...,
        description="Session configuration",
    )


# Marker line emitted by the bash wrapper at the end of every stateful
# command. Includes the exit code and the new working directory so the
# session can update its cached cwd without an extra ``docker exec`` call.
_EVO_MARKER = "__EVOMASTER_CMD_END__"


def _find_project_root(start: Path) -> Path | None:
    """Walk up from ``start`` looking for a directory containing ``evomaster/``.

    Mirrors the heuristic used by :mod:`evomaster.env.local` for symlink
    resolution, so relative volume paths in YAML ("./assets") behave the
    same way regardless of the session type.
    """
    try:
        current = start.resolve()
    except Exception:
        current = start
    while current != current.parent:
        candidate = current / "evomaster"
        if candidate.exists() and candidate.is_dir():
            return current
        current = current.parent
    return None


def _normalize_cpuset_spec(
    cpu_devices: "str | list[int] | list[str] | None",
) -> str | None:
    """Normalize ``cpu_devices`` into a string usable as ``--cpuset-cpus``.

    Accepts the same shapes as the local session's ``cpu_devices``:
    ``"0-15"``, ``"0,2,4"``, ``[0, 1, 2, 3]``, or ``None``.
    """
    if cpu_devices is None:
        return None
    if isinstance(cpu_devices, str):
        s = cpu_devices.strip()
        return s or None
    if isinstance(cpu_devices, (list, tuple)):
        if not cpu_devices:
            return None
        return ",".join(str(c) for c in cpu_devices)
    return str(cpu_devices)


def _resolve_host_path(host_path: str, config_dir: str | None = None) -> str:
    """Resolve a (possibly relative or ``~``-expanded) host path to an absolute path.

    Docker requires absolute host paths for bind mounts; a relative path
    silently gets interpreted as a *named volume*, which is almost never
    what the user intended. We also expand ``~`` here.

    Resolution order for relative paths:

    1. Project root discovered by walking up from ``config_dir`` until a
       directory containing ``evomaster/`` is found (same logic the local
       session uses for symlinks). This is what users typically expect for
       ``./assets`` / ``./data`` style entries in YAML.
    2. Project root discovered the same way from the current working
       directory.
    3. Current working directory itself (legacy fallback).
    """
    p = Path(os.path.expanduser(host_path))
    if p.is_absolute():
        return str(p.resolve())

    base: Path | None = None
    if config_dir:
        base = _find_project_root(Path(config_dir))
    if base is None:
        base = _find_project_root(Path.cwd())
    if base is None:
        base = Path.cwd()

    return str((base / p).resolve())


class DockerEnv(BaseEnv):
    """Docker environment implementation.

    Manages a single Docker container and exposes command/file primitives
    against it. See module docstring for the design rationale.
    """

    def __init__(self, config: DockerEnvConfig | None = None):
        """Initialize the Docker environment.

        Args:
            config: Docker environment configuration. Must include
                ``session_config``.
        """
        if config is None:
            raise ValueError("DockerEnv requires DockerEnvConfig with session_config")
        super().__init__(config)
        self.config: DockerEnvConfig = config
        self._container_id: str | None = None
        self._container_name: str | None = None
        # Whether *we* should be the one to stop/remove the container at
        # teardown time. Always False when ``use_existing_container`` is set,
        # so that we never tear down something the user owns.
        self._created_by_us: bool = False
        # Stable per-env identifier used to namespace state files inside the
        # container. Lets multiple DockerEnv instances coexist if they ever
        # share a container.
        self._session_uid: str = uuid.uuid4().hex[:12]
        # Per-thread initialization state for the in-container state dir.
        self._thread_local = threading.local()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def setup(self) -> None:
        """Initialize the Docker environment.

        Verifies the docker CLI is reachable, creates or attaches to a
        container, and seeds the per-session state directory inside it.
        """
        if self._is_ready:
            self.logger.warning("Environment already setup")
            return

        self.logger.info("Setting up Docker environment")
        self._ensure_docker_available()
        # Pre-container step: expose any non-workspace volumes as symlinks
        # inside the host workspace, so the user can browse the *same* tree
        # from the host that the container sees. Runs *before* the container
        # is created because it only touches host paths.
        self._setup_workspace_symlinks()
        self._create_or_get_container()
        self._initialize_container()
        self._is_ready = True
        self.logger.info(
            f"Docker environment ready: container={self._container_name} "
            f"({(self._container_id or '')[:12]})"
        )

    def teardown(self) -> None:
        """Clean up Docker environment resources.

        - If we created the container and ``auto_remove`` is true: stop and
          remove it.
        - If we created the container but ``auto_remove`` is false: leave it
          running so a subsequent ``open()`` call can reuse it.
        - If we attached to an existing container (``use_existing_container``):
          do nothing — never touch a container the user owns.
        """
        if not self._is_ready:
            return

        self.logger.info("Tearing down Docker environment")
        if self._container_id and self._created_by_us:
            sc = self.config.session_config
            if sc.auto_remove:
                self._stop_and_remove_container()
            else:
                self.logger.info(
                    f"Keeping container {self._container_id[:12]} running for reuse "
                    f"(auto_remove=false)"
                )
        elif self._container_id:
            self.logger.info(
                f"Detached from existing container {self._container_id[:12]} "
                f"(left untouched)"
            )

        # Reset thread-local init flags so a re-opened env will re-seed them.
        self._thread_local = threading.local()
        self._is_ready = False
        self.logger.info("Docker environment teardown complete")

    def _setup_workspace_symlinks(self) -> None:
        """Mirror non-workspace bind mounts into the host workspace via symlinks.

        Motivation: when the user configures multiple ``volumes`` with
        targets nested inside ``working_dir`` (e.g. a workspace bind mount
        at ``/workspace`` *and* ``/data/foo`` → ``/workspace/foo``),
        Docker layers the second mount on top of the first inside the
        container. From the host, however, ``<host_workspace>/foo`` is an
        ordinary empty directory because the host bind mount of
        ``/workspace`` doesn't know about the nested mount. Agents and
        users that want to inspect their runs from the host end up seeing
        an empty ``foo/`` under ``runs/<task>/workspace/``.

        We compensate by creating a symlink ``<host_workspace>/foo →
        <resolved host path for /data/foo>`` on the host, so the host
        view matches the in-container view.

        Safe to call repeatedly: existing symlinks are replaced, already-
        populated directories are left alone.
        """
        sc = self.config.session_config
        working_dir = (sc.working_dir or "/workspace").rstrip("/")
        if not working_dir:
            return

        volumes = sc.volumes or {}
        if not volumes:
            return

        config_dir = getattr(sc, "config_dir", None)

        # Locate the host path mounted at working_dir (if any).
        host_workspace: Path | None = None
        for host_path, container_path in volumes.items():
            cp = str(container_path).rstrip("/")
            if cp == working_dir:
                host_workspace = Path(
                    _resolve_host_path(host_path, config_dir=config_dir)
                )
                break
        if host_workspace is None:
            return

        try:
            host_workspace.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.warning(
                f"Cannot create host workspace {host_workspace}: {e}"
            )
            return

        for host_path, container_path in volumes.items():
            cp = str(container_path).rstrip("/")
            if cp == working_dir:
                continue
            prefix = working_dir + "/"
            if not cp.startswith(prefix):
                continue
            rel = cp[len(prefix):]
            if not rel:
                continue
            link_path = host_workspace / rel
            resolved_source = Path(
                _resolve_host_path(host_path, config_dir=config_dir)
            )

            try:
                link_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self.logger.warning(
                    f"Cannot create parent of {link_path}: {e}"
                )
                continue

            # Remove any prior placeholder. Only drop directories that are
            # either empty or already symlinks (never blow away real data).
            try:
                if link_path.is_symlink():
                    link_path.unlink()
                elif link_path.is_dir():
                    try:
                        link_path.rmdir()  # only succeeds if empty
                    except OSError:
                        self.logger.info(
                            f"Workspace path {link_path} is a non-empty "
                            f"directory; leaving it in place instead of "
                            f"replacing with a symlink to {resolved_source}."
                        )
                        continue
                elif link_path.exists():
                    # A regular file at that path would be destroyed by a
                    # symlink create; skip rather than delete user data.
                    self.logger.info(
                        f"Workspace path {link_path} already exists as a "
                        f"file; leaving it in place."
                    )
                    continue
            except Exception as e:
                self.logger.warning(
                    f"Could not clear existing path {link_path}: {e}"
                )
                continue

            try:
                os.symlink(resolved_source, link_path)
                self.logger.info(
                    f"Workspace symlink: {link_path} -> {resolved_source} "
                    f"(container path: {container_path})"
                )
            except OSError as e:
                self.logger.warning(
                    f"Failed to create workspace symlink "
                    f"{link_path} -> {resolved_source}: {e}"
                )

    def _stop_and_remove_container(self) -> None:
        """Stop and remove the container we created. Tolerant to ``--rm`` races."""
        cid = self._container_id
        if not cid:
            return
        self.logger.info(f"Stopping and removing container {cid[:12]}")
        try:
            subprocess.run(
                ["docker", "stop", cid],
                capture_output=True, text=True, timeout=60,
            )
        except subprocess.TimeoutExpired:
            self.logger.warning(f"`docker stop` timed out for {cid[:12]}")
        # If the container was started with ``--rm`` it is auto-removed once
        # ``docker stop`` returns; the subsequent ``docker rm -f`` then exits
        # non-zero. Treat that as success.
        try:
            subprocess.run(
                ["docker", "rm", "-f", cid],
                capture_output=True, text=True, timeout=30,
            )
        except subprocess.TimeoutExpired:
            self.logger.warning(f"`docker rm` timed out for {cid[:12]}")
        self._container_id = None
        self._container_name = None

    # ------------------------------------------------------------------ #
    # Container creation
    # ------------------------------------------------------------------ #

    def _ensure_docker_available(self) -> None:
        """Raise a clear error if the docker CLI / daemon is not reachable."""
        try:
            r = subprocess.run(
                ["docker", "version", "--format", "{{.Server.Version}}"],
                capture_output=True, text=True, timeout=10,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "`docker` CLI not found in PATH. Install Docker and ensure the "
                "daemon is running before using a Docker session."
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                "Timed out talking to the docker daemon (try `docker info`)."
            )
        if r.returncode != 0:
            raise RuntimeError(
                "Docker daemon unreachable: "
                f"{(r.stderr or r.stdout).strip() or 'unknown error'}"
            )

    def _container_running(self, ref: str) -> bool:
        r = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", ref],
            capture_output=True, text=True, timeout=10,
        )
        return r.returncode == 0 and r.stdout.strip() == "true"

    def _container_exists(self, ref: str) -> bool:
        r = subprocess.run(
            ["docker", "inspect", "-f", "{{.Id}}", ref],
            capture_output=True, text=True, timeout=10,
        )
        return r.returncode == 0 and bool(r.stdout.strip())

    def _container_id_from_ref(self, ref: str) -> str | None:
        r = subprocess.run(
            ["docker", "inspect", "-f", "{{.Id}}", ref],
            capture_output=True, text=True, timeout=10,
        )
        return r.stdout.strip() if r.returncode == 0 else None

    def _ensure_image_available(self, image: str) -> None:
        """Pull the image if local lookup misses, per ``pull_image`` policy.

        Policies (read from ``session_config.pull_image`` if present,
        defaulting to ``"missing"``):

        * ``"missing"`` (default): pull only if not present locally.
        * ``"always"``: pull every time (useful for moving tags).
        * ``"never"``: assume the image is already local; raise if not.
        """
        sc = self.config.session_config
        policy = getattr(sc, "pull_image", "missing")
        if policy not in {"missing", "always", "never"}:
            self.logger.warning(
                f"Unknown pull_image policy '{policy}'; falling back to 'missing'"
            )
            policy = "missing"

        if policy == "always":
            self.logger.info(f"Pulling image (policy=always): {image}")
            r = subprocess.run(
                ["docker", "pull", image],
                capture_output=True, text=True, timeout=600,
            )
            if r.returncode != 0:
                raise RuntimeError(
                    f"Failed to pull image {image}: {(r.stderr or r.stdout).strip()}"
                )
            return

        # missing / never: check first
        check = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True, text=True, timeout=15,
        )
        if check.returncode == 0:
            return  # already present
        if policy == "never":
            raise RuntimeError(
                f"Image '{image}' not present locally and pull_image='never'"
            )
        # missing: pull now
        self.logger.info(f"Image not found locally; pulling: {image}")
        r = subprocess.run(
            ["docker", "pull", image],
            capture_output=True, text=True, timeout=600,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"Failed to pull image {image}: {(r.stderr or r.stdout).strip()}"
            )

    def _create_or_get_container(self) -> None:
        """Create a new container, or attach to an existing one.

        Resolution order:

        1. ``use_existing_container`` is set → attach (never create / remove).
        2. ``container_name`` matches an existing container → reuse it (start
           if stopped). We claim ownership for lifecycle purposes (the user
           explicitly named it for us).
        3. Otherwise create a fresh container with all the configured limits.
        """
        sc = self.config.session_config

        # 1) Explicit attach.
        if sc.use_existing_container:
            ref = sc.use_existing_container
            if not self._container_exists(ref):
                raise RuntimeError(
                    f"use_existing_container='{ref}' does not exist."
                )
            if not self._container_running(ref):
                self.logger.info(f"Starting existing container {ref}")
                r = subprocess.run(
                    ["docker", "start", ref],
                    capture_output=True, text=True, timeout=60,
                )
                if r.returncode != 0:
                    raise RuntimeError(
                        f"Failed to start container '{ref}': {(r.stderr or r.stdout).strip()}"
                    )
            self._container_id = self._container_id_from_ref(ref)
            self._container_name = ref
            self._created_by_us = False
            return

        # 2) Named container reuse.
        container_name = sc.container_name or (
            f"evomaster-{os.getpid()}-{int(time.time())}-{self._session_uid}"
        )
        if sc.container_name and self._container_exists(container_name):
            self.logger.info(
                f"Found existing container '{container_name}'; reusing it. "
                f"Note: image / volume / env_vars settings in the current "
                f"config are ignored when reusing."
            )
            if not self._container_running(container_name):
                r = subprocess.run(
                    ["docker", "start", container_name],
                    capture_output=True, text=True, timeout=60,
                )
                if r.returncode != 0:
                    raise RuntimeError(
                        f"Failed to start container '{container_name}': "
                        f"{(r.stderr or r.stdout).strip()}"
                    )
            self._container_id = self._container_id_from_ref(container_name)
            self._container_name = container_name
            # The user named this for us -> we own its lifecycle.
            self._created_by_us = True
            return

        # 3) Fresh container.
        self._ensure_image_available(sc.image)

        cmd: list[str] = ["docker", "run", "-d", "--name", container_name]

        # Resource limits (only emit flags when set, so users can disable
        # them by setting an empty value or zero).
        if sc.memory_limit:
            cmd.extend(["--memory", str(sc.memory_limit)])
            # memory + swap to the same value so the container actually
            # hits OOM at `memory_limit` instead of silently spilling into
            # swap. Users can disable by passing `memory_limit: null`.
            cmd.extend(["--memory-swap", str(sc.memory_limit)])
        if sc.cpu_limit and float(sc.cpu_limit) > 0:
            cmd.extend(["--cpus", str(sc.cpu_limit)])
        # CPU pinning. Unlike `--cpus`, `--cpuset-cpus` actually restricts
        # which cores the container can see/use, so htop inside the
        # container only shows the pinned subset.
        cpuset_spec = _normalize_cpuset_spec(getattr(sc, "cpu_devices", None))
        if cpuset_spec:
            cmd.extend(["--cpuset-cpus", cpuset_spec])

        # GPU: ``--gpus`` syntax differs by spec; we normalize a few common
        # forms from YAML.
        gpu = sc.gpu_devices
        if gpu is not None:
            if isinstance(gpu, str):
                low = gpu.strip().lower()
                if low == "all":
                    cmd.extend(["--gpus", "all"])
                elif low in ("", "none", "null"):
                    pass  # explicitly disabled
                else:
                    cmd.extend(["--gpus", f'"device={gpu}"'])
            elif isinstance(gpu, (list, tuple)) and gpu:
                devices = ",".join(str(g) for g in gpu)
                cmd.extend(["--gpus", f'"device={devices}"'])

        # Network mode.
        if sc.network_mode:
            cmd.extend(["--network", str(sc.network_mode)])

        # Working directory inside the container.
        cmd.extend(["-w", sc.working_dir])

        # Volumes (resolve relative host paths to absolute and create them).
        config_dir = getattr(sc, "config_dir", None)
        for host_path, container_path in (sc.volumes or {}).items():
            resolved = _resolve_host_path(host_path, config_dir=config_dir)
            try:
                Path(resolved).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self.logger.warning(
                    f"Could not create host volume path {resolved}: {e}"
                )
            cmd.extend(["-v", f"{resolved}:{container_path}"])

        # Environment variables.
        for key, value in (sc.env_vars or {}).items():
            cmd.extend(["-e", f"{key}={value}"])

        # Auto-remove: pair ``--rm`` with the runtime auto_remove flag.
        if sc.auto_remove:
            cmd.append("--rm")

        # Image and a long-lived no-op so the container stays up between
        # docker exec calls.
        cmd.extend([sc.image, "tail", "-f", "/dev/null"])

        self.logger.info(
            "Starting container: "
            + " ".join(shlex.quote(c) for c in cmd)
        )
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if r.returncode != 0:
            raise RuntimeError(
                f"Failed to start container: {(r.stderr or r.stdout).strip()}"
            )
        self._container_id = r.stdout.strip()
        self._container_name = container_name
        self._created_by_us = True
        self.logger.info(
            f"Container started: {self._container_name} "
            f"({self._container_id[:12]})"
        )

    def _initialize_container(self) -> None:
        """One-time setup inside the container after it starts.

        Ensures ``working_dir`` exists and is writable, and creates the
        per-session state directory we use for cwd persistence.
        """
        sc = self.config.session_config
        wd = shlex.quote(sc.working_dir)
        # ``|| true`` because some images have read-only or already-set perms
        # that we don't want to fail the whole setup over.
        self.docker_exec(
            f"mkdir -p {wd} && chmod -R u+rwx {wd} 2>/dev/null || true",
            timeout=30,
        )
        # Seed the main thread's state dir; per-thread dirs are created
        # lazily on first ``exec_bash_stateful`` call.
        self._ensure_thread_state()

    # ------------------------------------------------------------------ #
    # Required abstract members
    # ------------------------------------------------------------------ #

    def get_session(self) -> Any:
        """DockerEnv does not provide a Session directly; the Session class
        owns this Env instance, not the other way around."""
        raise NotImplementedError("DockerEnv does not provide session directly")

    def submit_job(self, command: str, job_type: str = "debug", **kwargs: Any) -> str:
        raise NotImplementedError("DockerEnv does not support job submission")

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        raise NotImplementedError("DockerEnv does not support job status")

    def cancel_job(self, job_id: str) -> None:
        raise NotImplementedError("DockerEnv does not support job cancellation")

    @property
    def container_id(self) -> str | None:
        """Active container id (full hash), or None if not yet started."""
        return self._container_id

    @property
    def container_name(self) -> str | None:
        return self._container_name

    # ------------------------------------------------------------------ #
    # Per-thread state for cwd persistence
    # ------------------------------------------------------------------ #

    def _state_dir_for_thread(self) -> str:
        """Path inside the container to the current thread's state dir."""
        tid = threading.get_ident()
        return f"/tmp/evomaster_session_{self._session_uid}/thread_{tid}"

    def _ensure_thread_state(self) -> str:
        """Create the per-thread state dir on first use; cache the path."""
        if not getattr(self._thread_local, "initialized", False):
            d = self._state_dir_for_thread()
            sc = self.config.session_config
            # Seed cwd with the configured working_dir so the first command
            # starts in a predictable place even if the user didn't ``cd``.
            self.docker_exec(
                f"mkdir -p {shlex.quote(d)} && "
                f"echo {shlex.quote(sc.working_dir)} > {shlex.quote(d)}/cwd",
                timeout=10,
            )
            self._thread_local.initialized = True
            self._thread_local.state_dir = d
        return self._thread_local.state_dir

    # ------------------------------------------------------------------ #
    # Command execution
    # ------------------------------------------------------------------ #

    def docker_exec(
        self,
        command: str,
        timeout: int | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Run a single shell command inside the container via ``docker exec``.

        Low-level primitive: it does *not* persist working directory or any
        other state between calls. Most callers want
        :meth:`exec_bash_stateful` instead.
        """
        if not self._container_id:
            raise RuntimeError("Container not started")

        timeout = timeout or self.config.session_config.timeout
        cmd: list[str] = ["docker", "exec"]
        if workdir:
            cmd.extend(["-w", workdir])
        for k, v in (env or {}).items():
            cmd.extend(["-e", f"{k}={v}"])
        cmd.extend([self._container_id, "bash", "-c", command])

        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return {
                "stdout": r.stdout,
                "stderr": r.stderr,
                "exit_code": r.returncode,
                "output": r.stdout + r.stderr,
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Command timed out after {timeout}s",
                "exit_code": -1,
                "output": f"Command timed out after {timeout}s",
            }

    def exec_bash_stateful(
        self,
        command: str,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Run a bash command, persisting the working directory across calls.

        The command is wrapped so that:

        1. The shell ``cd``s into the cwd recorded by the previous call (or
           the configured ``working_dir`` on first use).
        2. The user command runs in a brace block (so ``set -e`` etc. inside
           do not abort the wrapper itself).
        3. After the command, the new ``pwd`` is written back to the state
           file, and a marker line carrying the exit code and pwd is printed
           on its own stdout line so the caller can parse it back.

        Returns the standard session result dict
        (``stdout`` / ``stderr`` / ``exit_code`` / ``working_dir`` / ``output``).
        """
        if not self._container_id:
            raise RuntimeError("Container not started")

        state_dir = self._ensure_thread_state()
        cwd_file = f"{state_dir}/cwd"
        sc = self.config.session_config
        default_cwd = sc.working_dir

        # We embed the user's command verbatim in a brace block. Newlines in
        # the user command are fine — ``bash -c`` accepts a multi-line script
        # as the first arg.
        wrapper = (
            "set +e\n"
            f'__cwd="$(cat {shlex.quote(cwd_file)} 2>/dev/null)"\n'
            f'[ -z "$__cwd" ] && __cwd={shlex.quote(default_cwd)}\n'
            f'cd "$__cwd" 2>/dev/null || cd {shlex.quote(default_cwd)}\n'
            "{\n"
            f"{command}\n"
            "}\n"
            "__exit=$?\n"
            f"pwd > {shlex.quote(cwd_file)} 2>/dev/null || true\n"
            f'printf "\\n%s exit=%s pwd=%s\\n" '
            f'{shlex.quote(_EVO_MARKER)} "$__exit" "$(pwd)"\n'
            "exit $__exit\n"
        )

        result = self.docker_exec(wrapper, timeout=timeout)
        stdout = result["stdout"]
        stderr = result["stderr"]
        exit_code = result["exit_code"]
        working_dir = default_cwd

        # Strip the marker line and recover exit code + cwd from it.
        marker_idx = stdout.rfind(_EVO_MARKER)
        if marker_idx != -1:
            # Marker line goes from marker_idx to the next newline.
            tail = stdout[marker_idx:]
            marker_line, _, _ = tail.partition("\n")
            stdout_clean = stdout[:marker_idx].rstrip("\n")
            try:
                rest = marker_line[len(_EVO_MARKER):].strip()
                # Format: ``exit=<n> pwd=<dir>``
                exit_part, _, pwd_part = rest.partition(" pwd=")
                if exit_part.startswith("exit="):
                    exit_code = int(exit_part[len("exit="):])
                if pwd_part:
                    working_dir = pwd_part.strip()
            except Exception:
                # Marker present but malformed — fall through with the raw
                # docker_exec values.
                pass
        else:
            # No marker: timeout or hard-killed. Keep the stored cwd so that
            # the next call still starts somewhere sensible.
            stdout_clean = stdout

        # Combine for the ``output`` convenience field used by callers that
        # don't separate stdout/stderr.
        combined = stdout_clean
        if stderr:
            combined = (combined + "\n" + stderr) if combined else stderr

        return {
            "stdout": stdout_clean,
            "stderr": stderr,
            "exit_code": exit_code,
            "working_dir": working_dir,
            "output": combined,
        }

    # ------------------------------------------------------------------ #
    # Volume helpers
    # ------------------------------------------------------------------ #

    def host_to_container_path(self, host_path: str) -> str | None:
        """Map an absolute host path to the corresponding path inside the container.

        Walks the configured bind mounts (longest host prefix first so nested
        mounts win) and returns the container path for the first match. When
        no bind mount covers ``host_path``, returns ``None`` — the file is
        not visible inside the container and the caller must either copy it
        in or add a volume mount.
        """
        sc = self.config.session_config
        if not sc.volumes:
            return None
        try:
            hp = Path(os.path.expanduser(host_path))
            hp = hp.resolve() if hp.is_absolute() else Path(
                _resolve_host_path(str(hp), config_dir=getattr(sc, "config_dir", None))
            ).resolve()
        except Exception:
            return None
        hp_str = str(hp)
        config_dir = getattr(sc, "config_dir", None)
        # Longest host prefix first so nested mounts take precedence.
        resolved_mounts: list[tuple[str, str]] = []
        for host_decl, container_decl in (sc.volumes or {}).items():
            try:
                resolved = _resolve_host_path(host_decl, config_dir=config_dir)
            except Exception:
                continue
            resolved_mounts.append((resolved, str(container_decl).rstrip("/") or "/"))
        resolved_mounts.sort(key=lambda kv: len(kv[0]), reverse=True)
        for host_root, container_root in resolved_mounts:
            if not host_root:
                continue
            if hp_str == host_root:
                return container_root or "/"
            prefix = host_root.rstrip("/") + "/"
            if hp_str.startswith(prefix):
                rel = hp_str[len(prefix):]
                if container_root in ("", "/"):
                    return "/" + rel
                return container_root + "/" + rel
        return None

    def is_mounted_path(self, container_path: str) -> tuple[bool, str | None]:
        """Check whether ``container_path`` lives inside one of our bind mounts.

        Returns ``(is_mounted, host_path)``: when ``True``, ``host_path`` is
        the corresponding absolute path on the host, which lets file I/O
        bypass ``docker cp`` for a big speedup on large files.
        """
        sc = self.config.session_config
        if not sc.volumes:
            return False, None
        cp = str(Path(container_path).as_posix())
        if not cp.startswith("/"):
            return False, None
        config_dir = getattr(sc, "config_dir", None)
        # Walk volumes longest mount-point first so nested mounts win.
        for host_path, mount_point in sorted(
            sc.volumes.items(), key=lambda kv: len(kv[1]), reverse=True
        ):
            mp = str(Path(mount_point).as_posix()).rstrip("/")
            if not mp:
                continue
            if cp == mp:
                return True, _resolve_host_path(host_path, config_dir=config_dir)
            if cp.startswith(mp + "/"):
                rel = cp[len(mp) + 1:]
                host_root = _resolve_host_path(host_path, config_dir=config_dir)
                return True, str(Path(host_root) / rel)
        return False, None

    # ------------------------------------------------------------------ #
    # File I/O
    # ------------------------------------------------------------------ #

    def upload_file(self, local_path: str, remote_path: str) -> None:
        """Upload a file (or directory) into the container.

        If ``remote_path`` is inside a bind mount, the file is copied on the
        host directly; otherwise ``docker cp`` is used.
        """
        if not self._container_id:
            raise RuntimeError("Container not started")

        is_mounted, host_path = self.is_mounted_path(remote_path)
        if is_mounted and host_path:
            try:
                Path(host_path).parent.mkdir(parents=True, exist_ok=True)
                if Path(local_path).is_dir():
                    if Path(host_path).exists():
                        shutil.rmtree(host_path)
                    shutil.copytree(local_path, host_path)
                else:
                    shutil.copy2(local_path, host_path)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to upload {local_path} -> host path {host_path}: {e}"
                )
            return

        # docker cp path: ensure remote dir exists and is writable.
        remote_dir = str(Path(remote_path).parent)
        self.docker_exec(
            f"mkdir -p {shlex.quote(remote_dir)} && "
            f"chmod 777 {shlex.quote(remote_dir)} 2>/dev/null || true",
            timeout=30,
        )
        r = subprocess.run(
            ["docker", "cp", local_path, f"{self._container_id}:{remote_path}"],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"docker cp failed: {(r.stderr or r.stdout).strip()}"
            )
        # Best-effort permission fix so the in-container user can read.
        self.docker_exec(
            f"chmod 666 {shlex.quote(remote_path)} 2>/dev/null || true",
            timeout=10,
        )

    def download_file(self, remote_path: str, timeout: int | None = None) -> bytes:
        """Read a single file out of the container as raw bytes."""
        if not self._container_id:
            raise RuntimeError("Container not started")

        is_mounted, host_path = self.is_mounted_path(remote_path)
        if is_mounted and host_path:
            if os.path.isdir(host_path):
                raise RuntimeError(
                    f"Cannot download a directory: {remote_path}. "
                    f"Use exec_bash to inspect its contents."
                )
            try:
                with open(host_path, "rb") as f:
                    return f.read()
            except FileNotFoundError:
                raise RuntimeError(
                    f"File not found: {remote_path} (host: {host_path})"
                )

        if self.is_directory(remote_path):
            raise RuntimeError(
                f"Cannot download a directory: {remote_path}. "
                f"Use exec_bash to inspect its contents."
            )

        timeout = timeout or 60
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        try:
            r = subprocess.run(
                ["docker", "cp", f"{self._container_id}:{remote_path}", tmp_path],
                capture_output=True, text=True, timeout=timeout,
            )
            if r.returncode != 0:
                err = (r.stderr or r.stdout).strip()
                if "is a directory" in err.lower() or "cannot copy directory" in err.lower():
                    raise RuntimeError(
                        f"Cannot download a directory: {remote_path}."
                    )
                raise RuntimeError(f"docker cp failed: {err}")
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def read_file_content(self, remote_path: str, encoding: str = "utf-8") -> str:
        """Convenience: read a text file out of the container."""
        is_mounted, host_path = self.is_mounted_path(remote_path)
        if is_mounted and host_path:
            try:
                with open(host_path, "r", encoding=encoding) as f:
                    return f.read()
            except FileNotFoundError:
                raise RuntimeError(
                    f"File not found: {remote_path} (host: {host_path})"
                )
        return self.download_file(remote_path).decode(encoding)

    def write_file_content(self, remote_path: str, content: str, encoding: str = "utf-8") -> None:
        """Convenience: write text content into a file inside the container."""
        is_mounted, host_path = self.is_mounted_path(remote_path)
        if is_mounted and host_path:
            try:
                Path(host_path).parent.mkdir(parents=True, exist_ok=True)
                with open(host_path, "w", encoding=encoding) as f:
                    f.write(content)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to write file {remote_path} (host: {host_path}): {e}"
                )
            return

        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp:
            tmp.write(content.encode(encoding))
            tmp_path = tmp.name
        try:
            self.upload_file(tmp_path, remote_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # ------------------------------------------------------------------ #
    # Path predicates
    # ------------------------------------------------------------------ #

    def path_exists(self, remote_path: str) -> bool:
        is_mounted, host_path = self.is_mounted_path(remote_path)
        if is_mounted and host_path:
            return os.path.exists(host_path)
        r = self.docker_exec(
            f'test -e {shlex.quote(remote_path)} && echo exists || echo no',
            timeout=10,
        )
        return r["stdout"].strip() == "exists"

    def is_file(self, remote_path: str) -> bool:
        is_mounted, host_path = self.is_mounted_path(remote_path)
        if is_mounted and host_path:
            return os.path.isfile(host_path)
        r = self.docker_exec(
            f'test -f {shlex.quote(remote_path)} && echo file || echo no',
            timeout=10,
        )
        return r["stdout"].strip() == "file"

    def is_directory(self, remote_path: str) -> bool:
        is_mounted, host_path = self.is_mounted_path(remote_path)
        if is_mounted and host_path:
            return os.path.isdir(host_path)
        r = self.docker_exec(
            f'test -d {shlex.quote(remote_path)} && echo dir || echo no',
            timeout=10,
        )
        return r["stdout"].strip() == "dir"
