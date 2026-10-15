"""Single Claude-Code agent rollout inside a private workspace.

移植自 ``playground/cc_branching/core/rollout/runner.py``。唯一结构性改动:
cc_rollout 是单 agent 布局, ``configs/`` 与 ``logs/`` 是 workspace 的平级目录
(见 ``layout``), 所以 ``agent.log`` 写到 ``logs/agent.log`` 而不是 ``configs/``。

Spawns ``claude -p`` with the workspace as cwd, its own ``CLAUDE_CONFIG_DIR``
(so the session transcript lives in the run's configs/ dir), a time budget, and
a ``FINISH`` early-stop marker the agent can drop after submitting a ckpt.
Output is teed to ``logs/agent.log`` (no trajectory capture).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .layout import config_dir_for, logs_dir_for

logger = logging.getLogger(__name__)

DONE_MARKER = "FINISH"
KILL_GRACE_SEC = 15.0
DONE_CONTENT_GRACE_SEC = 3.0  # wait for `echo <experiment_id> > FINISH` content to land


@dataclass
class AgentOutcome:
    stop_reason: str  # "completed" | "failed" | "timeout" | "done_marker" | "cancelled"
    returncode: int | None
    elapsed_sec: float
    log_path: Path
    session_id: str
    ckpt: str | None = None      # submitted ckpt read from FINISH content
    timed_out: bool = field(default=False)
    done_marker: bool = field(default=False)
    failure_kind: str | None = None


def resolve_claude(bin_name: str = "claude") -> str:
    path = shutil.which(bin_name)
    if not path:
        raise FileNotFoundError(
            f"claude CLI ('{bin_name}') not found on PATH — required for rollout agents"
        )
    return path


def build_command(
    claude_bin: str,
    run_dir: Path,
    public_dirs: list[str],
    settings_path: Path,
    permission_mode: str,
    *,
    session_id: str,
    resume: bool,
    system_prompt: str | None,
    prompt: str,
) -> list[str]:
    """Construct the ``claude -p`` command for a seed or a resumed session.

    A fresh seed pins the session id (``--session-id``) and appends the workspace
    protocol as the system prompt. A resume drops both and continues the prior
    transcript via ``--resume <session_id>`` — same workspace + same
    CLAUDE_CONFIG_DIR, so claude finds the seed session's jsonl there (the resume
    mechanism cc itself exposes, same wiring as the cc_branching runner).
    """
    cmd = [claude_bin, "-p"]

    if resume:
        cmd += ["--resume", session_id]
    else:
        # Fresh seed: pin the session id + append the workspace protocol as system prompt.
        cmd += ["--session-id", session_id, "--append-system-prompt", system_prompt or ""]

    if permission_mode == "bypassPermissions":
        # Equivalent per CLI docs, but the conventional form for headless
        # unattended runs.  As root it additionally needs IS_SANDBOX=1 (set in
        # run_agent) or claude refuses to start.
        cmd += ["--dangerously-skip-permissions"]
    else:
        cmd += ["--permission-mode", permission_mode]
    cmd += ["--settings", str(settings_path)]
    if public_dirs:
        # NOTE: --add-dir uses nargs='+' and would greedily consume a trailing
        # positional prompt, so the prompt is always passed via stdin instead.
        cmd += ["--add-dir", *public_dirs]
    return cmd


def run_agent(
    run_dir: Path,
    prompt: str,
    config: dict,
    *,
    session_id: str | None = None,
    resume: bool = False,
    log_name: str = "agent.log",
    stop_event: threading.Event | None = None,
) -> AgentOutcome:
    """Run one headless agent in *run_dir*; stop on budget, DONE marker, exit,
    or a global *stop_event* (graceful run-level cancellation).

    *run_dir* is the workspace; it must already contain ``README.md``.  Its
    sibling ``configs/`` must already contain ``settings.json`` (see
    ``permissions.write_settings``); ``logs/`` is created lazily.
    """
    run_dir = Path(run_dir).resolve()
    claude_bin = resolve_claude(config.get("claude", {}).get("bin", "claude"))
    public_dirs = config.get("run", {}).get("public_dirs", [])
    permission_mode = config.get("claude", {}).get("permission_mode", "acceptEdits")
    budget = float(config.get("run", {}).get("single_agent_running_time_seconds", 120))

    session_id = session_id or str(uuid.uuid4())
    # System state lives in the claude-INVISIBLE sibling configs/ dir — never in
    # the workspace the agent works in.  CLAUDE_CONFIG_DIR points there so the
    # transcript is stored out of the agent's reach.
    config_dir = config_dir_for(run_dir)
    claude_config = config_dir / ".claude-config"
    claude_config.mkdir(parents=True, exist_ok=True)

    readme = run_dir / "README.md"
    system_prompt = readme.read_text() if readme.exists() else ""

    settings_path = config_dir / "settings.json"
    if not settings_path.exists():
        raise FileNotFoundError(f"missing {settings_path} — run permissions.write_settings first")

    cmd = build_command(
        claude_bin, run_dir, public_dirs, settings_path, permission_mode,
        session_id=session_id, resume=resume,
        system_prompt=system_prompt, prompt=prompt,
    )

    log_dir = logs_dir_for(run_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_name
    env = dict(os.environ)
    env["CLAUDE_CONFIG_DIR"] = str(claude_config)
    if permission_mode == "bypassPermissions":
        # IS_SANDBOX=1 relaxes the root/sudo startup gate for
        # --dangerously-skip-permissions (undocumented, but required to run the
        # bypass flag as root — which this harness does in containers).
        env.setdefault("IS_SANDBOX", "1")
    # Sidecar LLM backend (optional): let config point claude at a custom
    # Anthropic-Messages-compatible endpoint instead of the default API.
    cc = config.get("claude", {})
    if cc.get("base_url"):
        env["ANTHROPIC_BASE_URL"] = cc["base_url"]
    if cc.get("auth_token"):
        env["ANTHROPIC_AUTH_TOKEN"] = cc["auth_token"]
    if cc.get("model"):
        env["ANTHROPIC_MODEL"] = cc["model"]
        # A single-model router serves every Claude tier; map all tiers to the
        # configured model so session-title / sonnet / opus / haiku requests hit
        # a served model id instead of a hardcoded unknown one (404/unknown).
        env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = cc.get("default_sonnet_model") or cc["model"]
        env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = cc.get("default_opus_model") or cc["model"]
        env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = cc.get("default_haiku_model") or cc["model"]
    env.setdefault("CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT", "1")

    logger.info("agent %s resume=%s budget=%.0fs cwd=%s", session_id, resume, budget, run_dir)
    logger.debug("cmd: %s", " ".join(cmd))

    # Own session (start_new_session) so a killpg can tear down the whole tree.
    # The prompt is fed via stdin because --add-dir would swallow a trailing
    # positional prompt (nargs='+').
    proc = subprocess.Popen(
        cmd,
        cwd=str(run_dir),
        env=env,
        stdout=open(log_path, "wb"),
        stderr=subprocess.STDOUT,
        stdin=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        proc.stdin.write(prompt.encode())
        proc.stdin.close()
    except BrokenPipeError:
        # The process exited before reading stdin — fine, poll() will surface it.
        pass

    started = time.time()
    stop_reason = "completed"
    done_marker_seen = False
    timeout_hit = False

    while True:
        # Graceful run-level cancellation (global max_running_time_seconds hit).
        if stop_event is not None and stop_event.is_set():
            stop_reason = "cancelled"
            break
        rc = proc.poll()
        if rc is not None:
            if rc != 0:
                stop_reason = "failed"
            break
        # Early stop: agent dropped FINISH after submitting an experiment.
        # The experiment_id IS the marker's content (`echo exp_0005 > FINISH`).
        if (run_dir / DONE_MARKER).exists():
            done_marker_seen = True
            stop_reason = "done_marker"
            # small grace so the `echo` redirection's content is fully written
            for _ in range(int(DONE_CONTENT_GRACE_SEC)):
                if (run_dir / DONE_MARKER).read_text().strip():
                    break
                time.sleep(1.0)
            break
        if time.time() - started > budget:
            timeout_hit = True
            stop_reason = "timeout"
            break
        time.sleep(1.0)

    elapsed = time.time() - started
    if stop_reason != "completed":
        _terminate_tree(proc)
        rc = proc.returncode

    logger.info(
        "agent %s %s rc=%s elapsed=%.0fs", session_id, stop_reason, rc, elapsed,
    )
    failure_kind = _classify_failure(log_path, rc) if stop_reason == "failed" else None
    return AgentOutcome(
        stop_reason=stop_reason,
        returncode=rc,
        elapsed_sec=elapsed,
        log_path=log_path,
        session_id=session_id,
        ckpt=_read_ckpt(run_dir),
        timed_out=timeout_hit,
        done_marker=done_marker_seen,
        failure_kind=failure_kind,
    )


def _classify_failure(log_path: Path, returncode: int | None) -> str | None:
    """Classify the small set of failures that need special recovery context."""
    if returncode in (None, 0):
        return None
    try:
        with log_path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 64 * 1024))
            tail = handle.read().decode(errors="replace")
    except OSError:
        return "agent_process_exit"
    if "role 'assistant' must not be empty" in tail:
        return "empty_assistant_message"
    return "agent_process_exit"


def _read_ckpt(run_dir: Path) -> str | None:
    """The submitted ckpt = FINISH content; None when empty / absent."""
    p = run_dir / DONE_MARKER
    if p.exists():
        v = p.read_text(errors="replace").strip()
        return v or None
    return None


def _terminate_tree(proc: subprocess.Popen) -> None:
    """SIGTERM then SIGKILL the whole process group."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), subprocess.signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=KILL_GRACE_SEC)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(os.getpgid(proc.pid), subprocess.signal.SIGKILL)
    except ProcessLookupError:
        pass
