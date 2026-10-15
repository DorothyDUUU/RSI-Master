"""Single-agent rollout orchestrator for cc_rollout.

一个 CC 一路 Rollout 到底 (docs/README.md): 没有 branches / director / eval_state /
handoff —— ``run()`` 只做: 读 config → 建 ``runs/<run_id>_<timestamp>/`` →
config 快照 → deadline watchdog (硬性超时兜底) → ``seed.seed_workspace`` (两种
mode) → ``runner.run_agent`` 跑一个 ``claude -p`` → 写 ``summary.md``。

退出状态 (继承 cc_branching runner):
- agent 在 workspace 里 ``echo <experiment_id> > FINISH`` → ``done_marker``;
- ``claude -p`` 进程自然退出 → ``completed``;
- ``claude -p`` 非零退出 → ``failed``;
- ``single_agent_running_time_seconds`` 预算耗尽 → ``timeout``;
- ``max_running_time_seconds`` 全局 deadline 触发 → ``cancelled``。
非 completed 一律 ``_terminate_tree`` (SIGTERM → KILL_GRACE_SEC → SIGKILL)。

**提前退出不算收尾, 会被 resume/restart。** ``completed`` (无 FINISH 的文本回合退出)
和 ``failed`` (异常中断) 若发生在 run window 还没到点、预算还有剩余时, 说明 agent
提前停了 —— orchestrator 先 ``claude --resume <session_id>`` 续**同一条会话** (保留
上下文, 即 cc_branching runner 的 resume 机制), resume 失败或再次提前退出才在同一
workspace fallback 到**全新干净会话** (读盘恢复)。``max_agent_restarts`` 限制全新
会话的 fallback 次数。只有 FINISH / 窗口耗尽 (timeout) / 全局 deadline (cancelled)
才是真正的收尾。

cc_with_expos 模式的 submit 训练管线在 agent 侧通过 ``./expos worker submit``
触发 (async → gpu_backend :8989), 本 orchestrator 不阻塞等待 GPU 结果 ——
score 出现在 workspace/experiment_os/experiments/<id>/result.json 里,
summary 只记录 agent 的 outcome + 指针。
"""

from __future__ import annotations

import datetime
import logging
import shutil
import threading
import time
import uuid
from pathlib import Path

import yaml

from . import runner, seed
from ._paths import runs_root

logger = logging.getLogger(__name__)

DEFAULT_MAX_AGENT_RESTARTS = 2
DEFAULT_AGENT_RESTART_BACKOFF_SEC = 15.0

# The only terminal ends of a run: the agent wrote FINISH (done_marker, and ckpt
# is set whenever FINISH has content), the per-attempt budget was spent (timeout
# == the run window itself on a single long attempt), or the global deadline hit
# (cancelled). Everything else with budget left is a *premature* stop to resume.
_TERMINAL_REASONS = ("done_marker", "timeout", "cancelled")


def run(config_path: str | Path, run_name: str | None = None) -> Path:
    """Run one cc_rollout: seed workspace → one CC agent → summary. Returns run root.

    *config_path* is the (materialized) rollout config; *run_name* overrides
    ``run.run_id``.  ``config.mode`` selects cc_naive vs cc_with_expos.
    """
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text())

    run_cfg = config.get("run", {}) or {}
    mode = str(config.get("mode", "") or seed.MODE_NAIVE).strip()
    if mode not in (seed.MODE_NAIVE, seed.MODE_WITH_EXPOS):
        raise ValueError(
            f"cc_rollout config mode must be {seed.MODE_NAIVE!r} or "
            f"{seed.MODE_WITH_EXPOS!r}, got {mode!r}"
        )

    # Global run deadline (run.max_running_time_seconds): absent/empty = unlimited.
    # A daemon watchdog flips stop_event at the deadline; the running agent polls
    # it inside run_agent and tears down gracefully (stop_reason=cancelled).
    max_run_raw = run_cfg.get("max_running_time_seconds")
    max_run = float(max_run_raw) if max_run_raw else None
    budget = float(run_cfg.get("single_agent_running_time_seconds", 120))

    # Run root = project-root runs/, one folder per launch:
    #   runs/<run_id>_<timestamp>/   — run_id from run.run_id (or --name); the
    #   timestamp keeps re-launches of the same run_id from colliding.
    run_id = run_name or run_cfg.get("run_id") or run_cfg.get("name") \
        or uuid.uuid4().hex[:8]
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_root = runs_root() / f"{run_id}_{stamp}"
    run_root.mkdir(parents=True, exist_ok=False)

    # Snapshot the input config into the run root.
    shutil.copy(config_path, run_root / "config.yaml")

    # Run-level deadline watchdog.
    stop_event = threading.Event()
    if max_run is not None:
        def _deadline_watchdog() -> None:
            time.sleep(max_run)
            logger.info(
                "=== max_running_time_seconds reached (%.0fs) — graceful stop ===",
                max_run,
            )
            stop_event.set()

        threading.Thread(target=_deadline_watchdog, name="run-deadline", daemon=True).start()
        logger.info("run deadline armed: %.0fs (agent budget %.0fs)", max_run, budget)

    # Seed the workspace (backend copy + README/BENCHMARK/GLOBAL_MEMORY; the
    # with_expos branch additionally seeds experiment_os + SUBMIT_GUIDE + ./expos).
    session_id = str(uuid.uuid4())
    workspace = seed.seed_workspace(run_root, config, mode, session_id)

    # Init prompt: run.task (injected by the adapter from run.py --task) or the
    # composed seed task (run params + BENCHMARK.md).  One CC, no branching.
    task = run_cfg.get("task") or seed.build_seed_task(config, mode=mode)
    agent_started = time.monotonic()
    outcome = runner.run_agent(
        workspace, task, config,
        session_id=session_id, resume=False,
        stop_event=stop_event,
    )
    attempts = [outcome]

    max_restarts = max(0, int(run_cfg.get(
        "max_agent_restarts", DEFAULT_MAX_AGENT_RESTARTS,
    )))
    backoff = max(0.0, float(run_cfg.get(
        "agent_restart_backoff_seconds", DEFAULT_AGENT_RESTART_BACKOFF_SEC,
    )))

    # An attempt that ends while the window still has budget is NOT the run
    # ending. Two premature ways it can end:
    #   * "completed" — the headless claude exited rc=0 without writing FINISH
    #     (a text-only, no-tool turn reads as "task complete" and exits);
    #   * "failed"    — abnormal interruption (crash / API error / pod death).
    # A restart first RESUMES the same session (`claude --resume <session_id>`
    # reloads the transcript, so the agent's context survives — the resume
    # mechanism claude itself exposes, same wiring as the cc_branching runner)
    # and only falls back to a brand-new clean session when the resume cannot
    # work. `max_agent_restarts` caps the fresh-session fallbacks; every extra
    # launch runs under the remaining budget (single_agent_running_time_seconds).
    fresh_fallbacks = 0
    for restart_index in range(1, max_restarts + 1):
        if outcome.ckpt or outcome.stop_reason in _TERMINAL_REASONS \
                or stop_event.is_set():
            break  # submitted FINISH, or the window / global deadline is spent.

        remaining = budget - (time.monotonic() - agent_started)
        if remaining <= backoff:
            logger.warning(
                "agent %s (kind=%s) stopped before the window but no budget "
                "remains for restart %d/%d",
                outcome.stop_reason, outcome.failure_kind or "",
                restart_index, max_restarts,
            )
            break
        if stop_event.wait(min(backoff, remaining)):
            break
        remaining = budget - (time.monotonic() - agent_started)
        if remaining <= 0:
            break
        retry_config = _with_agent_budget(config, remaining)

        # Resuming a transcript that already tripped the empty-assistant 400 just
        # replays the poisoned turn and fails again — for that one failure go
        # straight to a fresh clean session (its known-good recovery). Every other
        # premature end resumes in place first, keeping the agent's context.
        resume_first = not (
            outcome.stop_reason == "failed"
            and outcome.failure_kind == "empty_assistant_message"
        )
        if resume_first:
            logger.warning(
                "agent %s (kind=%s) ended with %.0fs budget left; resuming the "
                "same session (restart %d/%d)",
                outcome.stop_reason, outcome.failure_kind or "", remaining,
                restart_index, max_restarts,
            )
            outcome = runner.run_agent(
                workspace,
                _resume_prompt(task, outcome, remaining),
                retry_config,
                session_id=outcome.session_id,
                resume=True,
                log_name=f"agent.resume_{restart_index}.log",
                stop_event=stop_event,
            )
            attempts.append(outcome)
            if outcome.ckpt or outcome.stop_reason in _TERMINAL_REASONS:
                break  # the resumed session finished the run.
            if stop_event.is_set():
                break
            remaining = budget - (time.monotonic() - agent_started)
            if remaining <= backoff:
                logger.warning(
                    "resumed agent %s before the window again; too little budget "
                    "left for a fresh-session fallback",
                    outcome.stop_reason,
                )
                break
            if stop_event.wait(min(backoff, remaining)):
                break
            remaining = budget - (time.monotonic() - agent_started)
            if remaining <= 0:
                break

        # The resume did not finish the run (it crashed again, stopped again, or
        # was skipped for a poisoned transcript) — fall back to a brand-new clean
        # session that recovers the run's state from disk.
        fresh_fallbacks += 1
        retry_config = _with_agent_budget(config, remaining)
        retry_session_id = str(uuid.uuid4())
        logger.warning(
            "resume %s rc=%s kind=%s; fresh clean-session fallback %d/%d in the "
            "same workspace (remaining budget %.0fs)",
            outcome.stop_reason, outcome.returncode, outcome.failure_kind or "",
            fresh_fallbacks, max_restarts, remaining,
        )
        outcome = runner.run_agent(
            workspace,
            _recovery_prompt(task, outcome),
            retry_config,
            session_id=retry_session_id,
            resume=False,
            log_name=f"agent.restart_{fresh_fallbacks}.log",
            stop_event=stop_event,
        )
        attempts.append(outcome)

    _write_summary(run_root, config, outcome, mode, attempts)
    logger.info(
        "done (mode=%s, %s, %d attempt(s), %.0fs agent time). run root: %s",
        mode, outcome.stop_reason, len(attempts),
        sum(item.elapsed_sec for item in attempts), run_root,
    )
    if outcome.stop_reason == "failed":
        raise RuntimeError(
            f"cc_rollout agent failed after {len(attempts)} attempt(s); "
            f"see {run_root / 'summary.md'}"
        )
    return run_root


def _with_agent_budget(config: dict, remaining_seconds: float) -> dict:
    """Return a shallow config copy whose next agent shares the original budget."""
    copied = dict(config)
    copied["run"] = dict(config.get("run", {}) or {})
    copied["run"]["single_agent_running_time_seconds"] = remaining_seconds
    return copied


def _recovery_prompt(original_task: str, previous: runner.AgentOutcome) -> str:
    """Prompt a FRESH session (no transcript) to recover state from disk."""
    return f"""The previous control-agent process ended before the run window was
spent (stop reason {previous.stop_reason}, return code {previous.returncode},
failure kind {previous.failure_kind or 'unknown'}). Its transcript is being
discarded — you are a brand-new session, so recover state from disk, not from
memory. Continue the original task in this SAME workspace. Do not restart
completed work.

First read README.md and BENCHMARK.md, then read AGENT.md to pick up where the
last session left off, and inspect existing datasets, experiments, GPU script IDs,
logs, and results. Reconcile any jobs that are still running before submitting
replacements.

The run ends only when the wall-clock window is spent or you write FINISH near the
end of it. If time remains you are not done — do not stop early, and never end a
turn in plain prose (this session is headless: a text-only turn with no tool call
reads as "task complete" and ends the run). Always end your turn with a concrete
tool call.

Original objective:
{original_task}
"""


def _resume_prompt(
    original_task: str,
    previous: runner.AgentOutcome,
    remaining: float,
) -> str:
    """Prompt a RESUMED session (same transcript, context intact) to keep going."""
    return f"""Your previous process ended before the run window was spent (stop
reason {previous.stop_reason}, return code {previous.returncode}, failure kind
{previous.failure_kind or 'none'}). About {remaining:.0f}s of run budget remain.

This is a RESUME: your full conversation and working context are intact — do not
restart or re-derive completed work and do not re-read everything from scratch.
Pick up exactly where you left off and keep the research arc moving.

The run ends only when the wall-clock window is spent (or you write FINISH after a
complete analysis near the end of it). If time remains, you are not done: keep
working, do not stop early, and do not end this turn in plain prose — this session
is headless, a text-only turn reads as "task complete" and ends the run. Always end
your turn with a concrete tool call.

You were working toward:
{original_task}
"""


def _write_summary(
    run_root: Path,
    config: dict,
    outcome: runner.AgentOutcome,
    mode: str,
    attempts: list[runner.AgentOutcome],
) -> Path:
    """Write ``summary.md`` — agent outcome + pointers. No GPU eval here: the
    pipeline's score lands async in the workspace (async submit), so the summary
    records where to look, not a score."""
    text = [
        f"# cc_rollout Summary — {run_root.name}",
        "",
        f"- mode: `{mode}`",
        f"- task: {config.get('run', {}).get('task', '')}",
        f"- stop_reason: `{outcome.stop_reason}`",
        f"- returncode: `{outcome.returncode}`",
        f"- attempt_count: {len(attempts)}",
        f"- total_agent_elapsed_sec: {sum(item.elapsed_sec for item in attempts):.0f}",
        f"- final_attempt_elapsed_sec: {outcome.elapsed_sec:.0f}",
        f"- timed_out: `{outcome.timed_out}`",
        f"- FINISH content (submitted experiment): `{outcome.ckpt or ''}`",
        f"- session_id: `{outcome.session_id}`",
        f"- agent log: `{outcome.log_path}`",
        "",
        "## agent attempts",
        "",
        "| attempt | stop reason | return code | failure kind | elapsed sec | log |",
        "|---:|---|---:|---|---:|---|",
        *[
            f"| {index} | {item.stop_reason} | {item.returncode} | "
            f"{item.failure_kind or ''} | {item.elapsed_sec:.0f} | `{item.log_path}` |"
            for index, item in enumerate(attempts, 1)
        ],
        "",
        "## where to look next",
        "- agent trajectory / log: `logs/agent.log`",
        "- workspace: `workspace/` (README.md protocol, BENCHMARK.md, "
        "GLOBAL_MEMORY.md, public/ backend modules)",
        "- cc_with_expos: `workspace/experiment_os/experiments/<id>/result.json` "
        "holds the pipeline's final score (async); state graph in "
        "`workspace/experiment_os/state/state.json`",
        "",
    ]
    path = run_root / "summary.md"
    path.write_text("\n".join(text))
    return path


__all__ = ["run"]
