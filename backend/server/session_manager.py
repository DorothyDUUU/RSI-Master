"""Tmux-based shell session manager.

Every shell session is backed by a tmux session named ``gpu_shell_<hash>``.
Background scripts get their own tmux session ``gpu_backend_<hash>`` with
stdout/stderr redirected to a log file so the caller can tail progress.
"""

from __future__ import annotations

import functools
import logging
import os
import shlex
import signal
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .gpu_monitor import GPUMonitor
from .models import (
    ScriptInfo,
    ScriptLog,
    ScriptPriority,
    ScriptStatus,
    ScriptSubmitRequest,
    ScriptSubmitResponse,
    SessionExecRequest,
    SessionExecResponse,
    SessionStatus,
    ShellSession,
)
from .subproc import HUNG, run_hard

logger = logging.getLogger(__name__)

# Prefixes used to distinguish session types in ``tmux list-sessions``
SHELL_SESSION_PREFIX = "gpu_shell_"
BACKEND_SESSION_PREFIX = "gpu_backend_"

AUTO_BACKGROUND_SECONDS = 300  # 5 minutes
NON_EXCLUSIVE_LEASE_SECONDS = 1200  # 20 minutes — 非独占任务对分配卡的独占租约时长
LOG_DIR = Path("/tmp/gpu_backend_logs")
# Hard wall clock for any tmux subprocess: a wedged NFS/mount can put the tmux
# client in D-state, and ``subprocess.run(timeout=...)`` cannot kill D-state
# children.  ``run_hard`` abandons the child after this and degrades instead.
TMUX_WALL_MS = 15000
# Session-set sampler cadence — how fresh ``_alive_names`` is kept so
# list_sessions / get_script_status never spawn a tmux subprocess under lock.
TMUX_SAMPLE_INTERVAL = 5.0
# 会话缺失终结前需要连续 N 个「采样器健康」样本确认 (2026-08-31):
# 单次采样抖动 (tmux 偶发没列全 / 瞬时空) 不释放 booking; 健康采样连续 N 次都
# 看不到该会话才按「真死了」终结。代价是真实死亡最多延迟 ~N*TMUX_SAMPLE_INTERVAL
# 释放 booking —— 远小于误放行导致的节点 OOM。
SESSION_LOSS_CONFIRM_COUNT = 2
# 采样器健康窗口: 最近一次成功采样距今超过该值视为「stale」(线程死/一直失败)。
SAMPLER_STALE_WINDOW = 3 * TMUX_SAMPLE_INTERVAL

# 判定/审计日志落盘 (2026-08-31): 每个进程每个 log_dir 只挂一个 FileHandler。
# log_dir 由 GPU_BACKEND_LOG_DIR/<--log-dir> 路由到 run-dir (NFS 持久), 即使
# server stdout 丢失, SCAN state / RELEASE / 降级检测 / 终结判定都追加写到
# <log_dir>/scheduler.log。
_AUDIT_FILE_HANDLERS: dict[str, logging.FileHandler] = {}


def _elapsed_seconds(info: "ScriptInfo", now_aware: datetime) -> float:
    """Seconds from a script's start (or creation) to *now_aware*.

    测试/回放可能用 naive datetime 构造 ScriptInfo —— 与 aware 的 now 相减会抛
    TypeError, 这里统一把 naive ref 按 UTC 补时区。
    """
    ref = info.started_at or info.created_at
    if ref is None:
        return 0.0
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    return (now_aware - ref).total_seconds()


def _short_id() -> str:
    return uuid.uuid4().hex[:8]


def _guarded(method):
    """Serialize *method* under the SessionManager state lock.

    The server now runs blocking endpoints in FastAPI's threadpool and the VRAM
    scanner on its own thread, so concurrent callers can mutate the same
    script/queue/session registries.  ``threading.RLock`` is reentrant — nested
    calls from the scanner (sweep → status) are safe.  Polling loops that must
    NOT hold the lock (``exec_in_session``, ``_scanner_loop``) wrap only their
    mutation section instead of using this decorator.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._state_lock:
            return method(self, *args, **kwargs)
    return wrapper


def _run_tmux(*args: str, timeout: int = 10, check: bool = True) -> subprocess.CompletedProcess:
    """Thin wrapper around tmux with a D-state-hard wall clock.

    ``subprocess.run(timeout=...)`` cannot kill a D-state child (wedged NFS /
    mount), so it can block the caller — and any lock it holds — for minutes.
    ``run_hard`` abandons the child after ``TMUX_WALL_MS`` and returns the
    ``HUNG`` sentinel; a hung tmux degrades (returncode -1, empty output)
    instead of raising, so every caller keeps serving.
    """
    cmd = ["tmux", *args]
    logger.debug("tmux: %s", shlex.join(cmd))
    res = run_hard(cmd, timeout=timeout, wall_ms=TMUX_WALL_MS)
    if res is HUNG:
        logger.warning("tmux %s hung past %dms wall (D-state?) — degrading",
                       args[0] if args else "?", TMUX_WALL_MS)
        return HUNG
    if check and res.returncode != 0:
        raise subprocess.CalledProcessError(
            res.returncode, cmd, output=res.stdout, stderr=res.stderr
        )
    return res


class SessionManager:
    """Manages tmux-backed shell sessions and script submissions."""

    def __init__(self, log_dir: str | None = None) -> None:
        self._scripts: dict[str, ScriptInfo] = {}
        self._sessions: dict[str, ShellSession] = {}  # in-memory session registry
        self._log_dir = Path(log_dir or LOG_DIR)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._state_lock = threading.RLock()
        self._gpu_monitor = GPUMonitor()
        # Queue: list of script_ids waiting for resources (sorted by priority)
        self._queue: list[str] = []
        # Limits apply to QUEUED work, not running pipelines. No hidden GPU work.
        self._max_queued_per_owner = max(0, int(os.environ.get("GPU_BACKEND_MAX_QUEUED_PER_OWNER", "2")))
        self._max_backfills = max(0, int(os.environ.get("GPU_BACKEND_MAX_BACKFILLS", "4")))
        self._owner_last_dispatch: dict[str, int] = {}
        self._dispatch_sequence = 0
        self._backfills: dict[str, int] = {}
        # Scanner thread (started by the server lifespan via start_scanner).
        self._scanner_thread: threading.Thread | None = None
        # Sampled set of live tmux session names — refreshed on a background
        # thread (start_tmux_sampler) so lock-taking callers (list_sessions,
        # get_script_status, _session_exists) never spawn a tmux subprocess.
        # GIL-atomic frozenset swap; seeded fresh by the sampler's first tick.
        self._alive_names: frozenset[str] = frozenset()
        self._tmux_sampler_thread: threading.Thread | None = None
        # 采样器健康状态 (2026-08-31 降级防护): 最近一次采样尝试/成功时间戳 +
        # 降级标志。HUNG/rc!=0 时保持上一份 alive 缓存、标记降级 —— 判定逻辑靠
        # 它区分「确认空」和「不知道」。全部状态必须在锁外读 (GIL-atomic 赋值)。
        self._last_sample_ts: float | None = None
        self._last_sample_ok_ts: float | None = None
        self._sampler_degraded: bool = False
        # script_id → 连续「采样器健康且会话缺失」的次数 (达到 SESSION_LOSS_CONFIRM_COUNT 才终结)
        self._session_missing: dict[str, int] = {}
        # 判定/审计日志落盘 (scheduler.log, 见 _AUDIT_FILE_HANDLERS 注释)
        self._ensure_audit_file_handler(self._log_dir)

    def _ensure_audit_file_handler(self, log_dir: Path) -> None:
        """把本模块 logger 的判定/审计日志追加写到 ``<log_dir>/scheduler.log``。

        判断逻辑的日志必须落盘 (2026-08-31): SCAN state / RELEASE / 降级检测 /
        终结判定如果只走 server stdout, stdout 一丢审计就没了。log_dir 由
        ``GPU_BACKEND_LOG_DIR``/``--log-dir`` 路由到 run-dir (NFS 持久, 随 run 归档),
        这里把 INFO+ 级判定日志固化为文件。同一进程同一 log_dir 只挂一个 handler,
        多 SessionManager 实例不重复写。
        """
        key = str(log_dir)
        if key in _AUDIT_FILE_HANDLERS:
            return
        handler = logging.FileHandler(str(log_dir / "scheduler.log"), encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        # 确保模块 logger 至少 INFO, 否则根 logger 默认 WARNING 时判定日志根本到不了 handler。
        if logger.level > logging.INFO or logger.level == logging.NOTSET:
            logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        _AUDIT_FILE_HANDLERS[key] = handler
        logger.info("audit log enabled: %s (scheduler.log)", key)

    def _sampler_healthy(self) -> bool:
        """采样器是否健康可信: 未降级 && 最近成功采样距今未超时。

        False 时 get_script_status 不得凭「会话缺失」终结脚本 —— 因为这时
        _alive_names 可能是降级保留下来的旧缓存或从未采到过, 不代表真实状态。
        判定必须只看权威信号 (release 标记 / EXIT_CODE)。
        """
        if self._sampler_degraded:
            return False
        if self._last_sample_ok_ts is None:
            return False  # 还没成功采到过 (含测试构造、启动首 5s)
        return (time.time() - self._last_sample_ok_ts) <= SAMPLER_STALE_WINDOW

    # ── tmux session-set sampler ────────────────────────────────────────────

    def start_tmux_sampler(self) -> None:
        """Start the tmux session-set sampler thread (idempotent).

        Called by the server lifespan alongside ``start_scanner``.  Takes one
        synchronous sample first so sessions created during boot are visible
        immediately (a stale empty set would look like every RUNNING script had
        died).  The loop keeps ``_alive_names`` fresh; a wedged tmux degrades to
        an empty set instead of blocking anyone.
        """
        if self._tmux_sampler_thread is not None and self._tmux_sampler_thread.is_alive():
            return
        self._refresh_alive_names()
        self._tmux_sampler_thread = threading.Thread(
            target=self._tmux_sampler_loop, name="tmux-session-sampler", daemon=True
        )
        self._tmux_sampler_thread.start()
        logger.info("tmux session sampler thread started")

    def _tmux_sampler_loop(self) -> None:
        while True:
            time.sleep(TMUX_SAMPLE_INTERVAL)
            self._refresh_alive_names()

    def _refresh_alive_names(self) -> frozenset[str]:
        """One bounded ``tmux list-sessions``; updates ``_alive_names`` + health.

        降级语义 (2026-08-31 复盘, HLE 0830 OOM 根因): tmux 卡死/NFS D-state 时
        ``run_hard`` 返回 HUNG (rc=-1, 空输出) —— 旧实现把 HUNG 和「真的一无所有」
        都折叠成空集, 让所有 RUNNING 脚本「看起来全死了」→ 误终结 + booking 全释放
        → 新实验上卡 → 节点 OOM。

        现在:
        - 采样成功 (rc==0) 才替换 ``_alive_names``, 并更新 ``_last_sample_ok_ts``;
        - 采样失败/卡死 (HUNG / rc!=0) → 标记 ``_sampler_degraded``, **保持上一份
          缓存不动** (绝不置空), 判定逻辑见 ``_sampler_healthy`` / ``get_script_status``;
        - 每次成功样本的会话名按 DEBUG 落盘, 数量/健康随 SCAN state 落 INFO。
        """
        res = _run_tmux("list-sessions", "-F", "#{session_name}", check=False)
        self._last_sample_ts = time.time()
        if res is HUNG or res.returncode != 0:
            self._sampler_degraded = True
            logger.warning(
                "tmux sampler DEGRADED (rc=%s) at %s — keeping previous alive set "
                "(%d sessions), session-loss finalization SUSPENDED until tmux recovers",
                getattr(res, "returncode", "HUNG"),
                datetime.now(timezone.utc).isoformat(),
                len(self._alive_names),
            )
            return self._alive_names
        # 采样成功 (可能为空 = 确认无会话)
        names = (
            frozenset(n.strip() for n in res.stdout.split("\n") if n.strip())
            if res.stdout else frozenset()
        )
        was_degraded = self._sampler_degraded
        self._sampler_degraded = False
        self._last_sample_ok_ts = time.time()
        prev = self._alive_names
        self._alive_names = names
        if was_degraded:
            logger.warning(
                "tmux sampler RECOVERED — alive=%d sessions: %s",
                len(names), sorted(names),
            )
        if not names and prev:
            # 跳变告警: 健康采样下非空→空 = 所有会话真没了 (唯一合法的置空途径)。
            logger.warning(
                "tmux sampler confirmed EMPTY (was %d sessions) — RUNNING scripts "
                "will be finalized after %d confirmations; tmux is healthy so this "
                "is authoritative", len(prev), SESSION_LOSS_CONFIRM_COUNT,
            )
        # 原始输出落盘 (审计): 每次成功采样到底看到了哪些会话。
        logger.debug("tmux sample ok: %d sessions: %s", len(names), sorted(names))
        return names

    # ── VRAM scanner ────────────────────────────────────────────────────────

    @_guarded
    def _count_free_gpus(
        self,
        gpu_nums: int,
        max_mem_quota: int | None,
        need_exclusive_gpu: bool = True,  # 默认独占 (2026-08-29)
        booked: set[int] | None = None,
    ) -> tuple[int, list[int]]:
        """Count GPUs under threshold with enough free memory.
        Returns (count, device_ids). Ignores GPU core utilization — VRAM is the signal.

        *booked* = extra device ids to always exclude (default empty).

        Exclusivity semantics (see ``_device_leases``):
        - ``need_exclusive_gpu=True``: a card is usable only if no RUNNING gpu-backend
          script holds it — exclusive tasks land only on fully-clean cards.
        - ``need_exclusive_gpu=False``: a card is usable if every holder's
          lease has expired (or it was never booked) — sharing after the lease window.
          (Default is True since 2026-08-29 — exclusive tasks land only on clean cards.)
        Without the booking check, two rapid submissions land on the same still-"free"-
        looking card (model load takes >60s to show up in nvidia-smi), and stage-2 vLLM
        then can't reserve its 0.9xVRAM on the shared card — the task-#6 4/5-experiment
        failure.
        """
        gpu_info = self._gpu_monitor.get_gpu_info()
        booked = booked or set()
        leases = self._device_leases()
        now = time.time()
        candidates: list[tuple[int, int]] = []  # (device_id, free_mb)
        for d in gpu_info.devices:
            if d.device_id in booked:
                continue
            if d.device_id in leases:
                # Card is held by a RUNNING script (hard lease value is None —
                # must use membership, not .get(), to distinguish "no lease").
                if need_exclusive_gpu:
                    continue  # exclusive task needs a fully-clean card
                lease = leases[d.device_id]
                if lease is None or lease > now:
                    continue  # hard lease, or soft lease within its window
            if d.vram_usage_percent >= self._gpu_monitor.MAX_MEMORY_UTILIZATION:
                continue
            if max_mem_quota is not None and d.free_memory_mb < max_mem_quota:
                continue
            candidates.append((d.device_id, d.free_memory_mb))
        candidates.sort(key=lambda x: x[1], reverse=True)
        devices = [dev_id for dev_id, _ in candidates[:gpu_nums]]
        return len(devices), devices

    @_guarded
    def _device_leases(self) -> dict[int, float | None]:
        """Map device_id → lease expiry (epoch seconds) for every RUNNING script.

        - ``need_exclusive_gpu=True`` (or unknown ``started_at``): expiry ``None`` — a
          hard lease that never expires (the card stays exclusive until the script's
          tmux session dies).
        - ``need_exclusive_gpu=False``: expiry = ``started_at + NON_EXCLUSIVE_LEASE_SECONDS`` —
          the card opens to other non-exclusive tasks once this window passes.
        When multiple scripts share a device (past-lease sharing), keep the most
        conservative lease: any hard lease wins, otherwise the latest expiry.
        """
        leases: dict[int, float | None] = {}
        for info in self._scripts.values():
            if info.status not in {ScriptStatus.PENDING, ScriptStatus.RUNNING} or not info.gpu_devices:
                continue
            if info.status == ScriptStatus.PENDING or info.need_exclusive_gpu or info.started_at is None:
                expiry: float | None = None
            else:
                expiry = info.started_at.timestamp() + NON_EXCLUSIVE_LEASE_SECONDS
            for dev in info.gpu_devices:
                if dev not in leases:
                    leases[dev] = expiry
                elif expiry is None:
                    leases[dev] = None
                elif leases[dev] is not None and expiry > leases[dev]:
                    leases[dev] = expiry
        return leases

    @_guarded
    def _booked_devices(self) -> set[int]:
        """Device ids currently allocated to RUNNING scripts (anti double-booking).

        This is the "exclusive requester" view — any RUNNING script occupies the card
        regardless of whether its lease has expired. The booking is released by
        ``_sweep_running_scripts`` when a script's tmux session dies (background
        wrapper ends → tmux session exits).
        """
        return set(self._device_leases())

    def _lease_snapshot(self) -> dict[int, tuple[str, float | None]]:
        """device_id → (holder_script_id, lease_expiry) for every RUNNING script.

        expiry None = hard lease (need_exclusive_gpu or unknown started_at) — the
        card is held until the script's tmux session dies.  Only used to build the
        scan/release audit log; ``_device_leases`` remains the scheduling authority.
        """
        snapshot: dict[int, tuple[str, float | None]] = {}
        for info in self._scripts.values():
            if info.status not in {ScriptStatus.PENDING, ScriptStatus.RUNNING} or not info.gpu_devices:
                continue
            expiry: float | None = None
            if info.status != ScriptStatus.PENDING and not info.need_exclusive_gpu and info.started_at is not None:
                expiry = info.started_at.timestamp() + NON_EXCLUSIVE_LEASE_SECONDS
            for dev in info.gpu_devices:
                snapshot[dev] = (info.script_id, expiry)
        return snapshot

    def _leases_desc(self) -> str:
        """Compact ``{dev@holder(H|~Ns),...}`` of current GPU bookings (for logs)."""
        now = time.time()
        parts = [
            f"{dev}@{holder}({'H' if expiry is None else f'~{int(expiry - now)}s'})"
            for dev, (holder, expiry) in sorted(self._lease_snapshot().items())
        ]
        return "{" + ",".join(parts) + "}" if parts else "{}"

    def _running_scripts_desc(self) -> str:
        """Compact ``[sid:devs]`` of currently RUNNING scripts (for logs)."""
        return ",".join(
            f"{info.script_id}:{info.gpu_devices}"
            for info in self._scripts.values()
            if info.status == ScriptStatus.RUNNING
        ) or "-"

    @_guarded
    def _sweep_running_scripts(self) -> list[tuple[str, ScriptStatus]]:
        """Re-check RUNNING scripts; those whose tmux session is gone are marked
        completed so their GPU booking is released for the next scan.

        Returns ``(script_id, new_status)`` for scripts finalized this sweep —
        the scanner logs them so every booking release is auditable against the
        corresponding ``RELEASE`` log line.
        """
        finalized: list[tuple[str, ScriptStatus]] = []
        for script_id in list(self._scripts.keys()):
            info = self._scripts[script_id]
            if info.status == ScriptStatus.RUNNING:
                before = info.status
                self.get_script_status(script_id)
                if info.status != before:
                    finalized.append((script_id, info.status))
        return finalized

    @_guarded
    def _next_startable(self) -> tuple[int, str, int, list[int]] | None:
        """Return (queue_index, script_id, free_cnt, device_ids) of the first
        QUEUED task that fits under current VRAM, or None.

        R2/M3 (2026-08-26): 原来头部任务放不下就 ``break`` 饿死队列后面所有任务
        (head-of-line blocking) —— 一个大任务排队等卡时, 后面能放下的小任务也不
        启动。现在继续扫描整个队列, 按优先级从高到低找**第一个当前能放下**的任务;
        更高优先级的任务仍先试, 只是放不下时不再挡后面的任务。
        """
        self._order_queue()
        for i, sid in enumerate(self._queue):
            info = self._scripts.get(sid)
            if info is None or info.status != ScriptStatus.QUEUED:
                continue
            free_cnt, devices = self._count_free_gpus(
                info.gpu_nums,
                info.max_mem_quota,
                need_exclusive_gpu=info.need_exclusive_gpu,
            )
            if free_cnt >= info.gpu_nums:
                return i, sid, free_cnt, devices
            # Backfill is bounded: eventually let cards drain for a large job.
            if self._backfills.get(sid, 0) >= self._max_backfills:
                return None
            logger.debug("Scanner: %s needs %d GPU, only %d free — waiting",
                         sid, info.gpu_nums, free_cnt)
        return None

    def _order_queue(self) -> None:
        """Priority bands; least-recently-served owner, FIFO within each owner.

        Untagged scripts retain FIFO within priority. Ownership is an operational
        quota, not an authorization boundary (agents still have shell access).
        """
        rank = {ScriptPriority.HIGH: 0, ScriptPriority.NORMAL: 1, ScriptPriority.LOW: 2}
        self._queue.sort(key=lambda sid: (
            rank.get(self._scripts[sid].priority, 1),
            self._owner_last_dispatch.get(self._scripts[sid].queue_owner or "", 0),
            self._scripts[sid].created_at.timestamp(),
        ))

    def start_scanner(self) -> None:
        """Start the VRAM-scanner thread (idempotent).

        Called by the server lifespan; tests construct SessionManager directly
        without it so no stray threads linger after a test run.  The scanner is
        its own daemon thread (not an asyncio task on the HTTP event loop): a
        slow ``nvidia-smi``/tmux call must stall a scheduling thread, never the
        loop that answers every endpoint.
        """
        if self._scanner_thread is None or not self._scanner_thread.is_alive():
            self._scanner_thread = threading.Thread(
                target=self._scanner_loop, name="gpu-vram-scanner", daemon=True
            )
            self._scanner_thread.start()
            logger.info("VRAM scanner thread started")

    def _scanner_loop(self) -> None:
        """Background thread: every 5s pick the highest-priority queued task
        that fits current VRAM and launch it.

        Lock discipline (R10, 2026-08-27): the pick (sort / sweep / count) is
        pure-memory under ``_state_lock``; the actual launch
        (``tmux new-session`` — a subprocess that can wedge into D-state) runs
        OUTSIDE the lock (see ``_scanner_iter``).  A wedged tmux/nvidia-smi can
        stall this scheduler thread, never any lock-taking endpoint (/health,
        /scripts/...).  After each start, cooldown 60s to let VRAM register in
        nvidia-smi.
        """
        cooldown_until = 0.0
        while True:
            time.sleep(5)

            if not self._queue:
                continue

            if time.time() < cooldown_until:
                continue

            if self._scanner_iter():
                cooldown_until = time.time() + 60

    def _scanner_iter(self) -> bool:
        """One scan: sort the queue, sweep finished scripts, and launch the
        first task that fits.  Returns True if a task was launched.

        The pick is pure-memory under the lock (no subprocess — GPU state and
        tmux liveness come from caches); the launch (``tmux new-session``) runs
        AFTER the lock is released, so a wedged tmux can never pin the lock.

        Audit log (2026-08-31 复盘): 每个非空队列的扫描都打一条 ``SCAN state``
        快照 (queued/running/finalized/leases/free_excl/pick), 放行新实验时再打
        一条 ``RELEASE`` 记录本轮 freed 的 booking + 目标卡 prior holders —— 任何
        「为什么放行了新实验」都必须能在这两条日志上对上。
        """
        launch: tuple[str, ScriptInfo] | None = None
        with self._state_lock:
            self._order_queue()

            # Release finished scripts' bookings before counting free GPUs.
            # finalized = (script_id, new_status) — 本轮释放的 booking, 是
            # 放行依据的审计锚点。
            finalized = self._sweep_running_scripts()

            # Start the highest-priority task that fits (M3: 放不下的头部任务
            # 不再挡后面的任务)。
            pick = self._next_startable()

            # ── SCAN state 快照: 每次扫描一行, 让「放行了谁」可对上「谁释放了卡」。
            gpu_info = self._gpu_monitor.get_gpu_info()
            # free_excl = 当前对「独占请求者」可用的卡总数 (gpu_nums 取全设备数,
            # 避免 _count_free_gpus 按请求量截断 —— 审计要看的是总量, 不是单任务的够不够)。
            free_excl, _ = self._count_free_gpus(
                len(gpu_info.devices) or 1, None, need_exclusive_gpu=True,
            )
            finalized_desc = ",".join(f"{s}={st.value}" for s, st in finalized) or "-"
            pick_desc = "-"
            if pick is not None:
                _, p_sid, p_free, p_devices = pick
                pick_desc = f"{p_sid} → GPUs {p_devices} (free {p_free})"
            elif self._queue:
                pick_desc = "nothing fits"
            queue_desc = ",".join(
                f"{sid}({self._scripts[sid].priority.value},{self._scripts[sid].gpu_nums},"
                f"{'X' if self._scripts[sid].need_exclusive_gpu else 'S'})"
                for sid in self._queue
            ) or "-"
            logger.info(
                "SCAN state: queued=%d running=[%s] finalized=%s leases=%s "
                "free_excl=%d/%d alive=%d sampler_healthy=%s pick=[%s] | queue=[%s]",
                len(self._queue),
                self._running_scripts_desc(),
                finalized_desc,
                self._leases_desc(),
                free_excl,
                len(gpu_info.devices),
                len(self._alive_names),       # 采样器视角 (tmux 会话数)
                self._sampler_healthy(),      # 采样器健康 (False = 降级/未知)
                pick_desc,
                queue_desc,
            )

            if pick is None:
                return False  # nothing fits this scan — wait for VRAM to free up

            i, sid, free_cnt, devices = pick
            info = self._scripts[sid]
            # 放行依据: 目标卡被占前的 prior holders + 本轮 sweep 释放的 booking。
            prior_holders = self._leases_desc()
            # Pop and start
            for bypassed in self._queue[:i]:
                self._backfills[bypassed] = self._backfills.get(bypassed, 0) + 1
            self._queue.pop(i)
            self._backfills.pop(sid, None)
            self._dispatch_sequence += 1
            if info.queue_owner:
                self._owner_last_dispatch[info.queue_owner] = self._dispatch_sequence
            # Claim atomically before the lock-free tmux launch. Queued-only
            # cancellation must not race a selected-but-not-yet-started script.
            info.status = ScriptStatus.PENDING
            info.gpu_devices = devices
            info.started_at = datetime.now(timezone.utc)
            logger.info(
                "RELEASE %s: prio=%s need=%d exclusive=%s → GPUs %s (free %d); "
                "prior_holders=%s; freed_this_scan=%s",
                sid, info.priority.value, info.gpu_nums, info.need_exclusive_gpu,
                devices, free_cnt, prior_holders, finalized_desc,
            )
            launch = (sid, info)

        if launch is not None:
            sid, info = launch
            try:
                self._start_script(sid, info)
            except Exception as exc:
                with self._state_lock:
                    info.status = ScriptStatus.FAILED
                    info.completed_at = datetime.now(timezone.utc)
                    info.stderr = f"launch failed: {exc}"
                logger.exception("Script %s launch failed", sid)
            return True
        return False

    # ── Session CRUD ───────────────────────────────────────────────────────

    @_guarded
    def create_session(self, cwd: str | None = None) -> ShellSession:
        """Create a new persistent tmux shell session.

        Returns a ``ShellSession`` with the generated id and working directory.
        """
        session_id = f"{SHELL_SESSION_PREFIX}{_short_id()}"
        cwd = cwd or os.getcwd()
        res = _run_tmux("new-session", "-d", "-s", session_id, "-c", cwd)
        if res.returncode != 0:  # HUNG or genuine failure
            raise RuntimeError(f"tmux new-session failed (rc={res.returncode}) for {session_id}")
        # Make the new session visible to cache readers immediately — the
        # sampler won't see it for up to TMUX_SAMPLE_INTERVAL seconds.
        self._alive_names = self._alive_names | {session_id}
        logger.info("Created shell session %s (cwd=%s)", session_id, cwd)
        sess = ShellSession(
            session_id=session_id,
            status=SessionStatus.ACTIVE,
            cwd=cwd,
            created_at=datetime.now(timezone.utc),
            is_background=False,
        )
        self._sessions[session_id] = sess
        return sess

    def _session_exists(self, session_id: str) -> bool:
        """Check whether a tmux session with the given name exists.

        Reads the sampled session set first; a bounded live check covers
        sessions the sampler has not seen yet (external / just-created).  Never
        blocks beyond TMUX_WALL_MS, and never holds the state lock.
        """
        if session_id in self._alive_names:
            return True
        res = _run_tmux("list-sessions", "-F", "#{session_name}", check=False)
        if res.returncode != 0:
            return False
        return session_id in {n.strip() for n in res.stdout.split("\n")}

    def exec_in_session(self, request: SessionExecRequest) -> SessionExecResponse:
        """Execute a command inside an existing (or new) tmux session.

        When *session_id* is ``None``, or the named session does not exist,
        a fresh session is created automatically.
        """
        session_id = request.session_id
        is_new = False
        cwd = None

        if session_id and self._session_exists(session_id):
            # Use existing session
            is_new = False
            cwd = None
        else:
            # Create new (session_id was None, or it didn't exist)
            session = self.create_session()
            session_id = session.session_id
            is_new = True
            cwd = session.cwd

        # Use temp files for reliable output capture
        run_id = _short_id()
        out_file = self._log_dir / f"{session_id}_{run_id}.out"
        rc_file = self._log_dir / f"{session_id}_{run_id}.rc"

        # Use { ...; } (current shell, not subshell) so cd/pwd/export persist.
        # Strip trailing ; so we don't produce double-semicolons.
        cmd = request.command.rstrip().rstrip(";").rstrip()
        wrapped = (
            f"{{ {cmd}; }} > {shlex.quote(str(out_file))} 2>&1; "
            f"echo $? > {shlex.quote(str(rc_file))}"
        )
        _run_tmux("send-keys", "-t", session_id, f" {wrapped}", "Enter")

        # Poll for the rc file (indicates command completion).  0 = no timeout.
        deadline = time.time() + request.timeout if request.timeout > 0 else float('inf')
        while time.time() < deadline:
            if rc_file.exists():
                break
            time.sleep(0.2)

        # Read results
        stdout = ""
        exit_code = -1
        try:
            if out_file.exists():
                stdout = out_file.read_text().rstrip()
            if rc_file.exists():
                exit_code_str = rc_file.read_text().strip()
                try:
                    exit_code = int(exit_code_str)
                except (ValueError, IndexError):
                    exit_code = -1
        except (OSError, IOError):
            pass
        finally:
            # Clean up temp files
            out_file.unlink(missing_ok=True)
            rc_file.unlink(missing_ok=True)

        # Update tracked state (never hold the lock across the poll loop above)
        with self._state_lock:
            if session_id in self._sessions:
                self._sessions[session_id].last_command = request.command
                # Refresh actual working directory from tmux
                try:
                    cwd_res = _run_tmux("display", "-t", session_id, "-p", "#{pane_current_path}",
                                        check=False, timeout=3)
                    new_cwd = cwd_res.stdout.strip()
                    if new_cwd:
                        self._sessions[session_id].cwd = new_cwd
                        cwd = new_cwd
                except subprocess.SubprocessError:
                    pass

        if exit_code == -1 and not stdout and time.time() >= deadline:
            _run_tmux("send-keys", "-t", session_id, "C-c", check=False)
            logger.warning("Command in session %s timed out after %ds", session_id, request.timeout)
            return SessionExecResponse(
                session_id=session_id,
                return_code=-1,
                stdout=stdout,
                stderr="Command timed out after {}s".format(request.timeout),
                cwd=cwd,
                is_new=is_new,
            )

        return SessionExecResponse(
            session_id=session_id,
            return_code=exit_code,
            stdout=stdout,
            stderr="",
            cwd=cwd,
            is_new=is_new,
        )

    @_guarded
    def kill_session(self, session_id: str) -> bool:
        """Kill a tmux shell session and all its child processes."""
        # Check if the session exists first
        try:
            res = _run_tmux("list-sessions", "-F", "#{session_name}", check=False)
            active = [n.strip() for n in res.stdout.split("\n")]
            if session_id not in active:
                logger.info("Session %s does not exist", session_id)
                return False
        except subprocess.SubprocessError:
            return False

        try:
            _run_tmux("kill-session", "-t", session_id, check=False)
            logger.info("Killed session %s", session_id)
            return True
        except subprocess.SubprocessError as exc:
            logger.warning("Failed to kill session %s: %s", session_id, exc)
            return False

    @_guarded
    def list_sessions(self) -> list[ShellSession]:
        """Return every active ``gpu_shell_*`` and ``gpu_backend_*`` tmux session.

        Reads the sampler's session-set cache — no tmux subprocess on the lock
        path, so a wedged tmux/NFS cannot stall /health or /sessions.  Uses the
        in-memory registry for rich info (cwd, last_command, created_at);
        sessions not in the registry fall back to a bare tmux session entry.
        """
        active_names = self._alive_names

        # Clean up dead sessions from registry
        for sid in list(self._sessions.keys()):
            if sid not in active_names:
                self._sessions[sid].status = SessionStatus.KILLED

        sessions: list[ShellSession] = []
        for name in sorted(active_names):
            if not (name.startswith(SHELL_SESSION_PREFIX) or name.startswith(BACKEND_SESSION_PREFIX)):
                continue
            if name in self._sessions:
                # Use rich tracked data
                sessions.append(self._sessions[name])
            else:
                # Bare fallback for sessions created outside this process
                sessions.append(ShellSession(
                    session_id=name,
                    status=SessionStatus.ACTIVE,
                    cwd="/",
                    is_background=name.startswith(BACKEND_SESSION_PREFIX),
                ))
        return sessions

    # ── Script submission ───────────────────────────────────────────────────

    @_guarded
    def submit_script(self, request: ScriptSubmitRequest) -> ScriptSubmitResponse:
        """Submit a bash script — always queue, VRAM scanner starts when ready.

        FIFO within owner, owner fairness within priority. Capacity refusals
        return REJECTED without creating a script or consuming GPU resources.
        """
        script_path = Path(request.command_path)
        if not script_path.exists():
            return ScriptSubmitResponse(
                script_id="",
                status=ScriptStatus.FAILED,
                message=f"Script not found: {request.command_path}",
            )

        if request.queue_owner and self._max_queued_per_owner:
            waiting = sum(self._scripts[s].queue_owner == request.queue_owner for s in self._queue)
            if waiting >= self._max_queued_per_owner:
                return ScriptSubmitResponse(
                    script_id="", status=ScriptStatus.REJECTED, reason_code="queue_full",
                    message=f"Owner already has {waiting} queued jobs (limit {self._max_queued_per_owner}). "
                            "Do not spin-retry; analyze results, prepare data, or withdraw obsolete queued work.",
                )

        script_id = f"gpu_script_{_short_id()}"
        now = datetime.now(timezone.utc)
        free_cnt, _ = self._count_free_gpus(
            request.gpu_nums, request.max_mem_quota,
            need_exclusive_gpu=request.need_exclusive_gpu,
        )

        info = ScriptInfo(
            script_id=script_id,
            command_path=str(script_path),
            queue_owner=request.queue_owner,
            status=ScriptStatus.QUEUED,
            created_at=now,
            priority=request.priority,
            gpu_nums=request.gpu_nums,
            max_mem_quota=request.max_mem_quota,
            need_exclusive_gpu=request.need_exclusive_gpu,
        )
        self._scripts[script_id] = info

        self._queue.append(script_id)
        self._order_queue()

        # Scanner thread is started once by the server lifespan (start_scanner),
        # not here — a submit can arrive from a threadpool thread where no event
        # loop exists.

        logger.info("Script %s queued (need %d GPU, priority=%s, exclusive=%s, %d free now)",
                    script_id, request.gpu_nums, request.priority.value,
                    request.need_exclusive_gpu, free_cnt)
        return ScriptSubmitResponse(
            script_id=script_id,
            status=ScriptStatus.QUEUED,
            message=f"Queued — need {request.gpu_nums} GPU(s), currently {free_cnt} free. "
                    f"Priority: {request.priority.value}. Exclusive: {request.need_exclusive_gpu}. "
                    f"Queue size: {len(self._queue)}.",
        )

    def _start_script(self, script_id: str, info: ScriptInfo) -> None:
        """Launch *info*'s script in a background tmux session.

        Called by ``_scanner_loop`` AFTER releasing the state lock — the tmux
        subprocess here is the one thing that can wedge, so it must never run
        under ``_state_lock``.  On a wedged/failed launch the script is marked
        FAILED (and its GPU booking released via status != RUNNING) rather than
        left QUEUED forever.
        """
        session_id = f"{BACKEND_SESSION_PREFIX}{script_id.split('_')[-1]}"
        log_path = self._log_dir / f"{script_id}.log"

        # Set CUDA_VISIBLE_DEVICES if GPUs are allocated
        env_prefix = ""
        gpu_devices = info.gpu_devices if info else []
        if gpu_devices:
            env_prefix = f"CUDA_VISIBLE_DEVICES={','.join(str(d) for d in gpu_devices)} "

        script_cmd = f"{env_prefix}bash {shlex.quote(str(info.command_path))}"
        # Append exit-code capture + release marker
        release_sig = f"echo __RELEASE__{script_id}__ >> {shlex.quote(str(log_path))}"
        exit_marker = f"echo EXIT_CODE=$? >> {shlex.quote(str(log_path))}"
        wrapper = f"{script_cmd} > {shlex.quote(str(log_path))} 2>&1; {exit_marker}; {release_sig}"

        res = _run_tmux("new-session", "-d", "-s", session_id, wrapper)

        with self._state_lock:
            if res.returncode != 0:  # HUNG (-1) or genuine failure
                info.status = ScriptStatus.FAILED
                info.return_code = res.returncode
                info.completed_at = datetime.now(timezone.utc)
                info.stderr = f"tmux new-session failed (rc={res.returncode})"
                if script_id in self._queue:
                    self._queue.remove(script_id)
                logger.error("Script %s: %s", script_id, info.stderr)
                return

            info.status = ScriptStatus.RUNNING
            info.session_id = session_id
            info.log_path = str(log_path)
            info.started_at = info.started_at or datetime.now(timezone.utc)
            self._scripts[script_id] = info

            # Register in session registry so list-sessions shows rich info
            last_cmd = (
                f"submit -p {info.command_path}"
                f" --background -g {info.gpu_nums}"
                f" --priority {info.priority.value}"
                + (f" --max-mem-quota {info.max_mem_quota}" if info.max_mem_quota else "")
                + (" --need-exclusive-gpu" if info.need_exclusive_gpu else "")
                + f" -t 0"
            )
            self._sessions[session_id] = ShellSession(
                session_id=session_id,
                status=SessionStatus.ACTIVE,
                cwd="/",
                created_at=info.started_at,
                is_background=True,
                log_path=str(log_path),
                last_command=last_cmd,
            )
            # Make the new session visible to cache readers immediately.
            self._alive_names = self._alive_names | {session_id}

            logger.info("Background script %s → tmux %s, log %s, GPUs %s",
                        script_id, session_id, log_path, gpu_devices)

    def _submit_foreground(
        self,
        script_id: str,
        script_path: Path,
        request: ScriptSubmitRequest,
        now: datetime,
        info: ScriptInfo | None = None,
    ) -> ScriptSubmitResponse:
        """Run script in a foreground tmux session, blocking until completion.

        Output is captured via temp files.  If execution exceeds
        *AUTO_BACKGROUND_SECONDS* the task is moved to a background session.
        """
        session_id = f"{BACKEND_SESSION_PREFIX}{script_id.split('_')[-1]}"
        out_file = self._log_dir / f"{script_id}.out"
        rc_file = self._log_dir / f"{script_id}.rc"

        # Set CUDA_VISIBLE_DEVICES if GPUs are allocated
        gpu_devices = info.gpu_devices if info else []
        env_prefix = ""
        if gpu_devices:
            env_prefix = f"CUDA_VISIBLE_DEVICES={','.join(str(d) for d in gpu_devices)} "

        # Build the shell command
        parts: list[str] = []
        if request.resume_session_id:
            parts.append(
                f"source <(tmux show-environment -t {shlex.quote(request.resume_session_id)}) 2>/dev/null"
            )
        parts.append(f"{env_prefix}bash {shlex.quote(str(script_path))}")
        wrapper = (
            "{ " + "; ".join(parts) + "; }"
            f" > {shlex.quote(str(out_file))} 2>&1; "
            f"echo $? > {shlex.quote(str(rc_file))}"
        )

        _run_tmux("new-session", "-d", "-s", session_id, wrapper)

        if info is None:
            info = ScriptInfo(
                script_id=script_id,
                command_path=str(script_path),
                gpu_devices=gpu_devices,
                priority=request.priority,
                gpu_nums=request.gpu_nums,
                max_mem_quota=request.max_mem_quota,
                need_exclusive_gpu=request.need_exclusive_gpu,
            )
        info.status = ScriptStatus.RUNNING
        info.session_id = session_id
        info.started_at = now
        self._scripts[script_id] = info

        deadline = time.time() + request.timeout if request.timeout > 0 else float('inf')
        started = time.time()
        is_auto_bg = False

        while time.time() < deadline:
            if rc_file.exists():
                break

            # Auto-background: if running > 5 min, detach and let it finish
            elapsed = time.time() - started
            if elapsed > AUTO_BACKGROUND_SECONDS:
                log_path = self._log_dir / f"{script_id}.log"
                try:
                    if out_file.exists():
                        log_path.write_text(out_file.read_text())
                except (OSError, IOError):
                    pass
                info.log_path = str(log_path)
                info.status = ScriptStatus.RUNNING
                is_auto_bg = True
                logger.info("Script %s auto-converted to background after %.0fs", script_id, elapsed)
                break

            time.sleep(0.5)

        if is_auto_bg:
            return ScriptSubmitResponse(
                script_id=script_id,
                session_id=session_id,
                log_path=str(info.log_path),
                status=ScriptStatus.RUNNING,
                message=(
                    f"Script running > {AUTO_BACKGROUND_SECONDS}s, "
                    f"converted to background. Log: {info.log_path}"
                ),
            )

        # Completed or timeout — release GPU allocation
        self._release_allocation(script_id)

        # Read results
        stdout = ""
        exit_code = -1
        try:
            if out_file.exists():
                stdout = out_file.read_text().rstrip()
            if rc_file.exists():
                exit_code_str = rc_file.read_text().strip()
                try:
                    exit_code = int(exit_code_str)
                except (ValueError, IndexError):
                    exit_code = -1
        except (OSError, IOError):
            pass
        finally:
            out_file.unlink(missing_ok=True)
            rc_file.unlink(missing_ok=True)

        # Clean up the tmux session
        self.kill_session(session_id)

        if exit_code == -1 and not stdout and time.time() >= deadline:
            info.status = ScriptStatus.KILLED
            info.return_code = -1
            info.completed_at = datetime.now(timezone.utc)
            info.stderr = f"Script killed after timeout ({request.timeout}s)"
            return ScriptSubmitResponse(
                script_id=script_id,
                status=ScriptStatus.KILLED,
                return_code=-1,
                stderr=info.stderr,
                message=info.stderr,
            )

        status = ScriptStatus.COMPLETED if exit_code == 0 else ScriptStatus.FAILED
        info.status = status
        info.return_code = exit_code
        info.stdout = stdout
        info.completed_at = datetime.now(timezone.utc)

        return ScriptSubmitResponse(
            script_id=script_id,
            status=status,
            return_code=exit_code,
            stdout=stdout,
            message=f"Script completed with exit code {exit_code}",
        )

    # ── Script queries ──────────────────────────────────────────────────────

    @staticmethod
    def _read_log_tail(log_path: str | Path, nbytes: int = 8192) -> str:
        """Read the trailing *nbytes* of a (possibly multi-GB) script log.

        Wrapper 的退出标记 (``EXIT_CODE=`` / ``__RELEASE__``) 总是写在 log 末尾,
        只读尾部就够判断, 避免每 5s 全量读一次训练日志。
        """
        try:
            with Path(log_path).open("rb") as fh:
                fh.seek(0, 2)
                size = fh.tell()
                fh.seek(max(0, size - nbytes))
                return fh.read().decode("utf-8", errors="replace")
        except OSError:
            return ""

    @_guarded
    def _finalize_from_log(self, info: ScriptInfo) -> None:
        """Mark a RUNNING script COMPLETED/FAILED from its wrapper log.

        wrapper 在脚本结束时写 ``EXIT_CODE=$?`` 再写 ``__RELEASE__<id>__``;
        标记出现即管线 (或 wrapper) 已退出 —— 不管 tmux 会话是否还残留孤儿孙进程。
        读 EXIT_CODE 定状态 (0 → COMPLETED, 非 0 → FAILED), 释放 booking 让
        scanner 可以放新任务。
        """
        info.completed_at = datetime.now(timezone.utc)
        elapsed = _elapsed_seconds(info, info.completed_at)
        if info.log_path and Path(info.log_path).exists():
            log_text = self._read_log_tail(info.log_path)
            for line in reversed(log_text.split("\n")):
                if line.startswith("EXIT_CODE="):
                    try:
                        info.return_code = int(line.split("=", 1)[1])
                    except (ValueError, IndexError):
                        info.return_code = 0
                    info.status = (
                        ScriptStatus.COMPLETED if info.return_code == 0
                        else ScriptStatus.FAILED
                    )
                    logger.info(
                        "Script %s finalized %s (exit_code=%s, elapsed=%.0fs, booking released)",
                        info.script_id, info.status.value, info.return_code, elapsed,
                    )
                    return
        # 危险路径 (2026-08-31 复盘): 日志尾没有 EXIT_CODE= 也强制 COMPLETED/rc=0 并
        # 释放 booking。一个还在训练、只是 tmux 会话暂时从 sampler 缓存消失的脚本会
        # 被误判 —— 这正是 HLE 0830 节点 OOM 的根因链路起点。这里必须 WARNING 亮出来,
        # 不许被静默吞掉。
        info.status = ScriptStatus.COMPLETED
        info.return_code = 0
        logger.warning(
            "Script %s: no EXIT_CODE in log tail — forced COMPLETED rc=0 "
            "(booking released UNCONFIRMED, elapsed=%.0fs)",
            info.script_id, elapsed,
        )

    @_guarded
    def _authoritative_finalized(self, info: ScriptInfo) -> bool:
        """会话消失时, 日志里是否有权威终结信号 (release 标记或 EXIT_CODE=)。

        有 → wrapper 已退出、管线确实结束 —— 与 tmux 会话状态无关 (日志是文件制,
        tmux 卡死/NFS 抖动也读得到), 可无条件终结并释放 booking;
        无 → 只能走采样器健康确认路径 (降级防护, 见 ``get_script_status``)。
        """
        if not info.log_path or not Path(info.log_path).exists():
            return False
        tail = self._read_log_tail(info.log_path)
        if f"__RELEASE__{info.script_id}__" in tail:
            return True
        return any(line.startswith("EXIT_CODE=") for line in tail.split("\n"))

    def _sweep_release_marker(self, info: ScriptInfo) -> None:
        """R2/M4a: 如果 wrapper 已写 __RELEASE__<id>__ 标记, 视为管线结束.

        死 pipeline (被 kill / NCCL 死锁后 wrapper 完成但孤儿进程让 tmux 会话残留)
        只凭「会话没了」判完成的话会永远占用 booking —— 调度器把卡当被占 → 新任务
        全打包到剩卡上 → 反复 OOM。wrapper 退出时总是写 ``__RELEASE__<id>__``
        (见 ``_start_script``), 标记出现就终结脚本并释放 booking。
        """
        if info.status != ScriptStatus.RUNNING:
            return
        if not info.log_path or not Path(info.log_path).exists():
            return
        marker = f"__RELEASE__{info.script_id}__"
        if marker in self._read_log_tail(info.log_path):
            logger.info(
                "Script %s: release marker seen — finalizing (tmux session may "
                "linger with orphaned grandchildren)", info.script_id,
            )
            self._finalize_from_log(info)

    @_guarded
    def get_script_status(self, script_id: str) -> ScriptInfo | None:
        """Return the latest info for a script.

        Session liveness is read from the sampler cache — no tmux subprocess
        under the lock (a wedged tmux/NFS would otherwise hold it for minutes).
        """
        info = self._scripts.get(script_id)
        if info is None:
            return None

        # If background + running, check if the tmux session is still alive
        if info.status == ScriptStatus.RUNNING and info.session_id:
            if info.session_id in self._alive_names:
                # R2/M4a: 会话还在但 release 标记已写 → 管线确实结束, 释放 booking,
                # 不再让死任务独占卡 (belt-and-suspenders, 见 _sweep_release_marker)。
                self._session_missing.pop(info.script_id, None)
                self._sweep_release_marker(info)
                return info

            # ── 会话从采样缓存消失。终结门 (2026-08-31 降级防护, HLE 0830 OOM 修复):
            # 1) 权威信号优先: release 标记 / EXIT_CODE 在 → 管线确实结束, 无条件终结
            #    (日志是文件制, tmux 卡死也读得到, 不受采样器健康状态影响)。
            # 2) 否则采样器不健康/降级 → 保持 RUNNING 不放卡。tmux 卡死/NFS D-state 时
            #    HUNG 曾折叠成空缓存让所有脚本「看起来全死」→ 误终结 + 释放 booking
            #    → 新实验上卡 → 节点 OOM。降级时 booking 必须守住。
            # 3) 采样器健康 → 还需连续 SESSION_LOSS_CONFIRM_COUNT 次确认缺失才终结
            #    (单次采样抖动不释放 booking)。全部决策都落 scheduler.log。
            if self._authoritative_finalized(info):
                logger.warning(
                    "Script %s: session %s gone + authoritative marker/EXIT_CODE in "
                    "log — finalizing (booking released)", info.script_id, info.session_id,
                )
                self._session_missing.pop(info.script_id, None)
                self._finalize_from_log(info)
                return info
            if not self._sampler_healthy():
                # 降级/未知采样期间不累计确认计数 —— 恢复后从零开始, 宁慢勿误放。
                self._session_missing.pop(info.script_id, None)
                logger.warning(
                    "Script %s: session %s absent but sampler %s (last_ok=%s, "
                    "alive=%d) — HOLDING booking, NOT finalizing (degraded-sample "
                    "guard; no authoritative marker/EXIT_CODE)",
                    info.script_id, info.session_id,
                    "degraded" if self._sampler_degraded else "stale/never-ok",
                    self._last_sample_ok_ts, len(self._alive_names),
                )
                return info
            n = self._session_missing.get(info.script_id, 0) + 1
            self._session_missing[info.script_id] = n
            if n < SESSION_LOSS_CONFIRM_COUNT:
                logger.info(
                    "Script %s: session %s absent, sampler healthy — confirm %d/%d "
                    "(alive=%d sessions), holding booking",
                    info.script_id, info.session_id, n, SESSION_LOSS_CONFIRM_COUNT,
                    len(self._alive_names),
                )
                return info
            logger.warning(
                "Script %s: session %s absent confirmed %d healthy samples — "
                "finalizing from log", info.script_id, info.session_id, n,
            )
            self._session_missing.pop(info.script_id, None)
            self._finalize_from_log(info)
            return info

        return info

    @_guarded
    def get_script_log(self, script_id: str) -> ScriptLog | None:
        """Return script log output."""
        info = self._scripts.get(script_id)
        if info is None:
            return None

        stdout = info.stdout or ""
        if info.log_path and Path(info.log_path).exists():
            # Tail the log — a full read of a multi-GB training log on NFS would
            # hold a worker thread for seconds on every /log poll.
            stdout = self._read_log_tail(info.log_path, nbytes=256 * 1024)

        return ScriptLog(
            script_id=script_id,
            stdout=stdout,
            stderr=info.stderr or "",
            return_code=info.return_code,
            is_running=info.status == ScriptStatus.RUNNING,
        )

    def _kill_session_pgroup(self, session_id: str) -> bool:
        """R7 (2026-08-26): kill the whole process group rooted at the pane PID.

        ``tmux kill-session`` 只杀 pane 的 shell 及其直接子进程; 孤儿孙进程
        (vLLM::Worker_* / torchrun worker) 会继续占显存 —— 0826 批次 worker 主动
        kill 后 booking 释放了, 但卡被孤儿占满 → 新任务照样 OOM。这里先取 pane
        进程的 PGID, 用 ``os.killpg`` 整组送 SIGTERM (短暂宽限后 SIGKILL), 再让
        kill-session 收尾。拿不到 PID / 组已死 / 无权限都静默退回 kill-session。
        """
        try:
            res = _run_tmux(
                "display", "-t", session_id, "-p", "#{pane_pid}",
                check=False, timeout=3,
            )
            pane_pid = int(res.stdout.strip())
        except (subprocess.SubprocessError, ValueError):
            return False
        try:
            pgid = os.getpgid(pane_pid)
        except (ProcessLookupError, OSError):
            return False
        try:
            os.killpg(pgid, signal.SIGTERM)
            time.sleep(0.2)
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            # 组已退出 (SIGTERM 生效) 或没权限 —— 非致命, kill-session 继续收尾.
            pass
        return True

    @_guarded
    def kill_script(self, script_id: str) -> bool:
        """Kill a running/queued script and its tmux session.

        R7 (2026-08-26): 对运行中脚本先整进程组 kill (孤儿 vLLM worker 一并清掉,
        否则 booking 释放了但卡仍被孤儿占满), 再 tmux kill-session 收尾。
        """
        info = self._scripts.get(script_id)
        if info is None or info.status not in (ScriptStatus.RUNNING, ScriptStatus.QUEUED):
            return False

        # Queued → just remove from queue
        if info.status == ScriptStatus.QUEUED:
            if script_id in self._queue:
                self._queue.remove(script_id)
            info.status = ScriptStatus.KILLED
            info.completed_at = datetime.now(timezone.utc)
            info.return_code = -1
            return True

        # Running — kill whole process group (R7) + tmux session
        killed = False
        if info.session_id:
            killed = self._kill_session_pgroup(info.session_id)
            killed = self.kill_session(info.session_id) or killed

        info.status = ScriptStatus.KILLED
        info.completed_at = datetime.now(timezone.utc)
        info.return_code = -1
        return killed or True

    # ── Convenience ─────────────────────────────────────────────────────────

    @_guarded
    def cancel_queued_script(self, script_id: str) -> dict[str, Any]:
        """Atomic queued-only withdrawal; never falls back to killing a job."""
        info = self._scripts.get(script_id)
        if info is None:
            return {"cancelled": False, "status": "not_found"}
        if info.status == ScriptStatus.KILLED and info.stderr == "cancelled_before_start":
            return {"cancelled": True, "status": "killed", "already_cancelled": True}
        if info.status != ScriptStatus.QUEUED or script_id not in self._queue:
            return {"cancelled": False, "status": info.status.value, "reason": "not_queued"}
        self._queue.remove(script_id)
        self._backfills.pop(script_id, None)
        info.status = ScriptStatus.KILLED
        info.completed_at = datetime.now(timezone.utc)
        info.stderr = "cancelled_before_start"
        logger.info("WITHDRAW %s owner=%s (never started)", script_id, info.queue_owner)
        return {"cancelled": True, "status": "killed"}

    @_guarded
    def get_queue_snapshot(self) -> dict[str, Any]:
        """Cheap live scheduler state, no nvidia-smi or process operations."""
        self._order_queue()
        def row(info):
            return {"script_id": info.script_id, "queue_owner": info.queue_owner,
                    "status": info.status.value, "gpu_nums": info.gpu_nums,
                    "gpu_devices": info.gpu_devices, "priority": info.priority.value,
                    "created_at": info.created_at.isoformat(),
                    "started_at": info.started_at.isoformat() if info.started_at else None}
        return {"policy": "priority_owner_round_robin_fifo", "max_queued_per_owner": self._max_queued_per_owner,
                "max_backfills": self._max_backfills,
                "queued": [row(self._scripts[s]) for s in self._queue],
                "running": [row(i) for i in self._scripts.values() if i.status in {ScriptStatus.PENDING, ScriptStatus.RUNNING}],
                "note": "Queue order may change after dispatch; empty queue is not proof of idle GPUs."}

    def get_status_snapshot(self, gpu_monitor=None) -> dict[str, Any]:
        """Return a combined view of sessions + GPU info + queue."""
        gpu_info = gpu_monitor.get_gpu_info() if gpu_monitor else None
        queued = [
            {"script_id": sid,
             "priority": self._scripts[sid].priority.value if sid in self._scripts else "?",
             "gpu_nums": self._scripts[sid].gpu_nums if sid in self._scripts else 0}
            for sid in self._queue
        ]
        return {
            "sessions": self.list_sessions(),
            "gpu_info": gpu_info,
            "queue": queued,
        }
