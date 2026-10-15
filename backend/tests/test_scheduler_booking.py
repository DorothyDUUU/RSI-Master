"""Hermetic unit tests for SessionManager GPU booking-awareness (anti double-booking).

Regression (2026-08-12, task #6 aime 15h): 4/5 experiments failed at stage-2 with
``ValueError: Free memory on device (...) is less than desired GPU memory
utilization (0.9, ...)``. Root cause: the VRAM scanner had no notion of "card
already assigned to a RUNNING script" — two pipelines landed on card 0
(exp_0001 + exp_0002) and the rest went to gpu_occupy'd cards (~42GiB held),
leaving <125.83GiB for vLLM's default 0.9xVRAM reservation.

The fix: ``_count_free_gpus`` excludes devices already allocated to RUNNING
scripts (``_booked_devices``), and ``_sweep_running_scripts`` releases a card
once the script's tmux session ends.

These tests monkeypatch GPUMonitor (no real GPU) and tmux (no sessions), so they
run anywhere.  Run:

    cd backend && ../.venv/bin/python -m pytest tests/test_scheduler_booking.py -q
"""

from __future__ import annotations

import logging
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import pytest  # noqa: E402

from server.models import (  # noqa: E402
    GPUDevice,
    GPUInfo,
    ScriptInfo,
    ScriptPriority,
    ScriptStatus,
    ScriptSubmitRequest,
)
from server.session_manager import (  # noqa: E402
    SESSION_LOSS_CONFIRM_COUNT,
    SessionManager,
)
from server.subproc import HUNG  # noqa: E402


@pytest.fixture(autouse=True)
def _no_real_gpu(monkeypatch):
    """Keep SessionManager construction hermetic (no nvidia-smi / nvitop)."""
    monkeypatch.setattr(
        "server.gpu_monitor.GPUMonitor._check_nvidia_smi", lambda self: False
    )
    monkeypatch.setattr(
        "server.gpu_monitor.GPUMonitor._check_nvitop", lambda self: False
    )


def _gpu_info(devices: list[tuple[int, int, int]]) -> GPUInfo:
    """Build a fake GPUInfo from (device_id, used_mb, total_mb) tuples."""
    return GPUInfo(
        devices=[
            GPUDevice(
                device_id=dev_id,
                name=f"gpu-{dev_id}",
                total_memory_mb=total,
                used_memory_mb=used,
                free_memory_mb=total - used,
                gpu_utilization=0.0,
                memory_utilization=0.0,
            )
            for dev_id, used, total in devices
        ],
        total_devices=len(devices),
    )


def _running_script(
    session_id: str = "gpu_backend_x",
    gpu_devices=(0,),
    *,
    script_id: str = "gpu_script_test",
    started_at: datetime | None = None,
    need_exclusive_gpu: bool = False,
) -> ScriptInfo:
    return ScriptInfo(
        script_id=script_id,
        command_path="/tmp/nonexistent.sh",
        status=ScriptStatus.RUNNING,
        session_id=session_id,
        gpu_devices=list(gpu_devices),
        priority=ScriptPriority.HIGH,
        gpu_nums=1,
        started_at=started_at,
        need_exclusive_gpu=need_exclusive_gpu,
    )


def _started_ago(minutes: int) -> datetime:
    """A RUNNING script started *minutes* ago (aware UTC)."""
    return datetime.now(timezone.utc) - timedelta(minutes=minutes)


# ── _count_free_gpus excludes booked devices ────────────────────────────────

def test_count_free_gpus_excludes_booked(monkeypatch):
    mgr = SessionManager(log_dir="/tmp/test-gpu-log")
    monkeypatch.setattr(
        mgr._gpu_monitor,
        "get_gpu_info",
        lambda: _gpu_info([(0, 0, 140000), (1, 0, 140000), (2, 0, 140000)]),
    )

    n, devs = mgr._count_free_gpus(1, None, booked={1})
    assert n == 1 and devs == [0]

    n, devs = mgr._count_free_gpus(2, None, booked={0})
    assert n == 2 and set(devs) == {1, 2}

    # too few non-booked cards → caller's `free_cnt >= gpu_nums` keeps it queued
    n, devs = mgr._count_free_gpus(2, None, booked={0, 1})
    assert n == 1 and devs == [2]


def test_card_that_looks_vram_free_but_is_booked_is_not_reoffered(monkeypatch):
    """Card 0 shows 0 VRAM used (training not yet allocated) but is booked by a
    RUNNING script — the scanner must not hand it out again."""
    mgr = SessionManager(log_dir="/tmp/test-gpu-log")
    monkeypatch.setattr(
        mgr._gpu_monitor,
        "get_gpu_info",
        lambda: _gpu_info([(0, 0, 140000), (1, 0, 140000)]),
    )
    mgr._scripts["gpu_script_1"] = _running_script(gpu_devices=(0,))

    n, devs = mgr._count_free_gpus(1, None, booked=mgr._booked_devices())
    assert n == 1 and devs == [1]


# ── _booked_devices tracking ────────────────────────────────────────────────

def test_booked_devices_tracks_running_scripts_only():
    mgr = SessionManager(log_dir="/tmp/test-gpu-log")
    mgr._scripts["gpu_script_1"] = _running_script(gpu_devices=(0,))
    mgr._scripts["gpu_script_2"] = _running_script(
        session_id="gpu_backend_y", gpu_devices=(3,)
    )
    assert mgr._booked_devices() == {0, 3}

    mgr._scripts["gpu_script_2"].status = ScriptStatus.COMPLETED
    assert mgr._booked_devices() == {0}


# ── sweep releases bookings when the tmux session dies ──────────────────────

def test_sweep_releases_booking_when_session_dead(monkeypatch):
    """会话消失且采样器健康 → 连续 SESSION_LOSS_CONFIRM_COUNT 次确认后才终结释放.

    2026-08-31 降级防护: 单次采样看不到会话不立即释放 booking (可能是采样抖动),
    健康采样连续 N 次都看不到才按「真死了」终结。
    """
    mgr = SessionManager(log_dir="/tmp/test-gpu-log")
    # Session set is sampled by the tmux sampler; seed it like a sampler tick
    # that saw only an unrelated session (this script's session is gone).
    mgr._alive_names = frozenset({"other_session"})
    mgr._last_sample_ok_ts = time.time()   # 采样器健康
    mgr._sampler_degraded = False

    s = _running_script()
    mgr._scripts["gpu_script_1"] = s

    mgr._sweep_running_scripts()
    assert s.status == ScriptStatus.RUNNING   # confirm 1/N → 守 booking
    assert mgr._booked_devices() == {0}

    mgr._sweep_running_scripts()
    assert s.status == ScriptStatus.COMPLETED  # confirm N/N → 终结释放
    assert mgr._booked_devices() == set()


def test_sweep_keeps_booking_when_session_alive(monkeypatch):
    mgr = SessionManager(log_dir="/tmp/test-gpu-log")
    # R10: liveness is read from the sampler cache, not a live tmux call.
    mgr._alive_names = frozenset({"gpu_backend_x"})

    s = _running_script()
    mgr._scripts["gpu_script_1"] = s

    mgr._sweep_running_scripts()
    assert s.status == ScriptStatus.RUNNING
    assert mgr._booked_devices() == {0}


# ── 20-min exclusive lease (--need-exclusive-gpu) ──────────────────────────

def test_shared_task_sees_card_within_lease_as_unavailable(monkeypatch):
    """A non-exclusive task's 20-min lease blocks other non-exclusive tasks."""
    mgr = SessionManager(log_dir="/tmp/test-gpu-log")
    monkeypatch.setattr(
        mgr._gpu_monitor,
        "get_gpu_info",
        lambda: _gpu_info([(0, 0, 140000), (1, 0, 140000)]),
    )
    mgr._scripts["gpu_script_1"] = _running_script(
        gpu_devices=(0,), started_at=_started_ago(5), need_exclusive_gpu=False,
    )

    n, devs = mgr._count_free_gpus(1, None, need_exclusive_gpu=False)
    assert n == 1 and devs == [1]  # card 0 still within its 20-min lease


def test_shared_task_can_share_card_after_lease_expires(monkeypatch):
    """Once the lease has passed, a non-exclusive task may land on the same card."""
    mgr = SessionManager(log_dir="/tmp/test-gpu-log")
    monkeypatch.setattr(
        mgr._gpu_monitor,
        "get_gpu_info",
        lambda: _gpu_info([(0, 0, 140000), (1, 0, 140000)]),
    )
    mgr._scripts["gpu_script_1"] = _running_script(
        gpu_devices=(0,), started_at=_started_ago(30), need_exclusive_gpu=False,
    )

    # gpu_nums=2: both cards eligible → the expired-lease card 0 is shareable
    n, devs = mgr._count_free_gpus(2, None, need_exclusive_gpu=False)
    assert n == 2 and set(devs) == {0, 1}


def test_exclusive_task_requires_clean_card_even_with_expired_lease(monkeypatch):
    """An exclusive task never lands on a card holding any RUNNING script,
    even one whose soft lease has already expired."""
    mgr = SessionManager(log_dir="/tmp/test-gpu-log")
    monkeypatch.setattr(
        mgr._gpu_monitor,
        "get_gpu_info",
        lambda: _gpu_info([(0, 0, 140000), (1, 0, 140000)]),
    )
    mgr._scripts["gpu_script_1"] = _running_script(
        gpu_devices=(0,), started_at=_started_ago(30), need_exclusive_gpu=False,
    )

    n, devs = mgr._count_free_gpus(1, None, need_exclusive_gpu=True)
    assert n == 1 and devs == [1]  # card 0 still occupied by any RUNNING script


def test_hard_lease_never_expires_for_shared_requesters(monkeypatch):
    """need_exclusive_gpu=True holds the card forever — non-exclusive tasks
    never see it, regardless of how long ago it started."""
    mgr = SessionManager(log_dir="/tmp/test-gpu-log")
    monkeypatch.setattr(
        mgr._gpu_monitor,
        "get_gpu_info",
        lambda: _gpu_info([(0, 0, 140000), (1, 0, 140000)]),
    )
    mgr._scripts["gpu_script_1"] = _running_script(
        gpu_devices=(0,), started_at=_started_ago(120), need_exclusive_gpu=True,
    )

    n, devs = mgr._count_free_gpus(1, None, need_exclusive_gpu=False)
    assert n == 1 and devs == [1]


def test_shared_scripts_keep_latest_lease_on_same_card(monkeypatch):
    """Two non-exclusive scripts sharing one card: the card stays blocked for
    new non-exclusive tasks until the latest lease expires."""
    mgr = SessionManager(log_dir="/tmp/test-gpu-log")
    monkeypatch.setattr(
        mgr._gpu_monitor,
        "get_gpu_info",
        lambda: _gpu_info([(0, 0, 140000)]),
    )
    mgr._scripts["gpu_script_1"] = _running_script(
        gpu_devices=(0,), started_at=_started_ago(30), need_exclusive_gpu=False,
    )
    mgr._scripts["gpu_script_2"] = _running_script(
        session_id="gpu_backend_y", gpu_devices=(0,),
        started_at=_started_ago(5), need_exclusive_gpu=False,
    )

    n, devs = mgr._count_free_gpus(1, None, need_exclusive_gpu=False)
    assert n == 0 and devs == []  # newest lease (5 min ago) still blocks card 0


def test_submit_script_stores_need_exclusive_gpu(monkeypatch, tmp_path):
    """submit_script records the exclusivity flag on the ScriptInfo."""
    mgr = SessionManager(log_dir="/tmp/test-gpu-log")
    monkeypatch.setattr(
        mgr._gpu_monitor,
        "get_gpu_info",
        lambda: _gpu_info([(0, 0, 140000)]),
    )
    script = tmp_path / "echo.sh"
    script.write_text("#!/bin/bash\necho hi\n")

    req = ScriptSubmitRequest(
        command_path=str(script),
        timeout=0,
        run_in_background=True,
        gpu_nums=1,
        priority=ScriptPriority.NORMAL,
        need_exclusive_gpu=True,
    )
    resp = mgr.submit_script(req)
    assert mgr._scripts[resp.script_id].need_exclusive_gpu is True


# ── R2/M1+M2: quota 放置 (free_memory_mb >= max_mem_quota) ──────────────────

def test_count_free_gpus_respects_max_mem_quota(monkeypatch):
    """卡虽然 vram_usage<95%, 但剩余显存 < quota → 不放.

    R2: 35B job 被放到只剩 17MB-free 的卡上 OOM —— 放置谓词原来只认 vram% 不认
    绝对剩余显存。quota 后 free < quota 的卡直接被跳过。
    """
    mgr = SessionManager(log_dir="/tmp/test-gpu-log")
    # card0: used 130000/140000 → free 10000 MB, usage ~92.9% < 95% (老逻辑会放);
    # card1: 全新 140000 MB free.
    monkeypatch.setattr(
        mgr._gpu_monitor,
        "get_gpu_info",
        lambda: _gpu_info([(0, 130000, 140000), (1, 0, 140000)]),
    )

    # quota 50000MB: card0 free 10000 < 50000 → excluded; 请求 2 张只给 1 张 (card1)
    n, devs = mgr._count_free_gpus(2, 50000)
    assert n == 1 and devs == [1]

    # quota 8000MB: card0 free 10000 >= 8000 → 两张卡都可放
    n, devs = mgr._count_free_gpus(2, 8000)
    assert n == 2 and set(devs) == {0, 1}


# ── R2/M3: head-of-line blocking 修复 ───────────────────────────────────────

def test_next_startable_skips_blocked_head_of_queue(monkeypatch):
    """头部大任务放不下 (需 2 卡) 不再挡后面的小任务 (需 1 卡).

    R2/M3: 原来头部放不下就 break 饿死队列后面的任务; 现在继续扫描, 找第一个
    当前能放下的任务 (仍按优先级从高到低)。
    """
    mgr = SessionManager(log_dir="/tmp/test-gpu-log")
    monkeypatch.setattr(
        mgr._gpu_monitor,
        "get_gpu_info",
        lambda: _gpu_info([(0, 0, 140000)]),  # 只有 1 张卡
    )

    big = _running_script(gpu_devices=(), script_id="gpu_script_big")
    big.gpu_nums = 2
    big.status = ScriptStatus.QUEUED
    small = _running_script(gpu_devices=(), script_id="gpu_script_small")
    small.gpu_nums = 1
    small.status = ScriptStatus.QUEUED
    mgr._queue = ["gpu_script_big", "gpu_script_small"]
    mgr._scripts["gpu_script_big"] = big
    mgr._scripts["gpu_script_small"] = small

    pick = mgr._next_startable()
    assert pick is not None
    i, sid, free_cnt, devices = pick
    assert sid == "gpu_script_small"
    assert free_cnt == 1
    assert devices == [0]


def test_next_startable_none_when_nothing_fits(monkeypatch):
    """队列里所有任务当前都放不下 → None (不启动任何任务)."""
    mgr = SessionManager(log_dir="/tmp/test-gpu-log")
    monkeypatch.setattr(
        mgr._gpu_monitor,
        "get_gpu_info",
        lambda: _gpu_info([(0, 0, 140000)]),
    )
    a = _running_script(gpu_devices=(), script_id="gpu_script_a")
    a.gpu_nums = 2
    a.status = ScriptStatus.QUEUED
    mgr._queue = ["gpu_script_a"]
    mgr._scripts["gpu_script_a"] = a
    assert mgr._next_startable() is None


# ── R2/M4a: wrapper release 标记终结死 pipeline (释放 booking) ───────────────

def test_release_marker_sweeps_lingering_session(monkeypatch, tmp_path):
    """wrapper 已写 __RELEASE__ 标记但 tmux 会话还挂着 → 终结脚本释放 booking.

    R2/M4a: kill-script 只杀 wrapper、孤儿孙进程让会话残留时, 调度器不再把卡当
    永远被占 → 后续任务不被压到剩卡上 OOM。
    """
    mgr = SessionManager(log_dir="/tmp/test-gpu-log")
    # 会话还活着 (sampler cache 里有它) → 走 release-marker 终结路径
    mgr._alive_names = frozenset({"gpu_backend_x"})

    log = tmp_path / "gpu_script_test.log"
    log.write_text("training step 1\nEXIT_CODE=0\n__RELEASE__gpu_script_test__\n")
    s = _running_script()
    s.log_path = str(log)
    mgr._scripts["gpu_script_test"] = s

    out = mgr.get_script_status("gpu_script_test")
    assert out.status == ScriptStatus.COMPLETED
    assert out.return_code == 0
    assert mgr._booked_devices() == set()  # booking 释放


def test_release_marker_with_nonzero_exit_marks_failed(monkeypatch, tmp_path):
    """EXIT_CODE!=0 的 release 标记 → FAILED (而不是误报 COMPLETED)."""
    mgr = SessionManager(log_dir="/tmp/test-gpu-log")
    mgr._alive_names = frozenset({"gpu_backend_x"})

    log = tmp_path / "gpu_script_test.log"
    log.write_text("EXIT_CODE=6\n__RELEASE__gpu_script_test__\n")
    s = _running_script()
    s.log_path = str(log)
    mgr._scripts["gpu_script_test"] = s

    out = mgr.get_script_status("gpu_script_test")
    assert out.status == ScriptStatus.FAILED
    assert out.return_code == 6
    assert mgr._booked_devices() == set()


# ── R2/M4b+R7: kill 整进程组 ────────────────────────────────────────────────

def test_kill_script_kills_process_group(monkeypatch):
    """kill_script 对运行中脚本先整进程组 kill (R7), 再 kill-session.

    孤儿 vLLM worker 一并清掉, booking 立即释放 —— 否则卡被孤儿占满, 新任务照样
    OOM。
    """
    mgr = SessionManager(log_dir="/tmp/test-gpu-log")

    def fake_tmux(*args, **kwargs):
        if args and args[0] == "list-sessions":
            return subprocess.CompletedProcess(args, returncode=0, stdout="gpu_backend_x\n")
        if args and args[0] == "display":  # #{pane_pid}
            return subprocess.CompletedProcess(args, returncode=0, stdout="4242\n")
        return subprocess.CompletedProcess(args, returncode=0, stdout="")

    monkeypatch.setattr("server.session_manager._run_tmux", fake_tmux)
    monkeypatch.setattr("server.session_manager.os.getpgid", lambda pid: 4242)
    killed_groups: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "server.session_manager.os.killpg",
        lambda pgid, sig: killed_groups.append((pgid, sig)),
    )
    monkeypatch.setattr("server.session_manager.time.sleep", lambda _s: None)

    s = _running_script(need_exclusive_gpu=True)  # 独占任务, 死了必须释放
    mgr._scripts["gpu_script_test"] = s

    ok = mgr.kill_script("gpu_script_test")
    assert ok is True
    assert s.status == ScriptStatus.KILLED
    # SIGTERM + SIGKILL 都发给了 pane 的进程组
    assert killed_groups == [(4242, signal.SIGTERM), (4242, signal.SIGKILL)]
    assert mgr._booked_devices() == set()  # booking 立即释放


def test_kill_script_falls_back_when_pane_pid_unavailable(monkeypatch):
    """拿不到 pane PID (display 失败/不可解析) → 静默退回 kill-session."""
    mgr = SessionManager(log_dir="/tmp/test-gpu-log")
    called = {"pgroup": 0}

    def fake_tmux(*args, **kwargs):
        if args and args[0] == "list-sessions":
            return subprocess.CompletedProcess(args, returncode=0, stdout="gpu_backend_x\n")
        if args and args[0] == "display":
            return subprocess.CompletedProcess(args, returncode=0, stdout="not-a-pid\n")
        return subprocess.CompletedProcess(args, returncode=0, stdout="")

    monkeypatch.setattr("server.session_manager._run_tmux", fake_tmux)
    monkeypatch.setattr(
        "server.session_manager.os.killpg",
        lambda pgid, sig: called.__setitem__("pgroup", called["pgroup"] + 1),
    )
    monkeypatch.setattr("server.session_manager.time.sleep", lambda _s: None)

    s = _running_script()
    mgr._scripts["gpu_script_test"] = s

    ok = mgr.kill_script("gpu_script_test")
    assert ok is True
    assert s.status == ScriptStatus.KILLED
    assert called["pgroup"] == 0  # 没走 killpg, 退回 kill-session


# ── R9: gpu-backend 断联修复 (2026-08-26) ──────────────────────────────────
# 根因: 所有端点 async def 却在事件循环上同步跑阻塞调用 (exec_in_session 的
# 30min 忙等 / nvidia-smi / 整文件日志读) → 单事件循环冻结 → 客户端 ReadTimeout.
# 修复: 端点改线程池 (sync def) + 扫描器改独立守护线程 + RLock 防竞态.

def test_start_scanner_runs_on_daemon_thread(monkeypatch):
    """扫描器跑在独立守护线程 (而非 HTTP 事件循环), start 幂等.

    回归 (断联): 原来扫描器是 asyncio task, 每次扫描同步跑 nvidia-smi 冻结
    事件循环; 现在独立线程 + time.sleep(5), 慢 nvidia-smi 只卡调度线程.
    """
    mgr = SessionManager(log_dir="/tmp/test-gpu-log")
    gate = threading.Event()
    ran = {"n": 0}

    def fake_loop(self):
        ran["n"] += 1
        gate.wait(timeout=5)  # 卡住线程, 便于断言存活状态

    monkeypatch.setattr(SessionManager, "_scanner_loop", fake_loop)

    mgr.start_scanner()
    t = mgr._scanner_thread
    assert t is not None and t.is_alive()
    assert t.daemon is True                     # 不挡进程退出
    assert t.name == "gpu-vram-scanner"

    mgr.start_scanner()                          # 幂等 — 不重复起线程
    assert mgr._scanner_thread is t

    gate.set()
    t.join(timeout=2)
    assert not t.is_alive()
    assert ran["n"] == 1


def test_submit_script_works_without_event_loop(monkeypatch, tmp_path):
    """submit_script 不再需要运行中的事件循环 (断联修复: 端点线程池化后无 loop).

    回归: 原来 submit 里 ``asyncio.get_running_loop()`` + create_task 起扫描器,
    线程池里没有事件循环会 RuntimeError. 现在扫描器由 lifespan 起, submit 干净排队.
    """
    mgr = SessionManager(log_dir="/tmp/test-gpu-log")
    script = tmp_path / "echo.sh"
    script.write_text("#!/bin/bash\necho hi\n")
    resp = mgr.submit_script(ScriptSubmitRequest(
        command_path=str(script),
        timeout=0,
        run_in_background=True,
        gpu_nums=1,
        priority=ScriptPriority.HIGH,
        need_exclusive_gpu=True,
    ))
    assert resp.status == ScriptStatus.QUEUED
    assert resp.script_id in mgr._scripts
    assert mgr._scanner_thread is None  # 不在 submit 里起线程


def test_get_script_log_returns_tail(monkeypatch, tmp_path):
    """/log 只读日志尾部, 不整文件读 — 多 GB 训练日志不拖垮 worker 线程."""
    log = tmp_path / "train.log"
    log.write_text("HEAD_MARKER\n" + ("x" * (300 * 1024)) + "TAIL_MARKER\n")
    mgr = SessionManager(log_dir=tmp_path)
    s = _running_script(script_id="gpu_script_log")
    s.log_path = str(log)
    mgr._scripts[s.script_id] = s

    out = mgr.get_script_log("gpu_script_log").stdout
    assert "TAIL_MARKER" in out
    assert "HEAD_MARKER" not in out  # 头在 256KB 窗口外 → 被截掉, 证明是尾部读


# ── R11: 调度审计日志 (2026-08-31 HLE 0830 OOM 复盘) ─────────────────────────
# 主日志之前「被吞掉」: 误终结 RUNNING 脚本 (tmux 缓存空) 不落任何日志, 独占 booking
# 静默释放 → scanner 放新实验上卡 → 节点 OOM。以下测试锁住审计日志不变式:
#   ① 每次扫描打 SCAN state 快照; ② 放行新实验打 RELEASE 依据 (freed_this_scan +
#   prior_holders); ③ 无 EXIT_CODE 强制 COMPLETED 必须 WARNING; ④ sampler 缓存
#   非空→空跳变必须 WARNING。

def test_scanner_iter_logs_scan_state_and_release_basis(monkeypatch, tmp_path, caplog):
    """一次扫描 (sweep 终结旧脚本释放 booking → 放行新实验) 必须留下审计日志."""
    mgr = SessionManager(log_dir="/tmp/test-gpu-log")
    monkeypatch.setattr(
        mgr._gpu_monitor, "get_gpu_info",
        lambda: _gpu_info([(0, 0, 140000), (1, 0, 140000)]),
    )
    # 旧脚本独占卡 0, 会话从 sampler 缓存消失 → sweep 终结并释放 booking
    mgr._alive_names = frozenset({"other_session"})
    old_log = tmp_path / "gpu_script_old.log"
    old_log.write_text("training step 1\nEXIT_CODE=0\n__RELEASE__gpu_script_old__\n")
    old = _running_script(
        gpu_devices=(0,), script_id="gpu_script_old", session_id="gpu_backend_old",
    )
    old.log_path = str(old_log)
    mgr._scripts["gpu_script_old"] = old
    # 排队的独占新任务, 需 1 卡 (sweep 后卡 0 释放 → 可放行)
    new = _running_script(
        gpu_devices=(), script_id="gpu_script_new", session_id=None,
    )
    new.status = ScriptStatus.QUEUED
    new.gpu_nums = 1
    new.need_exclusive_gpu = True
    mgr._scripts["gpu_script_new"] = new
    mgr._queue = ["gpu_script_new"]

    started: list[str] = []
    monkeypatch.setattr(mgr, "_start_script", lambda sid, info: started.append(sid))

    caplog.set_level(logging.INFO)
    ok = mgr._scanner_iter()
    assert ok is True
    assert started == ["gpu_script_new"]

    # ① SCAN state 快照: 含 running / finalized / leases / free_excl / pick / 采样器健康
    assert "SCAN state:" in caplog.text
    assert "finalized=gpu_script_old=completed" in caplog.text
    assert "free_excl=2/2" in caplog.text
    assert "pick=[gpu_script_new → GPUs" in caplog.text
    assert "sampler_healthy=False" in caplog.text  # 本测试没喂采样 → 判定层视为未知
    # ② RELEASE 依据: freed_this_scan + prior_holders (启动前租约已空)
    assert "RELEASE gpu_script_new" in caplog.text
    assert "freed_this_scan=gpu_script_old=completed" in caplog.text
    assert "prior_holders={}" in caplog.text
    # ③ 终结了旧脚本 → 会话消失但有权威标记 (EXIT_CODE/release) → 无条件终结
    assert "authoritative marker/EXIT_CODE in log" in caplog.text
    assert "finalized completed (exit_code=0" in caplog.text


def test_finalize_without_exit_code_warns_unconfirmed(monkeypatch, tmp_path, caplog):
    """危险路径: 会话消失 + 日志尾无 EXIT_CODE= → 连续确认后强制 COMPLETED 必须 WARNING.

    这正是 HLE 0830 OOM 的根因形状 —— 仍在训练的脚本被误判完成、booking 未确认释放。
    降级防护后不再单次缺失就终结: 采样器健康 + 连续 N 次确认才放行, 且日志亮 WARNING,
    不许静默。
    """
    mgr = SessionManager(log_dir="/tmp/test-gpu-log")
    mgr._alive_names = frozenset({"other_session"})  # 本脚本会话「看起来」死了
    mgr._last_sample_ok_ts = time.time()   # 采样器健康 → 走确认路径
    mgr._sampler_degraded = False
    log = tmp_path / "gpu_script_test.log"
    log.write_text("training step 1\nstill training, no EXIT_CODE yet\n")
    s = _running_script()
    s.log_path = str(log)
    mgr._scripts["gpu_script_test"] = s

    caplog.set_level(logging.INFO)
    out = mgr.get_script_status("gpu_script_test")
    assert out.status == ScriptStatus.RUNNING   # confirm 1/2 → 还守着 booking
    assert "confirm 1/2" in caplog.text
    assert mgr._booked_devices() == {0}

    out = mgr.get_script_status("gpu_script_test")
    assert out.status == ScriptStatus.COMPLETED  # confirm 2/2 → 强制 COMPLETED
    assert "no EXIT_CODE in log tail" in caplog.text
    assert "booking released UNCONFIRMED" in caplog.text
    assert mgr._booked_devices() == set()


def test_tmux_sampler_empty_transition_warns(monkeypatch, caplog):
    """tmux sampler 健康采样下缓存非空→空 必须 WARNING 标注「authoritative」.

    唯一的合法置空途径 (tmux 真的没会话了) —— 后续会按确认计数终结; 健康采样
    下这是权威的, 与降级时的「不知道」语义区分开。
    """
    mgr = SessionManager(log_dir="/tmp/test-gpu-log")
    mgr._alive_names = frozenset({"gpu_backend_x", "gpu_backend_y"})

    monkeypatch.setattr(
        "server.session_manager._run_tmux",
        lambda *a, **k: subprocess.CompletedProcess(a, returncode=0, stdout="", stderr=""),
    )
    caplog.set_level(logging.WARNING)
    mgr._refresh_alive_names()
    assert mgr._alive_names == frozenset()
    assert mgr._sampler_degraded is False
    assert "confirmed EMPTY" in caplog.text
    assert "is authoritative" in caplog.text


# ── R11b: tmux sampler 降级防护 + 判定日志落盘 (2026-08-31 复盘) ───────────────
# HLE 0830 OOM 根因链: tmux 卡死 → HUNG 折叠成空缓存 → 所有 RUNNING 脚本被误终结
# → 独占 booking 静默全释放 → scanner 放新 35B 任务上卡 → 节点 RAM OOM。
# 修复后的判定不变式:
#   ① 采样失败/卡死 → 保持上一份 alive 缓存 + 标记降级 (绝不折叠成空集);
#   ② 会话消失时先看权威信号 (release 标记 / EXIT_CODE); 没有 → 采样器不健康就
#      HOLDING booking 绝不终结; 健康则连续 SESSION_LOSS_CONFIRM_COUNT 次确认才终结;
#   ③ 所有判定 (SCAN state / RELEASE / 降级 / 终结门) 追加写 <log_dir>/scheduler.log
#      (log_dir 由 GPU_BACKEND_LOG_DIR/--log-dir 路由到 run-dir NFS, 落盘持久)。

def test_sampler_degraded_keeps_previous_cache(monkeypatch, caplog):
    """tmux HUNG/失败 → 保持上一份 alive 缓存 + 标记降级 (绝不折叠成空集).

    这正是 HLE 0830 的触发点 —— 旧实现把 HUNG (rc=-1, 空输出) 和「真的一无所有」
    折叠成空集, 让所有 RUNNING 脚本「看起来全死」。降级后判定层必须知道「不知道」。
    """
    mgr = SessionManager(log_dir="/tmp/test-gpu-log")
    mgr._alive_names = frozenset({"gpu_backend_x", "gpu_backend_y"})

    monkeypatch.setattr(
        "server.session_manager._run_tmux",
        lambda *a, **k: HUNG,
    )
    caplog.set_level(logging.WARNING)
    names = mgr._refresh_alive_names()
    assert names == frozenset({"gpu_backend_x", "gpu_backend_y"})  # 缓存原样保留
    assert mgr._alive_names == frozenset({"gpu_backend_x", "gpu_backend_y"})
    assert mgr._sampler_degraded is True
    assert "DEGRADED" in caplog.text
    assert "SUSPENDED" in caplog.text
    assert "keeping previous alive set" in caplog.text


def test_degraded_sampler_blocks_new_init(monkeypatch, tmp_path):
    """降级场景全链路 (用户指定模拟): tmux 卡死 → 训练脚本会话「消失」但采样器降级
    → booking 必须守住 (训练继续 RUNNING) → 排队的新 35B 任务拿不到卡 → 不放行;
    且每一步判定都落盘 <log_dir>/scheduler.log。

    等价于 HLE 0830 的节点 OOM 保护 —— 只要 sampler 降级, 误终结链必须被切断,
    新 init 进不来。
    """
    mgr = SessionManager(log_dir=str(tmp_path))   # 独立 log_dir → 断言只属于本测试
    monkeypatch.setattr(
        mgr._gpu_monitor, "get_gpu_info",
        lambda: _gpu_info([(0, 0, 140000), (1, 0, 140000)]),
    )
    # 降级采样器: 从未成功采到过, 缓存空 (看起来所有会话都「没了」)
    mgr._sampler_degraded = True
    mgr._last_sample_ok_ts = None
    mgr._alive_names = frozenset()

    # 训练中的独占脚本持卡 0,1; 日志尾没有 EXIT_CODE/release 标记 (还在训练)
    train_log = tmp_path / "gpu_script_train.log"
    train_log.write_text("training step 1\n")
    train = _running_script(
        gpu_devices=(0, 1), script_id="gpu_script_train",
        session_id="gpu_backend_train", need_exclusive_gpu=True,
    )
    train.log_path = str(train_log)
    mgr._scripts["gpu_script_train"] = train

    # 排队的新任务需 2 卡 (匹配训练脚本持有的卡) → 卡没被误释放就必然放不下
    new = _running_script(
        gpu_devices=(), script_id="gpu_script_new", session_id=None,
    )
    new.status = ScriptStatus.QUEUED
    new.gpu_nums = 2
    new.need_exclusive_gpu = True
    mgr._scripts["gpu_script_new"] = new
    mgr._queue = ["gpu_script_new"]

    started: list[str] = []
    monkeypatch.setattr(mgr, "_start_script", lambda sid, info: started.append(sid))

    ok = mgr._scanner_iter()

    # ── 正确识别降级 → 新 init 被挡 ──
    assert ok is False
    assert started == []                          # 没有放行任何新任务
    assert mgr._scripts["gpu_script_train"].status == ScriptStatus.RUNNING  # 训练没被误终结
    assert mgr._booked_devices() == {0, 1}        # 独占 booking 全部守住
    assert mgr._scripts["gpu_script_new"].status == ScriptStatus.QUEUED     # 新任务仍在排队

    # ── 判定日志正确落盘 ──
    log_text = (tmp_path / "scheduler.log").read_text()
    assert "degraded-sample guard" in log_text      # 终结门: 降级守护
    assert "HOLDING booking" in log_text            # 守 booking 不动
    assert "sampler_healthy=False" in log_text      # SCAN state: 采样器视角
    assert "pick=[nothing fits]" in log_text        # 放不下 → 不放行
    assert "free_excl=0/2" in log_text              # 卡全被持有, 无干净卡


def test_audit_log_written_to_disk(monkeypatch, tmp_path):
    """正常放行路径的判定日志也必须落盘: SCAN state / RELEASE / finalize 写 scheduler.log."""
    mgr = SessionManager(log_dir=str(tmp_path))
    monkeypatch.setattr(
        mgr._gpu_monitor, "get_gpu_info",
        lambda: _gpu_info([(0, 0, 140000), (1, 0, 140000)]),
    )
    # 旧脚本会话消失 + 日志有权威 EXIT_CODE/release 标记 → 无条件终结
    mgr._alive_names = frozenset({"other_session"})
    old_log = tmp_path / "gpu_script_old.log"
    old_log.write_text("training step 1\nEXIT_CODE=0\n__RELEASE__gpu_script_old__\n")
    old = _running_script(
        gpu_devices=(0,), script_id="gpu_script_old", session_id="gpu_backend_old",
    )
    old.log_path = str(old_log)
    mgr._scripts["gpu_script_old"] = old
    # 排队的独占新任务, 需 1 卡 (旧脚本释放后卡 0 空出)
    new = _running_script(
        gpu_devices=(), script_id="gpu_script_new", session_id=None,
    )
    new.status = ScriptStatus.QUEUED
    new.gpu_nums = 1
    new.need_exclusive_gpu = True
    mgr._scripts["gpu_script_new"] = new
    mgr._queue = ["gpu_script_new"]

    started: list[str] = []
    monkeypatch.setattr(mgr, "_start_script", lambda sid, info: started.append(sid))

    ok = mgr._scanner_iter()
    assert ok is True
    assert started == ["gpu_script_new"]

    log_text = (tmp_path / "scheduler.log").read_text()
    assert "SCAN state:" in log_text
    assert "finalized=gpu_script_old=completed" in log_text
    assert "authoritative marker/EXIT_CODE in log" in log_text
    assert "RELEASE gpu_script_new" in log_text
    assert "freed_this_scan=gpu_script_old=completed" in log_text
    assert "finalized completed (exit_code=0" in log_text


def test_sampler_recovery_releases_dead_script(monkeypatch, tmp_path):
    """降级期间会话消失 → 守 booking; 采样器恢复 + 连续确认 → 终结并释放.

    验证恢复语义: 降级时累计的缺失不「秋后算账」, 恢复后从零开始计
    SESSION_LOSS_CONFIRM_COUNT 次健康确认 —— 宁慢勿误放。
    """
    mgr = SessionManager(log_dir=str(tmp_path))
    log = tmp_path / "gpu_script_test.log"
    log.write_text("training step 1\n")   # 无 EXIT_CODE/marker → 只能走确认路径
    s = _running_script()
    s.log_path = str(log)
    mgr._scripts["gpu_script_test"] = s

    # 降级: 会话消失 + 采样器不健康 → 守 booking
    mgr._alive_names = frozenset({"other_session"})
    mgr._sampler_degraded = True
    mgr._last_sample_ok_ts = None
    out = mgr.get_script_status("gpu_script_test")
    assert out.status == ScriptStatus.RUNNING
    assert mgr._booked_devices() == {0}

    # 采样器恢复健康, 会话仍消失 → confirm 1/2, 不终结
    mgr._sampler_degraded = False
    mgr._last_sample_ok_ts = time.time()
    out = mgr.get_script_status("gpu_script_test")
    assert out.status == ScriptStatus.RUNNING
    assert mgr._booked_devices() == {0}

    # confirm 2/2 → 终结并释放 booking
    out = mgr.get_script_status("gpu_script_test")
    assert out.status == ScriptStatus.COMPLETED
    assert mgr._booked_devices() == set()

    log_text = (tmp_path / "scheduler.log").read_text()
    assert "HOLDING booking" in log_text
    assert "confirmed 2 healthy samples" in log_text


def test_finished_script_releases_even_when_sampler_degraded(monkeypatch, tmp_path):
    """实验正常结束 (日志有权威 EXIT_CODE/release 标记) → 即使采样器降级也立即释放.

    反向担忧的边界 (用户 2026-08-31): 降级防护会不会把「真结束了」的 35B 也拦在外
    面、卡不释放? 不会 —— 终结门里权威信号优先级最高 (在采样器健康检查之前), 日志
    文件制不依赖 tmux, wrapper 的 EXIT trap 已写标记 → 无条件终结释放。只有「没有
    标记 + 降级」才守卡 —— 那种状态本来就该守 (无法区分真结束与卡死中)。
    """
    mgr = SessionManager(log_dir=str(tmp_path))
    mgr._sampler_degraded = True     # tmux 卡死 (最坏情况)
    mgr._last_sample_ok_ts = None
    mgr._alive_names = frozenset()   # 会话从缓存消失

    log = tmp_path / "gpu_script_done.log"
    log.write_text("eval score: 0.92\nEXIT_CODE=0\n__RELEASE__gpu_script_done__\n")
    s = _running_script(
        gpu_devices=(0, 1), script_id="gpu_script_done",
        session_id="gpu_backend_done", need_exclusive_gpu=True,
    )
    s.log_path = str(log)
    mgr._scripts["gpu_script_done"] = s

    out = mgr.get_script_status("gpu_script_done")
    assert out.status == ScriptStatus.COMPLETED
    assert mgr._booked_devices() == set()   # 卡立即释放, 不挡新实验

    log_text = (tmp_path / "scheduler.log").read_text()
    assert "authoritative marker/EXIT_CODE in log" in log_text
