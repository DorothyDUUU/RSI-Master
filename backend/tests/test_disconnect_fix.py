"""Hermetic tests for the R10 disconnect fix (2026-08-27).

Root cause found in the 0826 batch: ``_scanner_loop`` held ``_state_lock``
across ``nvidia-smi`` / ``tmux`` subprocess calls.  When the NVIDIA driver
wedged (D-state child), ``subprocess.run(timeout=10)`` could not kill it —
SIGKILL is deferred until the kernel unblocks — so the lock was pinned for
minutes, every lock-taking endpoint (/health included) queued, and the probe
declared the pod DOWN.

The fix has five parts, each tested here:
1. ``subproc.run_hard`` — independent wall clock that abandons a D-state child.
2. ``GPUMonitor`` background snapshot cache — getters never spawn a subprocess.
3. SessionManager tmux session-set sampler cache — liveness reads memory only.
4. ``_scanner_loop`` launches scripts OUTSIDE the lock.
5. ``/health`` is O(1) and stays up while nvidia-smi/tmux are wedged.

Run::

    cd backend && ../.venv/bin/python -m pytest tests/test_disconnect_fix.py -q
"""

from __future__ import annotations

import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import pytest  # noqa: E402

from server.models import (  # noqa: E402
    GPUDevice,
    GPUInfo,
    ScriptInfo,
    ScriptPriority,
    ScriptStatus,
)
from server.session_manager import SessionManager  # noqa: E402
from server.subproc import HUNG, run_hard  # noqa: E402


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
    """Fake GPUInfo from (device_id, used_mb, total_mb) tuples."""
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


def _running_script(script_id: str = "gpu_script_test",
                    session_id: str = "gpu_backend_x") -> ScriptInfo:
    return ScriptInfo(
        script_id=script_id,
        command_path="/tmp/nonexistent.sh",
        status=ScriptStatus.RUNNING,
        session_id=session_id,
        gpu_devices=[0],
        priority=ScriptPriority.HIGH,
        gpu_nums=1,
        started_at=datetime.now(timezone.utc),
    )


# ── 1. run_hard: D-state hard wall ──────────────────────────────────────────

def test_run_hard_returns_hung_within_wall():
    """A subprocess that outlives the wall clock returns HUNG fast, instead of
    blocking the caller until the (unkillable) child exits."""
    start = time.monotonic()
    res = run_hard(["sleep", "2"], timeout=10, wall_ms=300)
    elapsed = time.monotonic() - start
    assert res is HUNG
    assert elapsed < 1.5, f"run_hard blocked {elapsed:.2f}s instead of ~0.3s wall"


def test_run_hard_returns_real_result_on_success():
    res = run_hard(["echo", "ok"], timeout=5, wall_ms=2000)
    assert res is not HUNG
    assert res.returncode == 0
    assert res.stdout.strip() == "ok"


def test_run_hard_caps_leaked_threads():
    """With MAX_IN_FLIGHT wedged threads, further calls return HUNG without
    spawning yet another subprocess."""
    import server.subproc as sp
    # Simulate the cap being reached without actually leaking processes.
    with sp._in_flight_lock:
        sp._in_flight = sp.MAX_IN_FLIGHT
    try:
        res = run_hard(["echo", "x"], timeout=5, wall_ms=500)
        assert res is HUNG
    finally:
        with sp._in_flight_lock:
            sp._in_flight = 0


# ── 2. GPUMonitor: background snapshot cache ────────────────────────────────

def test_gpu_monitor_get_gpu_info_uses_cache_no_subprocess(monkeypatch):
    """Once a snapshot exists, getters return it without spawning a subprocess —
    the scheduler's lock path never touches nvidia-smi."""
    from server.gpu_monitor import GPUMonitor

    monkeypatch.setattr(
        "server.gpu_monitor.GPUMonitor._check_nvidia_smi", lambda self: True
    )
    g = GPUMonitor()
    g._cache = _gpu_info([(0, 0, 140000), (1, 0, 140000)])
    g._cache_ts = time.time()
    g._last_refresh_ok = True

    calls = {"n": 0}
    monkeypatch.setattr(
        "server.gpu_monitor.run_hard",
        lambda *a, **k: calls.__setitem__("n", calls["n"] + 1)
        or subprocess.CompletedProcess([], returncode=0, stdout=""),
    )
    info = g.get_gpu_info()
    assert info.total_devices == 2
    assert calls["n"] == 0, "get_gpu_info must not run nvidia-smi when cached"


def test_gpu_monitor_stale_snapshot_advertises_nothing(monkeypatch):
    """A wedged driver (last refresh failed) → no free GPUs advertised, and
    check_resources reports degraded — but never blocks."""
    from server.gpu_monitor import GPUMonitor

    monkeypatch.setattr(
        "server.gpu_monitor.GPUMonitor._check_nvidia_smi", lambda self: True
    )
    g = GPUMonitor()
    g._cache = _gpu_info([(0, 0, 140000)])
    g._cache_ts = time.time() - 300  # stale
    g._last_refresh_ok = False  # wedged

    info = g.get_gpu_info()
    assert info.devices == []
    assert info.available_devices == []

    assert g.snapshot_stale() is True
    health = g.check_resources()
    assert health["healthy"] is False
    assert any("stale" in issue for issue in health["issues"])


def test_gpu_monitor_no_gpu_never_stale():
    """A machine without nvidia-smi is never 'stale' — empty is honest."""
    from server.gpu_monitor import GPUMonitor
    g = GPUMonitor()  # _check_nvidia_smi patched False by fixture
    g._last_refresh_ok = False
    assert g.snapshot_stale() is False
    assert g.check_resources()["healthy"] is True


# ── 3. tmux session-set sampler cache ───────────────────────────────────────

def test_status_and_list_sessions_never_call_tmux(monkeypatch):
    """Liveness reads the sampled session set — no tmux subprocess under lock."""
    mgr = SessionManager(log_dir="/tmp/test-gpu-log")
    calls = {"n": 0}
    monkeypatch.setattr(
        "server.session_manager._run_tmux",
        lambda *a, **k: calls.__setitem__("n", calls["n"] + 1)
        or subprocess.CompletedProcess(["tmux"], returncode=0, stdout=""),
    )
    mgr._alive_names = frozenset({"gpu_backend_x"})
    s = _running_script()
    mgr._scripts[s.script_id] = s

    out = mgr.get_script_status(s.script_id)
    assert out.status == ScriptStatus.RUNNING
    assert len(mgr.list_sessions()) == 1
    assert calls["n"] == 0, "no tmux subprocess on the lock path"


def test_status_holds_booking_when_session_absent_and_sampler_unknown():
    """会话从缓存消失 + 采样器从未成功采到过 (wedged/未知) → 守 booking, 不终结.

    2026-08-31 降级防护 (HLE 0830 OOM 复盘): tmux 卡死/从未采样时, 会话缺失不
    代表脚本真死了 —— 误终结会让独占 booking 静默释放 → 新 35B 任务上卡 → 节点
    OOM。此时必须 HOLDING, 只信权威信号 (release 标记 / EXIT_CODE)。
    """
    mgr = SessionManager(log_dir="/tmp/test-gpu-log")
    mgr._alive_names = frozenset({"other_session"})
    mgr._last_sample_ok_ts = None    # sampler never ok → 判定层视为「不知道」
    mgr._sampler_degraded = False
    s = _running_script()
    mgr._scripts[s.script_id] = s

    out = mgr.get_script_status(s.script_id)
    assert out.status == ScriptStatus.RUNNING       # 保持 RUNNING
    assert mgr._booked_devices() == {0}             # booking 守住


def test_status_finalizes_when_session_absent_sampler_healthy_confirmed():
    """会话消失 + 采样器健康 + 连续 SESSION_LOSS_CONFIRM_COUNT 次确认 → 终结释放.

    采样器健康意味着「会话不在」是可信信息, 连续 N 次确认后才释放 booking —— 单次
    抖动 (tmux 偶发没列全) 不误放。
    """
    mgr = SessionManager(log_dir="/tmp/test-gpu-log")
    mgr._alive_names = frozenset({"other_session"})
    mgr._last_sample_ok_ts = time.time()    # 采样器健康
    mgr._sampler_degraded = False
    s = _running_script()
    mgr._scripts[s.script_id] = s

    out = mgr.get_script_status(s.script_id)
    assert out.status == ScriptStatus.RUNNING       # confirm 1/N → 还守着
    out = mgr.get_script_status(s.script_id)
    assert out.status == ScriptStatus.COMPLETED     # confirm N/N → 终结
    assert mgr._booked_devices() == set()


# ── 4. _scanner_loop launches outside the lock ──────────────────────────────

def test_start_script_marks_failed_on_hung_tmux_and_releases_lock(monkeypatch, tmp_path):
    """A wedged tmux (HUNG) on launch → script FAILED, removed from queue, and
    crucially the state lock is released (other endpoints can still acquire it)."""
    mgr = SessionManager(log_dir=tmp_path)
    script = tmp_path / "work.sh"
    script.write_text("#!/bin/bash\necho hi\n")
    info = ScriptInfo(
        script_id="gpu_script_t",
        command_path=str(script),
        status=ScriptStatus.QUEUED,
        gpu_nums=1,
        priority=ScriptPriority.NORMAL,
        gpu_devices=[0],
        started_at=datetime.now(timezone.utc),
    )
    mgr._scripts["gpu_script_t"] = info
    mgr._queue = ["gpu_script_t"]

    monkeypatch.setattr("server.session_manager._run_tmux", lambda *a, **k: HUNG)
    mgr._start_script("gpu_script_t", info)

    assert info.status == ScriptStatus.FAILED
    assert "gpu_script_t" not in mgr._queue
    assert mgr._booked_devices() == set()  # FAILED → no booking held

    # The state lock is free — another thread acquires it immediately.
    acquired = threading.Event()

    def _try_lock():
        with mgr._state_lock:
            acquired.set()

    t = threading.Thread(target=_try_lock)
    t.start()
    assert acquired.wait(timeout=1), "state lock still held after _start_script"
    t.join(timeout=1)


def test_scanner_launch_is_outside_lock(monkeypatch):
    """The scanner does its pick under the lock but launches the script after
    releasing it — a probe thread can acquire the lock while _start_script runs
    (so a HUNG tmux there cannot block /health)."""
    mgr = SessionManager(log_dir="/tmp/test-gpu-log")
    monkeypatch.setattr(
        mgr._gpu_monitor,
        "get_gpu_info",
        lambda: _gpu_info([(0, 0, 140000)]),
    )
    mgr._scripts["gpu_script_q"] = ScriptInfo(
        script_id="gpu_script_q",
        command_path="/tmp/never-submitted.sh",
        status=ScriptStatus.QUEUED,
        gpu_nums=1,
        priority=ScriptPriority.NORMAL,
    )
    mgr._queue = ["gpu_script_q"]

    lock_free_during_launch = {"v": False}

    def fake_start(self, script_id, info):
        # _scanner_iter calls this AFTER releasing the lock.  If the launch ran
        # under the lock, the probe thread below would block and time out.
        # (Class-attr patch → bound as a method, hence the `self`.)
        acquired = threading.Event()

        def _probe():
            with mgr._state_lock:
                lock_free_during_launch["v"] = True
                acquired.set()

        threading.Thread(target=_probe).start()
        acquired.wait(timeout=1)

    monkeypatch.setattr(SessionManager, "_start_script", fake_start)
    launched = mgr._scanner_iter()
    assert launched is True
    assert lock_free_during_launch["v"] is True


# ── 5. /health stays up under a wedged driver ───────────────────────────────

def test_health_stays_up_when_nvidia_smi_and_tmux_wedge(monkeypatch):
    """The integration guarantee: with nvidia-smi and tmux both wedged (HUNG),
    /health still answers within its probe timeout, degraded but alive."""
    from fastapi.testclient import TestClient
    from server.main import app

    monkeypatch.setattr(
        "server.gpu_monitor.GPUMonitor._check_nvidia_smi", lambda self: True
    )
    monkeypatch.setattr("server.gpu_monitor.run_hard",
                        lambda *a, **k: HUNG)
    monkeypatch.setattr("server.session_manager._run_tmux",
                        lambda *a, **k: HUNG)

    start = time.monotonic()
    with TestClient(app) as client:
        resp = client.get("/health", timeout=5)
        elapsed = time.monotonic() - start
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("healthy", "degraded")
    assert "resources" in data
    assert "active_sessions" in data
    # Never-sampled snapshot age must stay JSON-safe (no inf/NaN).
    assert isinstance(data["gpu_snapshot_age_s"], (int, float))
    import json
    json.dumps(data)  # must not raise on the probe payload
    assert elapsed < 3, f"/health took {elapsed:.2f}s under a wedged driver"
