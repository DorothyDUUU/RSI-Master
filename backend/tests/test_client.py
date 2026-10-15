"""Unit / integration tests for BackendClient.

These tests exercise the client against a live server.  If no server is running
they will fail with connection errors (intentional — this validates the full
client→server path).

Run::

    BACKEND_URL=http://localhost:8989 python -m backend.tests.test_client
"""

from __future__ import annotations

import os
import sys
import tempfile

from backend.gpu_backend.client import BackendClient, BackendError


def get_client() -> BackendClient:
    url = os.environ.get("BACKEND_URL", "http://localhost:8989")
    return BackendClient(base_url=url)


# ── Health / resources ─────────────────────────────────────────────────────

def test_health():
    with get_client() as c:
        data = c.health_check()
        assert data["status"] in ("healthy", "degraded")


def test_resources():
    with get_client() as c:
        info = c.get_resources()
        assert info.total_devices >= 0
        assert hasattr(info, "cpu_percent")


# ── Shell sessions ─────────────────────────────────────────────────────────

def test_exec_new_session():
    with get_client() as c:
        resp = c.exec_command("echo hello_client && pwd", timeout=10)
        assert resp.is_new
        assert resp.session_id.startswith("gpu_shell_")
        assert "hello_client" in resp.stdout
        assert resp.cwd is not None


def test_exec_existing_session():
    with get_client() as c:
        resp1 = c.exec_command("echo first", timeout=10)
        sid = resp1.session_id

        resp2 = c.exec_command("echo second", timeout=10, session_id=sid)
        assert not resp2.is_new
        assert "second" in resp2.stdout


def test_list_and_kill_session():
    with get_client() as c:
        resp = c.exec_command("echo test", timeout=10)
        sid = resp.session_id

        sessions = c.list_sessions()
        assert any(s.session_id == sid for s in sessions)

        assert c.kill_session(sid) is True
        # Already killed
        assert c.kill_session(sid) is False


# ── Scripts ────────────────────────────────────────────────────────────────

def test_submit_foreground():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write("#!/bin/bash\necho 'client_fg_ok'\n")
        path = f.name
    try:
        with get_client() as c:
            resp = c.submit_script(path, timeout=10, run_in_background=False)
            assert resp.status in ("completed", "failed")
            assert "client_fg_ok" in resp.stdout
    finally:
        os.unlink(path)


def test_submit_background():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write("#!/bin/bash\necho 'client_bg_ok'\nsleep 2\n")
        path = f.name
    try:
        with get_client() as c:
            resp = c.submit_script(path, timeout=10, run_in_background=True)
            assert resp.status == "running"
            assert resp.session_id is not None
            assert resp.log_path is not None
    finally:
        os.unlink(path)


def test_script_lifecycle():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write("#!/bin/bash\necho 'lifecycle_test'\n")
        path = f.name
    try:
        with get_client() as c:
            resp = c.submit_script(path, timeout=10, run_in_background=False)
            sid = resp.script_id

            # Status
            status = c.get_script_status(sid)
            assert status is not None
            assert status["script_id"] == sid

            # Log
            log = c.get_script_log(sid)
            assert log is not None
            assert "lifecycle_test" in log.stdout
    finally:
        os.unlink(path)


def test_script_not_found():
    with get_client() as c:
        resp = c.submit_script("/tmp/nonexistent_xyz.sh", timeout=10)
        assert resp.status == "failed"


def test_script_kill():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write("#!/bin/bash\nsleep 300\n")
        path = f.name
    try:
        with get_client() as c:
            resp = c.submit_script(path, timeout=10, run_in_background=True)
            assert c.kill_script(resp.script_id) is True
    finally:
        os.unlink(path)


# ── Status ─────────────────────────────────────────────────────────────────

def test_status():
    with get_client() as c:
        data = c.get_status()
        assert "sessions" in data
        assert "gpu_info" in data


# ── Runner ──────────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_health,
    test_resources,
    test_exec_new_session,
    test_exec_existing_session,
    test_list_and_kill_session,
    test_submit_foreground,
    test_submit_background,
    test_script_lifecycle,
    test_script_not_found,
    test_script_kill,
    test_status,
]


if __name__ == "__main__":
    print(f"Running {len(ALL_TESTS)} client tests\n" + "=" * 60)
    passed = 0
    for fn in ALL_TESTS:
        label = fn.__name__
        try:
            fn()
            print(f"[{label}] ✓ PASS")
            passed += 1
        except Exception as exc:
            print(f"[{label}] ✗ FAIL: {exc}")
            import traceback
            traceback.print_exc()
    print("=" * 60)
    print(f"Results: {passed}/{len(ALL_TESTS)} passed")
    sys.exit(0 if passed == len(ALL_TESTS) else 1)
