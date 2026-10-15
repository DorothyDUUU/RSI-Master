"""Integration tests for the GPU backend server.

Run against a local server::

    python -m backend.tests.test_server --url http://localhost:8989
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import httpx


def _test(name: str, url: str, fn) -> bool:
    print(f"[{name}]", end=" ", flush=True)
    try:
        fn(url)
        print("✓ PASS")
        return True
    except Exception as exc:
        print(f"✗ FAIL: {exc}")
        import traceback
        traceback.print_exc()
        return False


# ── Health ──────────────────────────────────────────────────────────────────

def test_health(url: str) -> None:
    with httpx.Client(timeout=10) as c:
        resp = c.get(f"{url}/health")
        resp.raise_for_status()
        data = resp.json()
        assert data["status"] in ("healthy", "degraded"), f"unexpected status: {data['status']}"
        assert "resources" in data
        assert "active_sessions" in data


# ── Resources ───────────────────────────────────────────────────────────────

def test_resources(url: str) -> None:
    with httpx.Client(timeout=10) as c:
        resp = c.get(f"{url}/resources")
        resp.raise_for_status()
        data = resp.json()
        assert "devices" in data
        assert "total_devices" in data
        assert "cpu_percent" in data


# ── Sessions ────────────────────────────────────────────────────────────────

def test_session_create_and_exec(url: str) -> None:
    """POST /sessions/exec without session_id → creates new session."""
    with httpx.Client(timeout=10) as c:
        resp = c.post(f"{url}/sessions/exec", json={
            "command": "echo hello_test && pwd",
            "timeout": 10,
        })
        resp.raise_for_status()
        data = resp.json()
        assert data["is_new"] is True
        assert data["session_id"].startswith("gpu_shell_")
        assert "hello_test" in data["stdout"]
        assert data["cwd"] is not None


def test_session_exec_existing(url: str) -> None:
    """POST /sessions/exec with session_id → existing session."""
    with httpx.Client(timeout=10) as c:
        # Create first
        resp = c.post(f"{url}/sessions/exec", json={"command": "echo first", "timeout": 10})
        resp.raise_for_status()
        sid = resp.json()["session_id"]

        # Execute again in same session
        resp = c.post(f"{url}/sessions/exec", json={
            "command": "echo second",
            "timeout": 10,
            "session_id": sid,
        })
        resp.raise_for_status()
        data = resp.json()
        assert data["is_new"] is False
        assert "second" in data["stdout"]


def test_list_sessions(url: str) -> None:
    with httpx.Client(timeout=10) as c:
        resp = c.get(f"{url}/sessions")
        resp.raise_for_status()
        data = resp.json()
        assert isinstance(data, list)


def test_kill_session(url: str) -> None:
    with httpx.Client(timeout=10) as c:
        # Create session
        resp = c.post(f"{url}/sessions/exec", json={"command": "echo test", "timeout": 10})
        resp.raise_for_status()
        sid = resp.json()["session_id"]

        # Kill it
        resp = c.post(f"{url}/sessions/{sid}/kill")
        resp.raise_for_status()
        data = resp.json()
        assert data["success"] is True

        # Killing again should 404
        resp = c.post(f"{url}/sessions/{sid}/kill")
        assert resp.status_code == 404


# ── Scripts ─────────────────────────────────────────────────────────────────

def test_submit_script_foreground(url: str) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write("#!/bin/bash\necho 'fg_script_ok'\n")
        script_path = f.name

    try:
        with httpx.Client(timeout=10) as c:
            resp = c.post(f"{url}/scripts/submit", json={
                "command_path": script_path,
                "timeout": 10,
                "run_in_background": False,
            })
            resp.raise_for_status()
            data = resp.json()
            assert data["status"] in ("completed", "failed", "running"), f"bad status: {data['status']}"
            assert "fg_script_ok" in data.get("stdout", "") or data["status"] == "running"
    finally:
        os.unlink(script_path)


def test_submit_script_background(url: str) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write("#!/bin/bash\necho 'bg_script_ok'\nsleep 2\n")
        script_path = f.name

    try:
        with httpx.Client(timeout=10) as c:
            resp = c.post(f"{url}/scripts/submit", json={
                "command_path": script_path,
                "timeout": 10,
                "run_in_background": True,
            })
            resp.raise_for_status()
            data = resp.json()
            assert data["status"] == "running"
            assert data["session_id"] is not None
            assert data["log_path"] is not None
            assert "gpu_backend_" in data["session_id"]
    finally:
        os.unlink(script_path)


def test_script_status_and_log(url: str) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write("#!/bin/bash\necho 'log_test_output'\n")
        script_path = f.name

    try:
        with httpx.Client(timeout=10) as c:
            # Submit foreground
            resp = c.post(f"{url}/scripts/submit", json={
                "command_path": script_path,
                "timeout": 10,
                "run_in_background": False,
            })
            resp.raise_for_status()
            data = resp.json()
            sid = data["script_id"]

            # Status
            resp = c.get(f"{url}/scripts/{sid}")
            resp.raise_for_status()
            status_data = resp.json()
            assert status_data["script_id"] == sid

            # Log
            resp = c.get(f"{url}/scripts/{sid}/log")
            resp.raise_for_status()
            log_data = resp.json()
            assert "log_test_output" in log_data["stdout"]
    finally:
        os.unlink(script_path)


def test_kill_script(url: str) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write("#!/bin/bash\nsleep 300\n")
        script_path = f.name

    try:
        with httpx.Client(timeout=10) as c:
            # Submit background so we can kill it
            resp = c.post(f"{url}/scripts/submit", json={
                "command_path": script_path,
                "timeout": 10,
                "run_in_background": True,
            })
            resp.raise_for_status()
            data = resp.json()
            sid = data["script_id"]

            # Kill
            resp = c.post(f"{url}/scripts/{sid}/kill")
            resp.raise_for_status()
            kill_data = resp.json()
            assert kill_data["success"] is True
    finally:
        os.unlink(script_path)


def test_status_aggregate(url: str) -> None:
    with httpx.Client(timeout=10) as c:
        resp = c.get(f"{url}/status")
        resp.raise_for_status()
        data = resp.json()
        assert "sessions" in data
        assert "gpu_info" in data


def test_script_not_found(url: str) -> None:
    with httpx.Client(timeout=10) as c:
        resp = c.post(f"{url}/scripts/submit", json={
            "command_path": "/tmp/nonexistent_script_xyz.sh",
            "timeout": 10,
        })
        resp.raise_for_status()
        data = resp.json()
        assert data["status"] == "failed"


# ── Runner ──────────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_health,
    test_resources,
    test_session_create_and_exec,
    test_session_exec_existing,
    test_list_sessions,
    test_kill_session,
    test_submit_script_foreground,
    test_submit_script_background,
    test_script_status_and_log,
    test_kill_script,
    test_status_aggregate,
    test_script_not_found,
]


def run_tests(url: str) -> bool:
    print(f"Running {len(ALL_TESTS)} tests against {url}\n" + "=" * 60)
    results = [_test(fn.__name__, url, fn) for fn in ALL_TESTS]
    print("=" * 60)
    passed = sum(results)
    print(f"Results: {passed}/{len(results)} passed")
    return all(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test GPU backend server")
    parser.add_argument("--url", default=os.environ.get("BACKEND_URL", "http://localhost:8989"),
                        help="Backend server URL")
    args = parser.parse_args()
    sys.exit(0 if run_tests(args.url) else 1)
