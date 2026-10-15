"""Smoke tests for the CLI argument parser.

These tests only exercise argument parsing — they do NOT require a running
server.  Run with::

    python -m backend.tests.test_cli
"""

from __future__ import annotations

import sys

from backend.gpu_backend.cli import build_parser


def test_parser_help():
    """``gpu-backend --help`` works."""
    parser = build_parser()
    # --help prints and exits, so catch SystemExit
    try:
        parser.parse_args(["--help"])
    except SystemExit:
        pass  # expected


def test_parser_exec_minimal():
    args = build_parser().parse_args(["exec", "--command", "echo hi"])
    assert args.command == "exec"
    assert args.command_str == "echo hi"
    assert args.timeout == 300
    assert args.session_id is None


def test_parser_exec_full():
    args = build_parser().parse_args([
        "exec", "-c", "nvidia-smi", "-s", "gpu_shell_abc", "-t", "60",
    ])
    assert args.command_str == "nvidia-smi"
    assert args.session_id == "gpu_shell_abc"
    assert args.timeout == 60


def test_parser_list_sessions():
    args = build_parser().parse_args(["list-sessions"])
    assert args.command == "list-sessions"


def test_parser_ls_alias():
    args = build_parser().parse_args(["ls"])
    assert args.command in ("list-sessions", "ls")


def test_parser_kill_session():
    args = build_parser().parse_args(["kill-session", "--session-id", "gpu_shell_xyz"])
    assert args.command == "kill-session"
    assert args.session_id == "gpu_shell_xyz"


def test_parser_submit_minimal():
    args = build_parser().parse_args(["submit", "--script-path", "/tmp/test.sh"])
    assert args.command == "submit"
    assert args.script_path == "/tmp/test.sh"
    assert args.background is False
    assert args.timeout == 300


def test_parser_submit_background():
    args = build_parser().parse_args([
        "submit", "-p", "/tmp/train.sh", "-b", "-t", "7200", "-r", "gpu_shell_abc",
    ])
    assert args.script_path == "/tmp/train.sh"
    assert args.background is True
    assert args.timeout == 7200
    assert args.resume_session_id == "gpu_shell_abc"


def test_parser_script_status():
    args = build_parser().parse_args(["script-status", "--script-id", "gpu_script_xyz"])
    assert args.command == "script-status"
    assert args.script_id == "gpu_script_xyz"


def test_parser_script_log():
    args = build_parser().parse_args(["script-log", "-s", "gpu_script_xyz"])
    assert args.script_id == "gpu_script_xyz"


def test_parser_kill_script():
    args = build_parser().parse_args(["kill-script", "-s", "gpu_script_xyz"])
    assert args.script_id == "gpu_script_xyz"


def test_parser_info():
    args = build_parser().parse_args(["info"])
    assert args.command == "info"


def test_parser_status():
    args = build_parser().parse_args(["status"])
    assert args.command == "status"


def test_parser_health():
    args = build_parser().parse_args(["health"])
    assert args.command == "health"


def test_parser_server_start():
    args = build_parser().parse_args(["server", "start", "--port", "9999", "--reload"])
    assert args.server_command == "start"
    assert args.port == 9999
    assert args.reload is True


ALL_TESTS = [
    test_parser_help,
    test_parser_exec_minimal,
    test_parser_exec_full,
    test_parser_list_sessions,
    test_parser_ls_alias,
    test_parser_kill_session,
    test_parser_submit_minimal,
    test_parser_submit_background,
    test_parser_script_status,
    test_parser_script_log,
    test_parser_kill_script,
    test_parser_info,
    test_parser_status,
    test_parser_health,
    test_parser_server_start,
]


if __name__ == "__main__":
    print(f"Running {len(ALL_TESTS)} CLI parser tests\n" + "=" * 60)
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
