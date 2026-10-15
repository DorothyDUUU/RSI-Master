"""CLI entry-point for the GPU backend client.

Usage::

    gpu-backend exec --command "nvidia-smi"
    gpu-backend list-sessions
    gpu-backend kill-session --session-id gpu_shell_abc123
    gpu-backend submit --script-path ./train.sh --background
    gpu-backend script-status --script-id gpu_script_abc123
    gpu-backend script-log --script-id gpu_script_abc123
    gpu-backend kill-script --script-id gpu_script_abc123
    gpu-backend info
    gpu-backend status
    gpu-backend health
    gpu-backend server start --port 8989

Configuration is read from ``config.yaml`` in the package directory.
Environment variables (``BACKEND_URL``, ``BACKEND_HOST``, ``BACKEND_PORT``)
take precedence.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from .client import BackendClient
from .config import get_backend_url, get_default_max_mem_quota, get_default_timeout


def _get_client() -> BackendClient:
    return BackendClient(base_url=get_backend_url())


def _print_json(obj: Any) -> None:
    if hasattr(obj, "model_dump"):
        print(json.dumps(obj.model_dump(), indent=2, default=str))
    elif isinstance(obj, dict):
        print(json.dumps(obj, indent=2, default=str))
    elif isinstance(obj, list):
        print(json.dumps([item.model_dump() if hasattr(item, "model_dump") else item for item in obj], indent=2, default=str))
    else:
        print(obj)


# ── Command handlers ────────────────────────────────────────────────────────

def cmd_exec(args: argparse.Namespace) -> int:
    client = _get_client()
    resp = client.exec_command(
        command=args.command_str,
        timeout=args.timeout,
        session_id=args.session_id,
    )
    print(f"[SESSION_ID] {resp.session_id}")
    print(f"[IS_NEW] {resp.is_new}")
    if resp.cwd:
        print(f"[PWD] {resp.cwd}")
    print(f"[RETURN_CODE] {resp.return_code}")
    if resp.stdout:
        print(f"[STDOUT]\n{resp.stdout}")
    if resp.stderr:
        print(f"[STDERR]\n{resp.stderr}", file=sys.stderr)
    return resp.return_code


def cmd_list_sessions(args: argparse.Namespace) -> int:
    client = _get_client()
    sessions = client.list_sessions()
    if not sessions:
        print("No active sessions.")
        return 0
    print(f"[TOTAL] {len(sessions)}")
    for s in sessions:
        bg = " [background]" if s.is_background else ""
        print(f"\n[SESSION_ID] {s.session_id}")
        print(f"[STATUS] {s.status.value}{bg}")
        print(f"[PWD] {s.cwd}")
        print(f"[CREATED] {s.created_at.strftime('%Y-%m-%d %H:%M:%S') if s.created_at else 'unknown'}")
        if s.last_command:
            print(f"[LAST_COMMAND] {s.last_command}")
        if s.log_path:
            print(f"[LOG_PATH] {s.log_path}")
    return 0


def cmd_kill_session(args: argparse.Namespace) -> int:
    client = _get_client()
    ok = client.kill_session(args.session_id)
    if ok:
        print(f"Session {args.session_id} killed.")
        return 0
    else:
        print(f"Failed to kill session {args.session_id}", file=sys.stderr)
        return 1


def cmd_submit(args: argparse.Namespace) -> int:
    client = _get_client()
    resp = client.submit_script(
        command_path=args.script_path,
        timeout=args.timeout,
        resume_session_id=args.resume_session_id,
        run_in_background=args.background,
        gpu_nums=args.gpu_nums,
        priority=args.priority,
        max_mem_quota=args.max_mem_quota,
        need_exclusive_gpu=args.need_exclusive_gpu,
    )
    print(f"[SCRIPT_ID] {resp.script_id}")
    print(f"[STATUS] {resp.status.value}")
    if resp.session_id:
        print(f"[SESSION_ID] {resp.session_id}")
    if resp.log_path:
        print(f"[LOG_PATH] {resp.log_path}")
    if resp.return_code is not None:
        print(f"[RETURN_CODE] {resp.return_code}")
    if resp.message:
        print(f"[MESSAGE] {resp.message}")
    if resp.stdout:
        print(f"[STDOUT]\n{resp.stdout}")
    if resp.stderr:
        print(f"[STDERR]\n{resp.stderr}", file=sys.stderr)
    return 0 if resp.status.value in ("completed", "running", "queued") else 1


def cmd_script_status(args: argparse.Namespace) -> int:
    client = _get_client()
    info = client.get_script_status(args.script_id)
    if info is None:
        print(f"Script {args.script_id} not found.", file=sys.stderr)
        return 1
    _print_json(info)
    return 0


def cmd_script_log(args: argparse.Namespace) -> int:
    client = _get_client()
    log = client.get_script_log(args.script_id)
    if log is None:
        print(f"Script {args.script_id} not found.", file=sys.stderr)
        return 1
    print(log.stdout or "(no output)")
    if log.stderr:
        print(f"\n--- stderr ---\n{log.stderr}", file=sys.stderr)
    return 0


def cmd_kill_script(args: argparse.Namespace) -> int:
    client = _get_client()
    ok = client.kill_script(args.script_id)
    if ok:
        print(f"Script {args.script_id} killed.")
        return 0
    else:
        print(f"Script {args.script_id} not found or not running.", file=sys.stderr)
        return 1


def cmd_queue(args: argparse.Namespace) -> int:
    with _get_client() as client:
        _print_json(client.get_queue())
    return 0


def cmd_cancel_queued(args: argparse.Namespace) -> int:
    with _get_client() as client:
        result = client.cancel_queued_script(args.script_id)
        _print_json(result)
    return 0 if result.get("cancelled") else 1


def cmd_info(args: argparse.Namespace) -> int:
    client = _get_client()
    info = client.get_resources()
    status = client.get_status()

    print("[GPU_DEVICES]")
    for d in info.devices:
        vram = getattr(d, 'vram_usage_percent', None)
        vram_str = f"vram {vram:5.1f}%" if vram is not None else f"mem {d.memory_utilization:5.1f}%"
        print(
            f"GPU {d.device_id}  {d.name:30s}  "
            f"{vram_str}  util {d.gpu_utilization:5.1f}%  "
            f"free {d.free_memory_mb:5d}MB"
        )
    print(f"\n[SYSTEM]  CPU: {info.cpu_percent:.1f}%  "
          f"Mem: {info.memory_percent:.1f}%  "
          f"({info.memory_used_gb:.1f}/{info.memory_total_gb:.1f} GB)")

    q = status.get("queue", [])
    if q:
        print(f"\n[QUEUE] ({len(q)} waiting)")
        for item in q:
            print(f"  {item['script_id']}  priority={item['priority']}  need={item['gpu_nums']} GPU(s)")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    client = _get_client()
    _print_json(client.get_status())
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    client = _get_client()
    status = client.health_check()
    _print_json(status)
    return 0 if status.get("status") == "healthy" else 1


def cmd_server_start(args: argparse.Namespace) -> int:
    """Start the backend server (convenience wrapper around uvicorn)."""
    import subprocess
    from .config import load_config

    cfg = load_config().get("server", {})
    host = args.host or cfg.get("host", "0.0.0.0")
    port = str(args.port or cfg.get("port", "8090"))
    log_level = args.log_level or cfg.get("log_level", "info")

    # wrapper/task 日志目录: --log-dir 优先, 否则 env, 否则 SessionManager 默认
    # /tmp/gpu_backend_logs。uvicorn 子进程继承 env, main.py lifespan 读它。
    if args.log_dir:
        os.environ["GPU_BACKEND_LOG_DIR"] = args.log_dir

    server_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [
        sys.executable, "-m", "uvicorn", "backend.server.main:app",
        "--host", host,
        "--port", port,
        "--log-level", log_level,
    ]
    if args.reload:
        cmd.append("--reload")

    print(f"Starting server on {host}:{port} (log-level={log_level})")
    os.chdir(os.path.dirname(server_dir))

    # 服务端主日志: --log-file 指定则 uvicorn stdout/stderr 一并落盘到该文件,
    # 否则保持 stdout (由启动方 shell 重定向, 如 `> /tmp/gpu_backend_8989.log`)。
    if args.log_file:
        fh = open(args.log_file, "ab")  # noqa: SIM115 — 子进程生命周期内持有
        try:
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
        except Exception:
            fh.close()
            raise
        try:
            return proc.wait()
        finally:
            fh.close()
    return subprocess.run(cmd).returncode


# ── Parser setup ────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpu-backend",
        description="GPU Backend CLI — manage shell sessions, submit scripts, monitor resources.",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")
    sub.required = True

    # exec
    p_exec = sub.add_parser("exec", help="Execute a command in a shell session")
    p_exec.add_argument("--command", "-c", required=True, dest="command_str", help="Shell command to run")
    p_exec.add_argument("--session-id", "-s", default=None, help="Target session (omit to create new)")
    p_exec.add_argument("--timeout", "-t", type=int, default=get_default_timeout(), help="Timeout in seconds")
    p_exec.set_defaults(func=cmd_exec)

    # list-sessions
    p_ls = sub.add_parser("list-sessions", aliases=["ls"], help="List active shell sessions")
    p_ls.set_defaults(func=cmd_list_sessions)

    # kill-session
    p_ks = sub.add_parser("kill-session", help="Kill a shell session")
    p_ks.add_argument("--session-id", "-s", required=True, help="Session to kill")
    p_ks.set_defaults(func=cmd_kill_session)

    # submit
    p_sub = sub.add_parser("submit", help="Submit a bash script for execution")
    p_sub.add_argument("--script-path", "-p", required=True, help="Path to bash script")
    p_sub.add_argument("--background", "-b", action="store_true", default=False, help="Run in background tmux session")
    p_sub.add_argument("--resume-session-id", "-r", default=None, help="Existing session to source env from")
    p_sub.add_argument("--timeout", "-t", type=int, default=get_default_timeout(), help="Timeout in seconds")
    p_sub.add_argument("--gpu-nums", "-g", type=int, required=True, help="Number of GPUs required (must be specified)")
    p_sub.add_argument("--priority", choices=["low", "normal", "high"], default="normal",
                       help="Scheduling priority: low, normal, high (default: normal)")
    p_sub.add_argument("--max-mem-quota", type=int, default=get_default_max_mem_quota(),
                       help="Minimum free GPU memory (MB) required per device (default from config)")
    p_sub.add_argument("--need-exclusive-gpu", action="store_true", dest="need_exclusive_gpu", default=True,
                       help="Exclusively hold the allocated GPU(s) for the task's whole lifetime "
                            "(default: on — 2026-08-29; use --no-need-exclusive-gpu to allow sharing)")
    p_sub.add_argument("--no-need-exclusive-gpu", action="store_false", dest="need_exclusive_gpu", default=True,
                       help="Allow other tasks to share the GPU(s) after the short lease window")
    p_sub.set_defaults(func=cmd_submit)

    # script-status
    p_ss = sub.add_parser("script-status", help="Get script execution status")
    p_ss.add_argument("--script-id", "-s", required=True, help="Script ID")
    p_ss.set_defaults(func=cmd_script_status)

    # script-log
    p_sl = sub.add_parser("script-log", help="Get script output log")
    p_sl.add_argument("--script-id", "-s", required=True, help="Script ID")
    p_sl.set_defaults(func=cmd_script_log)

    # kill-script
    p_sk = sub.add_parser("kill-script", help="Kill a running script")
    p_sk.add_argument("--script-id", "-s", required=True, help="Script ID")
    p_sk.set_defaults(func=cmd_kill_script)

    # info
    p_queue = sub.add_parser("queue", help="Show live waiting/running queue and owner limits (JSON)")
    p_queue.set_defaults(func=cmd_queue)
    p_cancel = sub.add_parser("cancel-queued", help="Withdraw ONLY an unstarted script; never kill running work")
    p_cancel.add_argument("--script-id", "-s", required=True)
    p_cancel.set_defaults(func=cmd_cancel_queued)

    p_info = sub.add_parser("info", help="Show GPU / CPU / memory resources")
    p_info.set_defaults(func=cmd_info)

    # status
    p_st = sub.add_parser("status", help="Show combined sessions + resources (JSON)")
    p_st.set_defaults(func=cmd_status)

    # health
    p_h = sub.add_parser("health", help="Health check")
    p_h.set_defaults(func=cmd_health)

    # server start
    p_srv = sub.add_parser("server", help="Server management commands")
    srv_sub = p_srv.add_subparsers(dest="server_command")
    srv_start = srv_sub.add_parser("start", help="Start the backend server")
    srv_start.add_argument("--host", default=None, help="Bind host (default: 0.0.0.0)")
    srv_start.add_argument("--port", type=int, default=None, help="Bind port (default: 8989)")
    srv_start.add_argument("--log-level", default=None, choices=["debug", "info", "warning", "error"])
    srv_start.add_argument("--reload", action="store_true", default=False, help="Enable auto-reload")
    srv_start.add_argument(
        "--log-dir", default=None,
        help="wrapper/task 日志目录 (默认 /tmp/gpu_backend_logs, pod 本地重启即丢; "
             "建议 <run_dir>/gpu_backend_logs 落到 run-dir NFS 持久)",
    )
    srv_start.add_argument(
        "--log-file", default=None,
        help="服务端主日志文件 (uvicorn stdout/stderr 一并落盘; 默认 stdout, "
             "由启动方 shell 重定向。建议 <run_dir>/gpu_backend_server.log)",
    )
    srv_start.set_defaults(func=cmd_server_start)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
