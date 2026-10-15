"""Bash Backend Server CLI 客户端。

查看运行中的进程、kill 特定进程、执行命令等。

用法：
    # 查看所有进程
    python -m tools.bash_backend.client ps

    # 只看运行中的
    python -m tools.bash_backend.client ps --status running

    # 查看系统状态
    python -m tools.bash_backend.client status

    # kill 特定进程
    python -m tools.bash_backend.client kill <process_id>

    # kill 全部
    python -m tools.bash_backend.client kill-all

    # 执行命令
    python -m tools.bash_backend.client exec "echo hello" --workspace /tmp

    # 查看过滤器
    python -m tools.bash_backend.client filters

    # 禁用/启用过滤器
    python -m tools.bash_backend.client toggle <filter_name> --off
    python -m tools.bash_backend.client toggle <filter_name> --on
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import httpx

DEFAULT_SERVER = os.environ.get("BASH_BACKEND_URL", "http://localhost:40059").rstrip("/")


def _get_base_url(args) -> str:
    return args.server.rstrip("/")


# ---------------------------------------------------------------------------
# 子命令实现
# ---------------------------------------------------------------------------


def cmd_ps(args):
    """列出进程。"""
    base = _get_base_url(args)
    params = {}
    if args.status:
        params["status"] = args.status

    resp = httpx.get(f"{base}/processes", params=params)
    resp.raise_for_status()
    procs = resp.json()

    if not procs:
        print("没有进程。")
        return

    # 表格输出
    header = f"{'ID':<10} {'STATUS':<12} {'EXIT':<6} {'DURATION':<10} {'COMMAND'}"
    print(header)
    print("-" * len(header) + "-" * 30)

    for p in procs:
        pid = p["process_id"]
        status = p["status"]
        exit_code = p.get("exit_code")
        exit_str = str(exit_code) if exit_code is not None else "-"
        duration = p.get("duration_seconds")
        dur_str = f"{duration:.1f}s" if duration is not None else "-"
        cmd = p["command"]
        # 截断长命令
        if len(cmd) > 60:
            cmd = cmd[:57] + "..."

        # 状态着色
        status_icon = {
            "running": "🟢",
            "completed": "✅",
            "timeout": "⏰",
            "killed": "💀",
            "filtered": "🚫",
        }.get(status, "❓")

        print(f"{pid:<10} {status_icon} {status:<10} {exit_str:<6} {dur_str:<10} {cmd}")

    print(f"\n共 {len(procs)} 个进程")


def cmd_status(args):
    """查看系统状态。"""
    base = _get_base_url(args)
    resp = httpx.get(f"{base}/status")
    resp.raise_for_status()
    data = resp.json()

    cpu = data["cpu_usage_percent"]
    mem = data["memory"]
    print(f"CPU 使用率:  {cpu}%")
    print(f"内存使用率:  {mem['usage_percent']}%")
    print(f"  总内存:    {mem['total_kb'] // 1024 // 1024} GB")
    print(f"  已用:      {mem['used_kb'] // 1024 // 1024} GB")
    print(f"  可用:      {mem['available_kb'] // 1024 // 1024} GB")
    print(f"活跃进程:    {data['active_processes']}")
    print(f"总跟踪进程:  {data['total_tracked_processes']}")


def cmd_kill(args):
    """Kill 指定进程。"""
    base = _get_base_url(args)
    resp = httpx.post(f"{base}/kill/{args.process_id}")
    resp.raise_for_status()
    data = resp.json()

    if data["success"]:
        print(f"✅ {data['message']}")
    else:
        print(f"❌ {data['message']}")
        sys.exit(1)


def cmd_kill_all(args):
    """Kill 所有活跃进程。"""
    base = _get_base_url(args)

    # 先确认
    if not args.yes:
        # 先看看有多少活跃进程
        resp = httpx.get(f"{base}/processes", params={"status": "running"})
        resp.raise_for_status()
        active = resp.json()
        if not active:
            print("没有活跃进程。")
            return

        print(f"即将 kill {len(active)} 个活跃进程：")
        for p in active:
            print(f"  {p['process_id']}: {p['command'][:60]}")

        answer = input("\n确认? (y/N): ").strip().lower()
        if answer != "y":
            print("已取消。")
            return

    resp = httpx.post(f"{base}/kill_all")
    resp.raise_for_status()
    data = resp.json()

    print(f"已 kill {data['killed_count']} 个进程")
    for msg in data["messages"]:
        print(f"  {msg}")


def cmd_exec(args):
    """执行命令。"""
    base = _get_base_url(args)
    payload = {
        "command": args.command,
        "agent_workspace": args.workspace or "",
        "timeout": args.timeout,
    }

    resp = httpx.post(f"{base}/execute", json=payload, timeout=args.timeout + 10)
    resp.raise_for_status()
    data = resp.json()

    if data["filtered"]:
        print(f"🚫 命令被过滤: {data['filter_reason']}", file=sys.stderr)
        sys.exit(1)

    if data["stdout"]:
        print(data["stdout"], end="")
    if data["stderr"]:
        print(data["stderr"], end="", file=sys.stderr)

    print(f"\n[exit_code={data['exit_code']}, duration={data['duration_seconds']}s, pid={data['process_id']}]",
          file=sys.stderr)
    sys.exit(data["exit_code"] if data["exit_code"] >= 0 else 1)


def cmd_filters(args):
    """列看过滤器。"""
    base = _get_base_url(args)
    resp = httpx.get(f"{base}/filters")
    resp.raise_for_status()
    filters = resp.json()

    print(f"{'NAME':<25} {'TYPE':<12} {'ENABLED':<8} {'DESCRIPTION'}")
    print("-" * 90)

    for f in filters:
        icon = "✅" if f["enabled"] else "❌"
        type_str = "pre-filter" if f["type"] == "pre_filter" else "post-check"
        print(f"{f['name']:<25} {type_str:<12} {icon:<8} {f['description']}")


def cmd_toggle(args):
    """启用/禁用过滤器。"""
    base = _get_base_url(args)
    enabled = not args.off  # --off 表示禁用，否则启用

    resp = httpx.post(f"{base}/filters/{args.filter_name}/toggle", json={"enabled": enabled})
    if resp.status_code == 404:
        print(f"❌ 过滤器 '{args.filter_name}' 不存在")
        sys.exit(1)

    resp.raise_for_status()
    data = resp.json()

    state = "启用" if data["enabled"] else "禁用"
    print(f"✅ 过滤器 '{data['name']}' 已{state}")


def cmd_health(args):
    """健康检查。"""
    base = _get_base_url(args)
    resp = httpx.get(f"{base}/health")
    resp.raise_for_status()
    data = resp.json()
    print(f"状态: {data['status']}")
    print(f"时间: {data['timestamp']}")


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Bash Backend Server CLI 客户端",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--server", "-s",
        default=DEFAULT_SERVER,
        help=f"服务器地址 (默认: {DEFAULT_SERVER})",
    )

    sub = parser.add_subparsers(dest="subcommand", help="子命令")

    # ps
    p_ps = sub.add_parser("ps", help="列出进程")
    p_ps.add_argument("--status", choices=["running", "completed", "timeout", "killed", "filtered"],
                       help="按状态过滤")

    # status
    sub.add_parser("status", help="系统资源状态")

    # kill
    p_kill = sub.add_parser("kill", help="kill 指定进程")
    p_kill.add_argument("process_id", help="进程 ID")

    # kill-all
    p_kill_all = sub.add_parser("kill-all", help="kill 所有活跃进程")
    p_kill_all.add_argument("--yes", "-y", action="store_true", help="跳过确认")

    # exec
    p_exec = sub.add_parser("exec", help="执行命令")
    p_exec.add_argument("command", help="要执行的命令")
    p_exec.add_argument("--workspace", "-w", default="", help="工作目录")
    p_exec.add_argument("--timeout", "-t", type=float, default=300, help="超时秒数")

    # filters
    sub.add_parser("filters", help="列看过滤器")

    # toggle
    p_toggle = sub.add_parser("toggle", help="启用/禁用过滤器")
    p_toggle.add_argument("filter_name", help="过滤器名称")
    p_toggle.add_argument("--off", action="store_true", help="禁用（默认启用）")

    # health
    sub.add_parser("health", help="健康检查")

    args = parser.parse_args()

    if not args.subcommand:
        parser.print_help()
        sys.exit(1)

    dispatch = {
        "ps": cmd_ps,
        "status": cmd_status,
        "kill": cmd_kill,
        "kill-all": cmd_kill_all,
        "exec": cmd_exec,
        "filters": cmd_filters,
        "toggle": cmd_toggle,
        "health": cmd_health,
    }

    try:
        dispatch[args.subcommand](args)
    except httpx.ConnectError:
        print(f"❌ 无法连接到服务器。请确认服务已启动: {args.server}", file=sys.stderr)
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        print(f"❌ HTTP 错误: {e.response.status_code} - {e.response.text}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
