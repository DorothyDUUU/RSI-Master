"""CLI 入口。

启动方式：
    python -m tools.bash_backend --port 40058
    python -m tools.bash_backend --port 40058 --workspace /path/to/workspace
"""

from __future__ import annotations

import argparse
import logging

from .app import create_app
from .constants import DEFAULT_MAX_SLEEP_SECONDS, DEFAULT_PORT, DEFAULT_WHITELIST, MAX_CONCURRENT_PROCESSES, MAX_OUTPUT_LENGTH

logger = logging.getLogger("bash_backend")


def main():
    parser = argparse.ArgumentParser(
        description="Bash Backend Server — 安全的 bash 命令执行服务",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"服务端口 (默认: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="绑定地址 (默认: 0.0.0.0)",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default="",
        help="默认工作目录",
    )
    parser.add_argument(
        "--whitelist",
        type=str,
        nargs="*",
        default=None,
        help="额外的白名单路径（追加到默认白名单）",
    )
    parser.add_argument(
        "--memory-limit",
        type=str,
        default=None,
        help="内存限制 (如 '4g', '512m')",
    )
    parser.add_argument(
        "--cpu-limit",
        type=float,
        default=0.0,
        help="CPU 时间限制（秒）",
    )
    parser.add_argument(
        "--cpu-devices",
        type=str,
        default=None,
        help="CPU 核心绑定 (如 '0-7')",
    )
    parser.add_argument(
        "--max-output",
        type=int,
        default=MAX_OUTPUT_LENGTH,
        help=f"最大输出长度（字符数，默认: {MAX_OUTPUT_LENGTH}）",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=MAX_CONCURRENT_PROCESSES,
        help=f"最大并发进程数（默认: {MAX_CONCURRENT_PROCESSES}）",
    )
    parser.add_argument(
        "--max-sleep-seconds",
        type=int,
        default=DEFAULT_MAX_SLEEP_SECONDS,
        help=f"long_sleep 过滤器上限, 拦截 sleep 超过该秒数的命令（默认: {DEFAULT_MAX_SLEEP_SECONDS}，即 40min）",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="",
        help="JSONL 日志存储目录 (默认: tools/bash_backend/data/)",
    )

    args = parser.parse_args()

    app = create_app(
        default_workspace=args.workspace,
        whitelist=args.whitelist,
        memory_limit=args.memory_limit,
        cpu_limit=args.cpu_limit,
        cpu_devices=args.cpu_devices,
        max_output_length=args.max_output,
        max_concurrent_processes=args.max_concurrent,
        max_sleep_seconds=args.max_sleep_seconds,
        data_dir=args.data_dir,
    )

    logger.info(f"Starting Bash Backend Server on {args.host}:{args.port}")
    logger.info(f"Default workspace: {args.workspace or '(not set)'}")
    logger.info(f"Whitelist: {DEFAULT_WHITELIST + (args.whitelist or [])}")
    logger.info(f"Memory limit: {args.memory_limit or '(none)'}")
    logger.info(f"CPU limit: {args.cpu_limit or '(none)'}s")
    logger.info(f"Max output: {args.max_output} chars")
    logger.info(f"Max concurrent: {args.max_concurrent}")
    logger.info(f"Max sleep: {args.max_sleep_seconds}s (long_sleep filter)")
    logger.info(f"Data dir: {args.data_dir or '(default)'}")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
