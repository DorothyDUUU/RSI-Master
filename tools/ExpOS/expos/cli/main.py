"""Top-level dispatcher for ``python -m expos.cli`` / ``expos``.

Sub-command tree (see :mod:`expos.cli.__init__` for usage examples)::

    expos <verb> <object> [options]
      init
      dump-state
      events tail
      state add-new-worker | get-current-worker | inspect-worker
      worker show-datapool | inspect-data | update-readme | add-data | submit
             | add-reviewer | finalize
      reviewer show-datapool | inspect-data | inspect-experiment
               | submit-review | finalize
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, Sequence

from ..service.worker_service import SubmitBackend
from . import init_cli, reviewer_cli, state_cli, worker_cli
from .context import ENV_ROOT, build_context

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="expos",
        description="ExperimentOS CLI — bash-callable frontend for the rsi_master workflow",
    )
    parser.add_argument(
        "--root",
        default=None,
        help=f"Path to the ExperimentOS root directory (or set ${ENV_ROOT})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON on stdout (default: YAML-like human output)",
    )
    parser.add_argument(
        "--backend-config",
        default=None,
        dest="backend_config",
        help="Path to expos/backends/config.yaml override (default: packaged config)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_cli.register(subparsers)
    state_cli.register(subparsers)
    worker_cli.register(subparsers)
    reviewer_cli.register(subparsers)
    return parser


def _command_path(args: argparse.Namespace) -> str:
    """Human-path of the invoked verb, e.g. ``worker.submit`` / ``events.tail``."""
    parts = [str(getattr(args, "command", "") or "")]
    for key in ("state_cmd", "worker_cmd", "reviewer_cmd", "events_cmd"):
        val = getattr(args, key, None)
        if val:
            parts.append(str(val))
            break
    return ".".join(part for part in parts if part)


def _audit_params(args: argparse.Namespace) -> dict[str, Any]:
    """Sanitize CLI args for the audit record.

    丢掉内部字段 (handler / root / json 等) 和超大字符串, 只保留简单的
    str/int/float/bool 或标量 list —— 让事后解析 events.jsonl 就能还原
    "agent 跑了哪条命令、带了什么参数"。
    """
    skip = {
        "handler",
        "root",
        "backend_config",
        "json",
        "command",
        "state_cmd",
        "worker_cmd",
        "reviewer_cmd",
        "events_cmd",
    }
    params: dict[str, Any] = {}
    for key, val in vars(args).items():
        if key in skip or val is None or val is False or val == []:
            continue
        if isinstance(val, str) and len(val) > 2000:
            val = val[:2000] + "…"
        scalar = isinstance(val, (str, int, float, bool))
        scalar_list = isinstance(val, list) and all(
            isinstance(x, (str, int, float)) for x in val
        )
        if scalar or scalar_list:
            params[key] = val
    return params


def main(
    argv: Sequence[str] | None = None,
    *,
    submit_backend: SubmitBackend | None = None,
) -> int:
    """CLI entrypoint used by ``python -m expos.cli`` and the console script."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 1
    ctx = build_context(args, submit_backend=submit_backend)
    try:
        rc = handler(ctx, args)
    finally:
        # 审计: 每条 CLI 命令 (无论读/写) 都落一条事件到 events.jsonl, 让
        # state 成为完整可解析的操作记录 (init 后 events.jsonl 已就绪)。
        try:
            ctx.os.events.emit(
                "cli_invoked",
                tool=f"cli.{_command_path(args)}",
                payload={"args": _audit_params(args)},
            )
        except Exception as exc:  # noqa: BLE001 — audit 绝不能搞挂 CLI
            logger.debug("audit cli_invoked failed: %s", exc)
    return rc


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
