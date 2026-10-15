"""``expos init`` and ``expos events`` subcommands.

- ``init``   — bootstrap the ExperimentOS directory tree
- ``events`` — tail the structured event stream
- ``state.json`` — pretty-print the raw state file
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .context import CLIContext
from .output import emit_error, emit_json, emit_yaml_like


def register(subparsers: argparse._SubParsersAction) -> None:
    init = subparsers.add_parser("init", help="Initialize an ExperimentOS root directory")
    init.add_argument("--task", required=True, help="Human-readable task description")
    init.add_argument("--benchmark", default="aime_2025")
    init.add_argument(
        "--run-contract",
        default=None,
        help="Optional JSON file / inline JSON with run contract fields",
    )
    init.add_argument(
        "--eval-all",
        default=None,
        help="JSON 文件 / 内联 JSON — 作为 state['eval_all'] 落盘 (bench_id/limit/"
             "max_tokens 等; 非空时 stage-3 走 benchmark suite 统一评测)",
    )
    init.add_argument("--overwrite", action="store_true", help="Reset state.json if it exists")
    init.set_defaults(handler=_handle_init)

    events = subparsers.add_parser("events", help="Structured event stream utilities")
    ev_subs = events.add_subparsers(dest="events_cmd", required=True)
    tail = ev_subs.add_parser("tail", help="Show the most recent events")
    tail.add_argument("--limit", type=int, default=100)
    tail.add_argument("--worker-id", default=None)
    tail.add_argument("--experiment-id", default=None)
    tail.add_argument("--reviewer-id", default=None)
    tail.add_argument("--type", dest="event_types", nargs="*", default=None)
    tail.set_defaults(handler=_handle_events_tail)

    dump_state = subparsers.add_parser("dump-state", help="Print state.json")
    dump_state.set_defaults(handler=_handle_dump_state)


def _handle_init(ctx: CLIContext, args: argparse.Namespace) -> int:
    run_contract = _load_run_contract(args.run_contract)
    eval_all = _load_eval_all(args.eval_all)
    state = ctx.os.initialize(
        task_description=args.task,
        benchmark=args.benchmark,
        run_contract=run_contract,
        eval_all=eval_all,
        overwrite=args.overwrite,
    )
    result = {
        "root": str(ctx.root),
        "state_path": str(ctx.os.state_path),
        "events_path": str(ctx.os.events.path),
        "task": state.get("task_description"),
        "benchmark": state.get("benchmark"),
    }
    if args.json:
        emit_json(result)
    else:
        emit_yaml_like(result)
    return 0


def _handle_events_tail(ctx: CLIContext, args: argparse.Namespace) -> int:
    events = ctx.os.events.filter(
        worker_id=args.worker_id,
        experiment_id=args.experiment_id,
        reviewer_id=args.reviewer_id,
        types=args.event_types,
        limit=args.limit,
    )
    if args.json:
        emit_json({"events": events})
    else:
        for event in events:
            print(json.dumps(event, ensure_ascii=False, default=str))
    return 0


def _handle_dump_state(ctx: CLIContext, args: argparse.Namespace) -> int:
    state = ctx.os.load_state()
    if args.json:
        emit_json(state)
    else:
        emit_yaml_like(state)
    return 0


def _load_run_contract(value: str | None) -> dict | None:
    if not value:
        return None
    if value.strip().startswith("{"):
        return json.loads(value)
    path = Path(value).expanduser()
    if not path.exists():
        raise SystemExit(f"run-contract path does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_eval_all(value: str | None) -> dict | None:
    """Parse ``--eval-all`` (inline JSON or path to a JSON file) into a dict.

    ``None`` / 空串 → 返回 ``None`` (不覆盖 init 的默认 ``{}`` 语义)。
    """
    if not value:
        return None
    value = value.strip()
    if value.startswith("{"):
        return json.loads(value)
    path = Path(value).expanduser()
    if not path.exists():
        raise SystemExit(f"eval-all path does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
