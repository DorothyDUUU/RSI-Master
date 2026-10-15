"""``expos state ...`` subcommands."""

from __future__ import annotations

import argparse

from .context import CLIContext
from .output import emit_error, emit_json, emit_yaml_like


def register(subparsers: argparse._SubParsersAction) -> None:
    """Attach ``state`` and its sub-subcommands to the top-level parser."""
    parser = subparsers.add_parser("state", help="Global DAG operations")
    state_subs = parser.add_subparsers(dest="state_cmd", required=True)

    add_worker = state_subs.add_parser("add-new-worker", help="Instantiate a new worker node")
    add_worker.add_argument("--parent-worker-id", default=None, help="Optional parent worker id")
    add_worker.add_argument("--objective", default="", help="Optional worker objective seed")
    add_worker.set_defaults(handler=_handle_add_worker)

    get_current = state_subs.add_parser(
        "get-current-worker", help="Show state.json + current_worker_id summary"
    )
    get_current.set_defaults(handler=_handle_get_current)

    inspect = state_subs.add_parser("inspect-worker", help="Detailed info for one worker")
    inspect.add_argument("--worker-id", required=True)
    inspect.add_argument("--event-limit", type=int, default=200)
    inspect.set_defaults(handler=_handle_inspect_worker)

    sync = state_subs.add_parser(
        "sync",
        help="Reconcile disk results into state.json + events.json (fix orphans)",
    )
    sync.add_argument(
        "--worker-id",
        default=None,
        help="Only sync experiments owned by this worker (default: all)",
    )
    sync.set_defaults(handler=_handle_sync)

    finalize = state_subs.add_parser(
        "finalize-worker",
        help="Manually mark a worker terminal (abort) to free a concurrency slot",
    )
    finalize.add_argument("--worker-id", required=True)
    finalize.add_argument(
        "--status",
        default="aborted",
        help="Final status: aborted (default) or failed",
    )
    finalize.add_argument("--notes", default="", help="Note appended to HISTORY.md")
    finalize.set_defaults(handler=_handle_finalize_worker)


def _handle_add_worker(ctx: CLIContext, args: argparse.Namespace) -> int:
    try:
        result = ctx.state.add_new_worker(
            parent_worker_id=args.parent_worker_id,
            objective=args.objective,
        )
    except ValueError as exc:
        return emit_error(str(exc))
    if args.json:
        emit_json(result)
    else:
        emit_yaml_like(result)
    return 0


def _handle_get_current(ctx: CLIContext, args: argparse.Namespace) -> int:
    result = ctx.state.get_current_worker()
    if args.json:
        emit_json(result)
    else:
        emit_yaml_like(result)
    return 0


def _handle_inspect_worker(ctx: CLIContext, args: argparse.Namespace) -> int:
    try:
        result = ctx.state.inspect_worker(args.worker_id, event_limit=args.event_limit)
    except KeyError as exc:
        return emit_error(str(exc), code=2)
    if args.json:
        emit_json(result)
    else:
        emit_yaml_like(result)
    return 0


def _handle_sync(ctx: CLIContext, args: argparse.Namespace) -> int:
    try:
        result = ctx.state.sync(worker_id=args.worker_id)
    except Exception as exc:  # noqa: BLE001
        return emit_error(str(exc), code=2)
    if args.json:
        emit_json(result)
    else:
        emit_yaml_like(result)
    return 0


def _handle_finalize_worker(ctx: CLIContext, args: argparse.Namespace) -> int:
    """Manual abort/finalize — frees the concurrency slot held by a worker.

    Identical semantics to the ``state_finalize_worker`` tool: both just mark
    the worker node terminal (``aborted`` default), which the DAG-driven cancel
    hook picks up on the worker's next turn.
    """
    try:
        result = ctx.worker.finalize(
            args.worker_id,
            status=args.status,
            notes=args.notes or "Aborted via CLI `state finalize-worker`.",
        )
    except Exception as exc:  # noqa: BLE001
        return emit_error(str(exc), code=2)
    if args.json:
        emit_json(result)
    else:
        emit_yaml_like(result)
    return 0
