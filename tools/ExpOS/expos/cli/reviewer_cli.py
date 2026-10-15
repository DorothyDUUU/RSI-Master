"""``expos reviewer ...`` subcommands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..service.contracts import ReviewReport
from .context import CLIContext
from .output import emit_error, emit_json, emit_yaml_like


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("reviewer", help="Reviewer-scoped CLI")
    reviewer_subs = parser.add_subparsers(dest="reviewer_cmd", required=True)

    show = reviewer_subs.add_parser("show-datapool", help="List every data_pool entry")
    show.set_defaults(handler=_handle_show_datapool)

    inspect_data = reviewer_subs.add_parser("inspect-data", help="Show one data_pool entry")
    inspect_data.add_argument("--data-id", required=True)
    inspect_data.set_defaults(handler=_handle_inspect_data)

    inspect_experiment = reviewer_subs.add_parser(
        "inspect-experiment", help="Parse an experiment folder into a review-ready record"
    )
    inspect_experiment.add_argument("--experiment-id", required=True)
    inspect_experiment.set_defaults(handler=_handle_inspect_experiment)

    submit_review = reviewer_subs.add_parser(
        "submit-review", help="Persist a ReviewReport JSON file into reviews/"
    )
    submit_review.add_argument("--reviewer-id", required=True)
    submit_review.add_argument("--experiment-id", required=True)
    submit_review.add_argument(
        "--report",
        required=True,
        help="Path to a JSON file containing the ReviewReport payload",
    )
    submit_review.set_defaults(handler=_handle_submit_review)

    finalize = reviewer_subs.add_parser("finalize", help="Mark reviewer as finished/aborted")
    finalize.add_argument("--reviewer-id", required=True)
    finalize.add_argument("--status", default="finished")
    finalize.set_defaults(handler=_handle_finalize)


def _handle_show_datapool(ctx: CLIContext, args: argparse.Namespace) -> int:
    payload = {"entries": ctx.reviewer.show_datapool()}
    if args.json:
        emit_json(payload)
    else:
        emit_yaml_like(payload)
    return 0


def _handle_inspect_data(ctx: CLIContext, args: argparse.Namespace) -> int:
    try:
        result = ctx.reviewer.inspect_data(args.data_id)
    except KeyError as exc:
        return emit_error(str(exc), code=2)
    if args.json:
        emit_json(result)
    else:
        emit_yaml_like(result)
    return 0


def _handle_inspect_experiment(ctx: CLIContext, args: argparse.Namespace) -> int:
    try:
        result = ctx.reviewer.inspect_experiment(args.experiment_id)
    except KeyError as exc:
        return emit_error(str(exc), code=2)
    if args.json:
        emit_json(result)
    else:
        emit_yaml_like(result)
    return 0


def _handle_submit_review(ctx: CLIContext, args: argparse.Namespace) -> int:
    report_path = Path(args.report).expanduser()
    if not report_path.exists():
        return emit_error(f"report file does not exist: {report_path}", code=2)
    try:
        raw = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return emit_error(f"invalid JSON in report file: {exc}", code=2)
    # Enforce the reviewer_id/experiment_id from the CLI, even if the file
    # forgot to include them.
    raw["reviewer_id"] = args.reviewer_id
    raw["experiment_id"] = args.experiment_id
    raw.setdefault("worker_id", "")
    try:
        report = ReviewReport(**raw)
    except Exception as exc:  # noqa: BLE001
        return emit_error(f"invalid ReviewReport payload: {exc}", code=2)
    try:
        result = ctx.reviewer.submit_review(
            reviewer_id=args.reviewer_id,
            experiment_id=args.experiment_id,
            report=report,
        )
    except KeyError as exc:
        return emit_error(str(exc), code=2)
    if args.json:
        emit_json(result)
    else:
        emit_yaml_like(result)
    return 0


def _handle_finalize(ctx: CLIContext, args: argparse.Namespace) -> int:
    try:
        result = ctx.reviewer.finalize(args.reviewer_id, status=args.status)
    except KeyError as exc:
        return emit_error(str(exc), code=2)
    if args.json:
        emit_json(result)
    else:
        emit_yaml_like(result)
    return 0
