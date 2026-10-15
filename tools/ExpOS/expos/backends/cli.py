"""``python -m expos.backends`` — server + client CLI for the ExpOS backend.

Usage::

    # Start the FastAPI server on config.yaml host/port
    python -m expos.backends server start

    # Health check
    python -m expos.backends health

    # Manually submit a trial (mostly for debugging / smoke tests)
    python -m expos.backends submit \\
        --experiment-id exp_x --worker-id worker_x \\
        --benchmark gsm8k \\
        --train-data /tmp/train.jsonl \\
        --training-config-path /path/to/train.yaml \\
        --training-mode llama_factory_sft \\
        --workspace /tmp/exp_x
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .client import BackendError, SubmitBackendClient
from .config import load_backend_config
from .models import SubmitRequest


def _print(obj: Any) -> None:
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump()
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def _handle_server_start(args: argparse.Namespace) -> int:
    import uvicorn  # local import: uvicorn 是 server 侧依赖

    cfg = load_backend_config(args.config)
    host = args.host or cfg.host
    port = args.port or cfg.port
    uvicorn.run("expos.backends.server:app", host=host, port=port, reload=args.reload)
    return 0


def _handle_health(args: argparse.Namespace) -> int:
    with SubmitBackendClient(load_backend_config(args.config)) as client:
        _print(client.health())
    return 0


def _handle_submit(args: argparse.Namespace) -> int:
    request = SubmitRequest(
        experiment_id=args.experiment_id,
        worker_id=args.worker_id,
        benchmark=args.benchmark,
        train_data_path=args.train_data,
        training_config_path=args.training_config_path,
        training_mode=args.training_mode,
        experiment_workspace=args.workspace,
    )
    with SubmitBackendClient(load_backend_config(args.config)) as client:
        result = client.submit(
            **request.model_dump(exclude={"experiment_workspace"}),
            experiment_workspace=Path(args.workspace),
        )
    _print(result)
    return 0 if result.get("status") != "failed" else 3


def _handle_trial(args: argparse.Namespace) -> int:
    with SubmitBackendClient(load_backend_config(args.config)) as client:
        try:
            info = client.trial_info(args.trial_id)
        except BackendError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    _print(info)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="expos.backends",
        description="ExpOS backend server + client CLI",
    )
    parser.add_argument("--config", default=None, help="Path to backend config.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # server ...
    server = subparsers.add_parser("server", help="Backend server controls")
    server_subs = server.add_subparsers(dest="server_cmd", required=True)
    start = server_subs.add_parser("start", help="Run the FastAPI server via uvicorn")
    start.add_argument("--host", default=None)
    start.add_argument("--port", type=int, default=None)
    start.add_argument("--reload", action="store_true")
    start.set_defaults(handler=_handle_server_start)

    # health
    health = subparsers.add_parser("health", help="Ping /health on the backend")
    health.set_defaults(handler=_handle_health)

    # submit
    submit = subparsers.add_parser("submit", help="Fire POST /submit against the backend")
    submit.add_argument("--experiment-id", required=True)
    submit.add_argument("--worker-id", required=True)
    submit.add_argument(
        "--benchmark",
        default="gsm8k",
        choices=["gsm8k", "aime_2025", "arena_hard_writing", "bfcl", "gpqa_main", "healthbench_easy", "human_eval"],
    )
    submit.add_argument("--train-data", required=True, dest="train_data")
    submit.add_argument(
        "--training-config-path",
        required=True,
        dest="training_config_path",
        help="Absolute path to a complete LLaMA-Factory YAML.",
    )
    submit.add_argument(
        "--training-mode",
        default="llama_factory_sft",
        choices=["llama_factory_sft"],
        help="llama_factory_sft = 全参 SFT (唯一 llama_factory 训练方式; LoRA 已下线).",
    )
    submit.add_argument("--workspace", required=True, help="Experiment workspace path")
    submit.set_defaults(handler=_handle_submit)

    # trial
    trial = subparsers.add_parser("trial", help="Fetch a trial receipt from the backend")
    trial.add_argument("--trial-id", required=True)
    trial.set_defaults(handler=_handle_trial)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 1
    return handler(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
