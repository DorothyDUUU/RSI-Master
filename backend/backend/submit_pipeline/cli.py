"""``python -m backend.backend.submit_pipeline`` — serverless submit-pipeline CLI.

用法 (agent workspace 内, 经 ``public/pipeline-submit`` launcher)::

    # 提交一次训练 (llama_factory): 渲染 pipeline.sh -> 提交 gpu_backend -> 打印回执
    public/pipeline-submit submit \\
        --data train.jsonl --train-config train.yaml \\
        --mode llama_factory_sft --bench-id gpqa --limit 100

    # 提交一次 slime 训练 (model_dir/torch_dist_dir 等走 slime 配置 YAML)
    public/pipeline-submit submit \\
        --data train.parquet --train-config slime.yaml \\
        --mode slime_sft --bench-id gpqa

    # 查实验状态 / 日志 / 列表
    public/pipeline-submit status exp_0005
    public/pipeline-submit log exp_0005
    public/pipeline-submit list

设计: 无 HTTP server —— CLI 直接 渲染 pipeline.sh + 调 gpu_backend 异步提交。
每个实验在 ``--experiments-dir`` (默认 ``./experiments``) 下建 ``exp_####/``,
含 ``work_order.json`` / ``config.json`` / ``pipeline.sh`` / ``result.json`` (EXIT
trap 写终态)。返回的 experiment_id 就是 agent 写进 FINISH 的值。
"""

from __future__ import annotations

import os

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from .constants import BENCHMARK_ROOT
from .models import SubmitRequest, SubmitResponse
from .render import build_pipeline_script


def _next_experiment_id(experiments_dir: Path) -> str:
    """Scan ``exp_%04d`` dirs and return the next id (``exp_0001`` if none)."""
    if not experiments_dir.is_dir():
        return "exp_0001"
    max_n = 0
    for p in experiments_dir.glob("exp_*"):
        if p.is_dir():
            try:
                max_n = max(max_n, int(p.name.rsplit("_", 1)[1]))
            except (ValueError, IndexError):
                continue
    return f"exp_{max_n + 1:04d}"


def _write_artifacts(request: SubmitRequest, workspace: Path, key_paths: dict[str, str]) -> None:
    """Write work_order.json / config.json / placeholder result.json into exp dir."""
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "work_order.json").write_text(
        json.dumps(
            {
                "schema_version": "rsi_master.work_order.v2",
                "experiment_id": request.experiment_id,
                "worker_id": request.worker_id,
                "benchmark": request.benchmark,
                "objective": request.objective,
                "hypothesis": request.hypothesis,
                "changed_variable": request.changed_variable,
                "parent_experiment_id": request.parent_experiment_id,
                "training_config_path": request.training_config_path,
                "training_mode": request.training_mode,
                "train_data_path": request.train_data_path,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (workspace / "config.json").write_text(
        json.dumps(
            {
                "experiment_id": request.experiment_id,
                "status": "running",
                "score": None,
                "failure_reason": "",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    # placeholder result.json: pipeline EXIT trap 会覆盖为终态; 占位让 round-end
    # 评测在管线尚未落终态时能识别"实验已存在、在跑"。
    (workspace / "result.json").write_text(
        json.dumps(
            {
                "schema_version": "rsi_master.result.v2",
                "experiment_id": request.experiment_id,
                "worker_id": request.worker_id,
                "status": "pending",
                "eval_status": "pending",
                "benchmark": request.benchmark,
                "training_mode": request.training_mode,
                "score": None,
                "failure_reason": "",
                "submit_result": {
                    "status": "pending",
                    "result_json_path": str(key_paths["result_json_path"]),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_receipt(workspace: Path, response: "SubmitResponse") -> None:
    """Persist the async submit receipt into config.json for round-end polling.

    ``_wait_for_experiment`` (rollout.eval_state) reads ``gpu_script_id`` here as
    a fast-path signal that the gpu script terminated.  config.json is
    system-owned runtime state — the pipeline bash never overwrites it (only
    result.json, via the EXIT trap), so this is race-free.
    """
    cfg_path = workspace / "config.json"
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        cfg = {}
    cfg.update({
        "status": "running",
        "gpu_script_id": response.gpu_script_id,
        "gpu_log_path": response.gpu_log_path,
        "submitted_at": None,  # filled by pipeline result.json created_at
    })
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_slime_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"cannot read slime config {path}: {exc}")
    if not isinstance(cfg, dict):
        raise SystemExit(f"slime config {path} must be a YAML dict")
    return cfg


def _cmd_submit(args: argparse.Namespace) -> int:
    experiments_dir = Path(args.experiments_dir).resolve()
    experiment_id = args.experiment_id or _next_experiment_id(experiments_dir)
    workspace = experiments_dir / experiment_id

    slime_config = _load_slime_config(args.slime_config)

    eval_all: dict[str, Any] = {
        "split": args.split or "",
        "limit": args.limit,
        "max_tokens": args.max_tokens or 0,
        "max_workers": args.max_workers or 0,
        "serve_gpu_memory_utilization": args.serve_util,
        "serve_max_model_len": args.max_model_len,
        "bmk_root": args.bmk_root,
    }
    if args.vllm_port:
        eval_all["vllm_port"] = args.vllm_port

    request = SubmitRequest(
        experiment_id=experiment_id,
        worker_id=args.worker_id or "worker_0001",
        benchmark=args.benchmark or (args.bench_id or "rollout"),
        train_data_path=str(Path(args.data).resolve()),
        training_config_path=str(Path(args.train_config).resolve()),
        training_mode=args.mode,
        experiment_workspace=str(workspace),
        gpu_nums=args.gpu_nums,
        need_exclusive_gpu=not args.share_gpu,
        objective=args.objective or "",
        hypothesis=args.hypothesis or "",
        changed_variable=args.changed_variable or "",
        parent_experiment_id=args.parent_experiment_id,
        bench_id=args.bench_id,
        eval_all=eval_all,
        slime_config=slime_config,
    )

    try:
        pipeline_script, key_paths = build_pipeline_script(request, workspace)
    except (OSError, ValueError) as exc:
        print(json.dumps({
            "status": "failed",
            "experiment_id": experiment_id,
            "failure_reason": f"failed to render pipeline.sh: {exc}",
        }, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1

    _write_artifacts(request, workspace, key_paths)

    from .driver import run_trial
    response = run_trial(request)
    if response.status == "failed":
        print(json.dumps(response.to_dict(), indent=2, ensure_ascii=False), file=sys.stderr)
        return 1
    _write_receipt(workspace, response)
    _print_response(response)
    return 0


def _read_result_json(workspace: Path) -> dict[str, Any] | None:
    p = workspace / "result.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _resolve_workspace(experiments_dir: Path, experiment_id: str) -> Path | None:
    workspace = Path(experiments_dir).resolve() / experiment_id
    return workspace if workspace.is_dir() else None


def _cmd_status(args: argparse.Namespace) -> int:
    workspace = _resolve_workspace(args.experiments_dir, args.experiment_id)
    if workspace is None:
        print(json.dumps({"status": "not_found", "experiment_id": args.experiment_id}))
        return 1
    result = _read_result_json(workspace)
    if result is None:
        print(json.dumps({"status": "pending", "experiment_id": args.experiment_id,
                          "result_json": "missing"}))
        return 0
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    experiments_dir = Path(args.experiments_dir).resolve()
    rows = []
    if experiments_dir.is_dir():
        for workspace in sorted(experiments_dir.glob("exp_*"), reverse=True):
            if not workspace.is_dir():
                continue
            result = _read_result_json(workspace)
            rows.append({
                "experiment_id": workspace.name,
                "status": (result or {}).get("status", "unknown"),
                "score": (result or {}).get("score"),
                "eval_status": (result or {}).get("eval_status"),
            })
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return 0


def _cmd_health(args: argparse.Namespace) -> int:
    # Lazy import: gpu_backend 的 package __init__ 会拉起 server (psutil 等), 只在
    # health 真正需要它时才引入 —— status/list/submit 不依赖 gpu_backend 可 import。
    from backend.gpu_backend.client import BackendClient
    from backend.gpu_backend.config import get_backend_url

    try:
        with BackendClient(base_url=get_backend_url(), timeout=10.0) as client:
            print(json.dumps(client.health_check(), indent=2, ensure_ascii=False, default=str))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "unreachable", "error": str(exc)}), file=sys.stderr)
        return 1


def _print_response(response: SubmitResponse) -> None:
    """Human + machine-readable receipt: agent reads experiment_id to write FINISH."""
    print(json.dumps({
        "status": response.status,
        "experiment_id": response.trial_path.rsplit("/", 1)[-1],
        "gpu_script_id": response.gpu_script_id,
        "result_json_path": response.result_json_path,
        "checkpoint_dir": response.checkpoint_dir,
        "tensorboard_dir": response.tensorboard_dir,
        "eval_dir": response.eval_dir,
        "pipeline_script_path": response.pipeline_script_path,
        "gpu_log_path": response.gpu_log_path,
        "summary": response.summary,
    }, indent=2, ensure_ascii=False, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="submit_pipeline",
        description="submit-train pipeline: render pipeline.sh + submit to gpu_backend (async).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_submit = sub.add_parser("submit", help="submit a training pipeline")
    p_submit.add_argument("--experiments-dir", default="./experiments", help="experiments root (default ./experiments)")
    p_submit.add_argument("--experiment-id", default="", help="override auto-assigned exp_%04d")
    p_submit.add_argument("--data", required=True, help="raw training data (jsonl/parquet)")
    p_submit.add_argument("--train-config", required=True, help="training config YAML (LF) / slime config YAML")
    p_submit.add_argument("--mode", choices=["llama_factory_sft", "slime_sft"], default="llama_factory_sft")
    p_submit.add_argument("--bench-id", default="", help="benchmark suite format/<id>/ name (required for eval)")
    p_submit.add_argument("--benchmark", default="", help="report label; defaults to bench-id")
    p_submit.add_argument("--worker-id", default="", help="worker label for work_order.json")
    p_submit.add_argument("--limit", type=int, default=500, help="eval_all sample limit (default 500)")
    p_submit.add_argument("--max-tokens", type=int, default=0, help="eval_all max_tokens override (0 = benchmark suite config)")
    p_submit.add_argument("--max-workers", type=int, default=0, help="eval_all max_workers override")
    p_submit.add_argument("--split", default="", help="eval_all split override")
    p_submit.add_argument("--serve-util", type=float, default=0.6, help="vLLM gpu-memory-utilization")
    p_submit.add_argument("--max-model-len", type=int, default=32768, help="vLLM max-model-len")
    p_submit.add_argument("--vllm-port", type=int, default=0, help="fixed vLLM port (default random high port)")
    p_submit.add_argument("--bmk-root", default=os.environ.get("BENCHMARK_ROOT", ""))
    p_submit.add_argument("--slime-config", default="", help="slime training YAML (mode=slime_sft)")
    p_submit.add_argument("--gpu-nums", type=int, default=1)
    p_submit.add_argument("--share-gpu", action="store_true", help="non-exclusive GPU lease")
    p_submit.add_argument("--objective", default="")
    p_submit.add_argument("--hypothesis", default="")
    p_submit.add_argument("--changed-variable", default="")
    p_submit.add_argument("--parent-experiment-id", default="")
    p_submit.set_defaults(func=_cmd_submit)

    p_status = sub.add_parser("status", help="print an experiment's result.json")
    p_status.add_argument("experiment_id")
    p_status.add_argument("--experiments-dir", default="./experiments")
    p_status.set_defaults(func=_cmd_status)

    p_list = sub.add_parser("list", help="list experiments under the experiments dir")
    p_list.add_argument("--experiments-dir", default="./experiments")
    p_list.set_defaults(func=_cmd_list)

    p_health = sub.add_parser("health", help="gpu_backend health")
    p_health.set_defaults(func=_cmd_health)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
