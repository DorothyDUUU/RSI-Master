"""``expos worker ...`` subcommands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .context import CLIContext
from .output import emit_error, emit_json, emit_yaml_like


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("worker", help="Worker-scoped CLI")
    worker_subs = parser.add_subparsers(dest="worker_cmd", required=True)

    show = worker_subs.add_parser("show-datapool", help="List every data_pool entry")
    show.set_defaults(handler=_handle_show_datapool)

    inspect_data = worker_subs.add_parser("inspect-data", help="Show one data_pool entry")
    inspect_data.add_argument("--data-id", required=True)
    inspect_data.set_defaults(handler=_handle_inspect_data)

    update_readme = worker_subs.add_parser(
        "update-readme", help="Append a note to a data_pool README"
    )
    update_readme.add_argument("--data-id", required=True)
    update_readme.add_argument("--summary", required=True, help="Text to append")
    update_readme.set_defaults(handler=_handle_update_readme)

    add_data = worker_subs.add_parser("add-data", help="Register a new data_pool entry")
    add_data.add_argument("--worker-id", default=None)
    add_data.add_argument("--source-experiment-id", default=None)
    add_data.add_argument("--benchmark", default="")
    add_data.add_argument("--artifact-path", default=None, help="File to copy into the pool")
    add_data.add_argument("--artifact-type", default="train_data")
    add_data.add_argument("--num-rows", type=int, default=None)
    add_data.add_argument("--score", type=float, default=None)
    add_data.add_argument("--tags", nargs="*", default=None)
    add_data.add_argument("--readme", default=None, help="Optional custom README body")
    add_data.add_argument("--data-id", default=None, help="Optional explicit data_id")
    add_data.add_argument(
        "--synthesis-funnel",
        default=None,
        help=(
            'Optional JSON object with the data-synthesis funnel counts, e.g. '
            '{"generated":1777,"struct_ok":1752,"exec_pass":1367,"decontam_kept":554,'
            '"dedup_kept":554}. Lands in the data_added event payload (traj_analysis D4).'
        ),
    )
    add_data.set_defaults(handler=_handle_add_data)

    submit = worker_subs.add_parser("submit", help="Trigger a training + eval trial")
    submit.add_argument("--worker-id", required=True)
    submit.add_argument("--data-path", required=True, help="Absolute path to training data")
    submit.add_argument(
        "--training-config-path",
        required=True,
        dest="training_config_path",
        help=(
            "The single config channel, interpreted per training_mode. "
            "llama_factory modes: absolute path to a complete LLaMA-Factory "
            "training YAML; all hyper-parameter changes must be baked into the "
            "file itself (no inline dict overrides). slime_sft: the SAME field "
            "holds a slime config file (YAML/JSON) — model_dir/torch_dist_dir "
            "required; num_epoch, num_gpus, tp_size, ep_size, has_nvlink, "
            "output_dir, ckpt_dir. Parsed and merged over run defaults; "
            "overrides of framework-managed defaults are warned, not blocked."
        ),
    )
    submit.add_argument(
        "--training-mode",
        required=True,
        choices=["llama_factory_sft", "slime_sft"],
        help=(
            "Training pipeline variant. llama_factory_sft = full-parameter SFT "
            "(the only allowed llama_factory mode; **LoRA 已下线** 2026-08-20), "
            "training YAML must set finetuning_type: full; the trained checkpoint "
            "is served directly. slime_sft (Megatron-LM) runs input-data preflight "
            "+ training only via launch_sft.sh; serve/eval 本期跳过 "
            "(eval_status=skipped)."
        ),
    )
    submit.add_argument(
        "--slime-config",
        default=None,
        metavar="JSON",
        help=(
            "(可选, 仅 slime_sft) 训练配置的**内联覆盖** (JSON object), 键映射到 "
            "launch_sft.sh 的 env: model_dir / torch_dist_dir (必填), num_epoch, "
            "num_gpus, tp_size, ep_size, has_nvlink, output_dir, ckpt_dir。"
            "正常情况应把配置写进 --training-config-path 文件; 此参数只在调试时覆盖 "
            "文件里同名键。data_file 由 submit 自动指到净化副本, 无需传入。"
        ),
    )
    submit.add_argument("--objective", default="")
    submit.add_argument("--hypothesis", default="")
    submit.add_argument("--changed-variable", default="")
    submit.add_argument("--parent-experiment-id", default=None)
    submit.add_argument(
        "--gpu-nums",
        type=int,
        default=None,
        help="Number of GPUs this trial needs. 缺省 (不传) 按 base_model 命中档位默认回填 "
        "(4B=2, 8B=1, 35B=4); 显式传值逐位校验。gpu-backend picks the card; "
        "do NOT set CUDA_VISIBLE_DEVICES yourself.",
    )
    submit.add_argument(
        "--need-exclusive-gpu",
        action="store_true",
        dest="need_exclusive_gpu",
        default=True,
        help=(
            "独占整卡到任务 tmux 会话结束 (gpu-backend --need-exclusive-gpu)。"
            "默认 True (2026-08-29): 长挂训练+serve+eval 不被抢占; 需要共享时才"
            "显式加 --no-need-exclusive-gpu。"
        ),
    )
    submit.add_argument(
        "--no-need-exclusive-gpu",
        action="store_false",
        dest="need_exclusive_gpu",
        default=True,
        help="退化为共享模式 (不独占整卡)。",
    )
    submit.set_defaults(handler=_handle_submit)

    add_reviewer = worker_subs.add_parser(
        "add-reviewer", help="Create a reviewer node bound to the given experiments"
    )
    add_reviewer.add_argument("--worker-id", required=True)
    add_reviewer.add_argument(
        "--experiment-ids", nargs="+", required=True, help="One or more experiment_ids"
    )
    add_reviewer.add_argument(
        "--objective",
        default="",
        help="What the reviewer should focus on / verify for these experiments",
    )
    add_reviewer.set_defaults(handler=_handle_add_reviewer)

    finalize = worker_subs.add_parser("finalize", help="Mark worker as finished/aborted")
    finalize.add_argument("--worker-id", required=True)
    finalize.add_argument("--status", default="finished")
    finalize.add_argument("--notes", default="")
    finalize.set_defaults(handler=_handle_finalize)


# ── Handlers ─────────────────────────────────────────────────────────────────


def _handle_show_datapool(ctx: CLIContext, args: argparse.Namespace) -> int:
    payload = {"entries": ctx.worker.show_datapool()}
    if args.json:
        emit_json(payload)
    else:
        emit_yaml_like(payload)
    return 0


def _handle_inspect_data(ctx: CLIContext, args: argparse.Namespace) -> int:
    try:
        result = ctx.worker.inspect_data(args.data_id)
    except KeyError as exc:
        return emit_error(str(exc), code=2)
    if args.json:
        emit_json(result)
    else:
        emit_yaml_like(result)
    return 0


def _handle_update_readme(ctx: CLIContext, args: argparse.Namespace) -> int:
    try:
        result = ctx.worker.update_readme(args.data_id, args.summary)
    except KeyError as exc:
        return emit_error(str(exc), code=2)
    if args.json:
        emit_json(result)
    else:
        emit_yaml_like(result)
    return 0


def _handle_add_data(ctx: CLIContext, args: argparse.Namespace) -> int:
    synthesis_funnel = None
    if args.synthesis_funnel:
        try:
            synthesis_funnel = json.loads(args.synthesis_funnel)
        except json.JSONDecodeError as exc:
            return emit_error(f"--synthesis-funnel is not valid JSON: {exc}", code=2)
    try:
        result = ctx.worker.add_data(
            worker_id=args.worker_id,
            source_experiment_id=args.source_experiment_id,
            benchmark=args.benchmark,
            artifact_path=args.artifact_path,
            artifact_type=args.artifact_type,
            num_rows=args.num_rows,
            score=args.score,
            tags=args.tags,
            readme=args.readme,
            data_id=args.data_id,
            synthesis_funnel=synthesis_funnel,
        )
    except FileNotFoundError as exc:
        return emit_error(str(exc), code=2)
    if args.json:
        emit_json(result)
    else:
        emit_yaml_like(result)
    return 0


def _handle_submit(ctx: CLIContext, args: argparse.Namespace) -> int:
    cfg_path = Path(args.training_config_path).expanduser()
    if not cfg_path.exists():
        return emit_error(
            f"training-config-path does not exist: {cfg_path}", code=2
        )
    try:
        slime_config: dict[str, Any] | None = None
        if args.slime_config:
            try:
                parsed = json.loads(args.slime_config)
            except json.JSONDecodeError as exc:
                return emit_error(f"--slime-config is not valid JSON: {exc}", code=2)
            if not isinstance(parsed, dict):
                return emit_error("--slime-config must be a JSON object", code=2)
            slime_config = parsed
        result = ctx.worker.submit(
            worker_id=args.worker_id,
            data_path=args.data_path,
            training_config_path=str(cfg_path),
            training_mode=args.training_mode,
            objective=args.objective,
            hypothesis=args.hypothesis,
            changed_variable=args.changed_variable,
            parent_experiment_id=args.parent_experiment_id,
            gpu_nums=args.gpu_nums,
            need_exclusive_gpu=args.need_exclusive_gpu,
            slime_config=slime_config,
        )
    except (KeyError, ValueError) as exc:
        # KeyError: unknown worker_id; ValueError: slime_sft 下 training-config-path
        # 解析失败 (文件缺失 / 非 YAML-JSON / 不是 mapping)。
        return emit_error(str(exc), code=2)
    if args.json:
        emit_json(result)
    else:
        emit_yaml_like(result)
    return 0 if result.get("status") != "failed" else 3


def _handle_add_reviewer(ctx: CLIContext, args: argparse.Namespace) -> int:
    try:
        result = ctx.worker.add_reviewer(
            worker_id=args.worker_id,
            experiment_ids=args.experiment_ids,
            objective=args.objective,
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
        result = ctx.worker.finalize(args.worker_id, status=args.status, notes=args.notes)
    except KeyError as exc:
        return emit_error(str(exc), code=2)
    if args.json:
        emit_json(result)
    else:
        emit_yaml_like(result)
    return 0


