"""Render ``pipeline.sh`` for a submit request, dispatching on training_mode.

- ``llama_factory_sft`` -> llama-factory 完整管线
  (预检 -> 全参 SFT -> vLLM serve -> eval), 与旧单文件逐字节等价。LoRA 已下线
  (2026-08-20), 训练 YAML ``finetuning_type: lora/qlora`` 在渲染期被拒。
- ``slime_sft`` -> 数据预检 (格式校验) + 训练委托 ``launch_sft.sh`` (env 覆盖)
  + stage-1.5 torch_dist->HF 转换 + vLLM serve + eval (与 llama-factory 共用
  stage-2/3)。``slime_config`` 的 ``model_dir`` / ``torch_dist_dir`` 必填, 缺了
  render 期直接 ValueError; ``skip_eval: true`` 可退回只训练不评测。
"""

from __future__ import annotations

import os

import logging
import stat
from pathlib import Path
from typing import Any

from .constants import (
    BENCHMARK_ROOT,
    _BENCHMARK_EVAL,
    _DECONTAM_SCRIPT,
    _DEFAULT_EVAL_ALL_LIMIT,
    _DEFAULT_LIMIT,
    _LLAMA_FACTORY_CWD,
    _LLAMA_FACTORY_ENV,
    _LLAMA_FACTORY_PREFLIGHT,
    _LLAMA_FACTORY_PYTHON,
    _LLAMA_FACTORY_TEMPLATE_PREFLIGHT,
    _SLIME_CONVERT_TO_HF,
    _SLIME_LAUNCH_SFT,
    _SLIME_MEGATRON_LM,
    _SLIME_PREFLIGHT,
    _SLIME_PYTHON,
    _VLLM_ENV,
    _VLLM_SERVE_GUARD,
)
from .fragments import compose_llama_factory, compose_slime
from .models import SubmitRequest
from .stages.serve_eval import (
    _benchmark_eval_cfg,
    _judge_exports,
    _pick_vllm_port,
    _serve_args,
)

logger = logging.getLogger(__name__)

#: LoRA/QLoRA 已下线 (2026-08-20): 只拒这两个 adapter 类型。``freeze`` 产出完整
#: 权重 checkpoint、vLLM 可直接 serve, 保留允许。
_LOFA_FINETUNING_TYPES = {"lora", "qlora"}


def _reject_lora_finetuning(training_config_path: str) -> None:
    """守卫 (b): 训练 YAML ``finetuning_type`` ∈ {lora, qlora} 时拒绝提交。

    放在渲染层是因为 ``POST /submit`` 的 HTTP server 路径绕过
    ``WorkerService.submit``, 这里才是 driver / server / server-stub /
    ``validate_on_host`` / 测试全部路径的单点收口。YAML 缺失/不可读 -> 跳过
    (stage-0 数据预检本就覆盖该错误, 勿二次拦截)。
    """
    try:
        import yaml

        cfg = yaml.safe_load(Path(training_config_path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — 读不到/不可解析 → 跳过守卫
        return
    if not isinstance(cfg, dict):
        return
    ft = str(cfg.get("finetuning_type") or "").strip().lower()
    if ft in _LOFA_FINETUNING_TYPES:
        raise ValueError(
            f"训练 YAML `finetuning_type: {ft}` 已下线 (2026-08-20, LoRA 停用): "
            "训练只能走全参 SFT —— 请改 `finetuning_type: full`"
        )


def _as_bool(value: Any, default: bool = False) -> bool:
    """slime_config 的布尔键宽松解析 (yaml bool / 字符串 / 缺失)."""
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _as_int(value: Any, default: int = 0) -> int:
    """slime_config 的整型键宽松解析 (缺失/不可解析 -> default)."""
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _build_eval_config(
    request: SubmitRequest,
) -> tuple[dict[str, Any], str, int, str]:
    """stage-3 评测派发: 返回 (eval_cfg, eval_args, limit, judge_exports).

    ``request.bench_id`` 非空 -> benchmark suite 统一评测 (eval_all), 否则按
    ``request.benchmark`` 分立评测。llama-factory 与 slime 路径共用。
    """
    if request.bench_id:
        eval_cfg = dict(_BENCHMARK_EVAL["eval_all"])
        ea = request.eval_all or {}
        bench_id = str(request.bench_id)
        eval_args = f'--bench-id "{bench_id}"'
        split = ea.get("split")
        if split:
            eval_args += f' --split "{split}"'
        # trial_name 用 experiment_id: 满足 run_eval.py 的 SAFE_NAME, 且让
        # benchmark suite 的 data/output/<id>/<trial> 与实验一一对应。
        eval_args += f' --trial-name "{request.experiment_id}"'
        if ea.get("bmk_root"):
            eval_args += f' --bmk-root "{ea["bmk_root"]}"'
        try:
            limit = int(ea.get("limit") or 0)
        except (TypeError, ValueError):
            limit = 0
        if limit <= 0:
            limit = _DEFAULT_EVAL_ALL_LIMIT  # 正数; 绝不传 -1.
        for knob, key in (("max_tokens", "max_tokens"), ("max_workers", "max_workers")):
            raw = ea.get(knob)
            if not raw:
                continue
            try:
                eval_cfg[key] = int(raw)
            except (TypeError, ValueError):
                logger.warning("eval_all %s=%r ignored (not an int)", knob, raw)
        # LLM judge: 显式 ``evaluation.eval_all.judge_env`` > 按 bench_id 自动匹配
        # _BENCHMARK_EVAL 已知需要 judge 的 bench (healthbench_easy /
        # arena_hard_writing, 与 benchmark suite bench_id 同名). 凭据来自
        # ``evaluation.eval_all.judge`` (base_url/api_key), 优先于渲染进程 env.
        known_judge_benches = {
            "healthbench_easy",
            "arenahardwriting",      # benchmark suite canonical id
            "arena_hard_writing",    # historical per-benchmark id
        }
        effective_judge_env = bool(
            ea.get(
                "judge_env",
                _BENCHMARK_EVAL.get(bench_id, {}).get(
                    "judge_env", bench_id in known_judge_benches
                ),
            )
        )
        judge_creds = ea.get("judge") or {}
        judge_exports = _judge_exports(
            bench_id, {"judge_env": effective_judge_env}, creds=judge_creds
        )
    else:
        eval_cfg = _benchmark_eval_cfg(request.benchmark)
        eval_args = eval_cfg["extra_args"]
        limit = _DEFAULT_LIMIT
        judge_exports = _judge_exports(request.benchmark, eval_cfg)
    return eval_cfg, eval_args, limit, judge_exports


def _render_llama_factory(request: SubmitRequest, workspace: Path) -> str:
    """llama-factory 路径: stage-0 预检 + SFT + serve + eval, 单次 substitute."""
    _reject_lora_finetuning(request.training_config_path)
    checkpoint_dir = workspace / "checkpoints"
    tensorboard_dir = workspace / "tensorboard"
    eval_dir = workspace / "eval"
    result_json = workspace / "result.json"
    pipeline_log = workspace / "pipeline.log"
    port = _pick_vllm_port(
        request.experiment_id,
        override=request.eval_all.get("vllm_port"),
    )

    eval_cfg, eval_args, limit, judge_exports = _build_eval_config(request)
    serve_gpu_util, serve_max_model_len = _serve_args(request, eval_cfg)

    # 数据集目录由 stage-0 去污在运行时生成 (decontam_stage.py 自动建注册表):
    # <workspace>/llama_factory_data/{decontaminated_train.jsonl, dataset_info.json},
    # stage-1 以 dataset_dir=$DATASET_DIR 消费。替代 ExpOS 的渲染期 staging。
    dataset_dir = workspace / "llama_factory_data"
    ea = request.eval_all or {}
    # serve env (经 eval_all 透传): 默认 _VLLM_ENV = vllm_env_0819 (transformers
    # 5.15.1), 与训练 env (llama_factory_env 5.8.0) 同属 5.x, tokenizer 格式兼容,
    # 训练产物原样 serve, 无 tokenizer 恢复步骤. ``vllm_env`` 可指向其它 env.
    # stage-0b 模板-eos 预检默认开启; 0819 后已是**咨询性** (qwen3/qwen3_nothink
    # 分叉不再拒绝, 只打印说明), 无需为 replace_eos 模板跳过.
    # ``skip_template_preflight: true`` 仍保留为兜底逃生口 (预检脚本缺失/临时问题).
    vllm_env = str(ea.get("vllm_env") or "").strip() or _VLLM_ENV
    skip_preflight = bool(ea.get("skip_template_preflight"))
    return compose_llama_factory(
        include_template_preflight=not skip_preflight,
    ).substitute(
        EXPERIMENT_ID=request.experiment_id,
        WORKER_ID=request.worker_id,
        BENCHMARK=request.benchmark,
        TRAINING_MODE=request.training_mode,
        TRAINING_CONFIG_PATH=str(request.training_config_path),
        TRAIN_DATA_PATH=str(request.train_data_path),
        WORKSPACE=str(workspace),
        CHECKPOINT_DIR=str(checkpoint_dir),
        TENSORBOARD_DIR=str(tensorboard_dir),
        EVAL_DIR=str(eval_dir),
        RESULT_JSON=str(result_json),
        PIPELINE_LOG=str(pipeline_log),
        VLLM_PORT=str(port),
        VLLM_GPU_MEM_UTIL=str(serve_gpu_util),
        VLLM_MAX_MODEL_LEN=str(serve_max_model_len),
        # 单卡 dense 路径无 TP 拆分; 只有 slime 大模型 (35a3b) 才传 TP 参数.
        VLLM_TP_SIZE_ARGS="",
        LLAMA_FACTORY_ENV=_LLAMA_FACTORY_ENV,
        LLAMA_FACTORY_CWD=_LLAMA_FACTORY_CWD,
        DATASET_DIR=str(dataset_dir),
        PREFLIGHT_SCRIPT=_LLAMA_FACTORY_PREFLIGHT,
        TEMPLATE_PREFLIGHT=_LLAMA_FACTORY_TEMPLATE_PREFLIGHT,
        VLLM_ENV=vllm_env,
        VLLM_SERVE_GUARD=_VLLM_SERVE_GUARD,
        EVAL_SCRIPT_DIR=eval_cfg["eval_dir"],
        EVAL_PY=eval_cfg["eval_py"],
        METRIC_KEY=eval_cfg["metric_key"],
        EVAL_ARGS=eval_args,
        JUDGE_EXPORTS=judge_exports,
        MAX_TOKENS=str(eval_cfg["max_tokens"]),
        MAX_WORKERS=str(eval_cfg["max_workers"]),
        LIMIT=str(limit),
        # stage-0 去污 (decontamination): heavy python + 入口 + 评测目标。
        DECONTAM_PYTHON=_LLAMA_FACTORY_PYTHON,
        DECONTAM_SCRIPT=_DECONTAM_SCRIPT,
        BENCH_ID=str(request.bench_id or ""),
        BMK_ROOT=str(ea.get("bmk_root") or BENCHMARK_ROOT),
    )


def _resolve_slime_config(
    request: SubmitRequest, workspace: Path, ckpt_dir: Path
) -> dict[str, str]:
    """校验 ``slime_config`` 并展开成 launch_sft.sh 的 env 值 (字符串).

    ``model_dir`` / ``torch_dist_dir`` 必填 —— 训练 ``--load`` 必须是 torch_dist
    产物 (先跑 convert_hf_to_torch_dist.sh)。缺了这里直接抛错, 不等到 GPU 上。
    """
    slime_cfg = request.slime_config or {}
    model_dir = str(slime_cfg.get("model_dir") or "").strip()
    torch_dist_dir = str(slime_cfg.get("torch_dist_dir") or "").strip()
    if not model_dir:
        raise ValueError(
            "training_mode=slime_sft requires slime_config.model_dir (HF base weights)"
        )
    if not torch_dist_dir:
        raise ValueError(
            "training_mode=slime_sft requires slime_config.torch_dist_dir "
            "(torch_dist weights for --load; run convert_hf_to_torch_dist.sh first)"
        )
    # 模型架构 spec 脚本 (slime/scripts/models/*.sh)。可选: 不传时 launch_sft.sh
    # 用其内置默认 (35B MoE); 稠密模型 (如 Qwen3-8B) 必须传对应 dense spec。
    model_spec = str(slime_cfg.get("model_spec") or "").strip()
    gpu_nums = int(request.gpu_nums or slime_cfg.get("num_gpus") or 1)
    # Keep the fallback topology valid when the caller omits TP/EP/CP. The old
    # fixed TP=2, EP=4 default produced world_size=8 even for a 2-GPU request.
    tp_default = 1 if gpu_nums == 1 else 2
    ep_default = gpu_nums // tp_default if gpu_nums % tp_default == 0 else 1
    tp_size = int(slime_cfg.get("tp_size") or tp_default)
    ep_size = int(slime_cfg.get("ep_size") or ep_default)
    cp_size = int(slime_cfg.get("cp_size") or 1)
    world_size = tp_size * ep_size * cp_size
    if min(tp_size, ep_size, cp_size) < 1 or world_size != gpu_nums:
        raise ValueError(
            "slime parallelism must satisfy TP×EP×CP == gpu_nums: "
            f"{tp_size}×{ep_size}×{cp_size}={world_size}, gpu_nums={gpu_nums}"
        )
    # rollout/global batch、seq 等训练超参 (可配)。默认沿用 launch_sft.sh 内置值,
    # 但注意: 默认 rollout_batch_size=128 要求数据 >128 行, 小数据集 (infra-test
    # slime_rows=64) 必须显式传, 否则 num_rollout_per_epoch=0 -> AssertionError。
    # serve/eval 相关 (可选, 有默认): hf_dir 是 stage-1.5 转换落点;
    # vocab_size 去 Megatron embedding padding (0 = 不传); vllm_tp_size 大模型
    # MoE 多卡 serve 的 TP 拆分 (1 = 单卡); skip_eval 退回只训练不评测。
    return {
        "model_dir": model_dir,
        "torch_dist_dir": torch_dist_dir,
        "model_spec": model_spec,
        "output_dir": str(slime_cfg.get("output_dir") or workspace),
        "ckpt_dir": str(slime_cfg.get("ckpt_dir") or ckpt_dir),
        # stage-0 去污产物 (decontam_stage.py 写净化 parquet 到该固定路径),
        # 训练 DATA_FILE 永远指净化数据, 不碰原始 train_data_path。
        "data_file": str(workspace / "artifacts" / "decontaminated_train.parquet"),
        "num_epoch": str(slime_cfg.get("num_epoch") or 3),
        "num_gpus": str(gpu_nums),
        "tp_size": str(tp_size),
        "ep_size": str(ep_size),
        "has_nvlink": str(slime_cfg.get("has_nvlink") or 0),
        "rollout_batch_size": str(slime_cfg.get("rollout_batch_size") or 128),
        "global_batch_size": str(slime_cfg.get("global_batch_size") or 128),
        "seq_length": str(slime_cfg.get("seq_length") or 4096),
        "max_tokens_per_gpu": str(slime_cfg.get("max_tokens_per_gpu") or 4096),
        "cp_size": str(cp_size),
        "hf_dir": str(slime_cfg.get("hf_dir") or workspace / "hf_model"),
        "vocab_size": str(slime_cfg.get("vocab_size") or 0),
        "add_missing": str(slime_cfg.get("add_missing", True)),
        "skip_eval": str(slime_cfg.get("skip_eval", False)),
        "vllm_tp_size": str(slime_cfg.get("vllm_tp_size") or 1),
    }


def _render_slime(request: SubmitRequest, workspace: Path, ckpt_dir: Path) -> str:
    """slime 路径: 预检 + 训练 + torch_dist->HF 转换 + (默认) serve/eval.

    与 llama-factory 共用 stage-2/3 与评测派发 (_build_eval_config), 单次 substitute。
    ``slime_config.skip_eval=true`` 时退回只训练+转换 (eval_status=skipped)。
    """
    cfg = _resolve_slime_config(request, workspace, ckpt_dir)
    eval_cfg, eval_args, limit, judge_exports = _build_eval_config(request)
    port = _pick_vllm_port(
        request.experiment_id,
        override=request.eval_all.get("vllm_port"),
    )
    serve_gpu_util, serve_max_model_len = _serve_args(request, eval_cfg)
    ea = request.eval_all or {}
    vllm_env = str(ea.get("vllm_env") or "").strip() or _VLLM_ENV

    vocab_size = _as_int(cfg["vocab_size"], 0)
    vocab_args = f"--vocab-size {vocab_size}" if vocab_size else ""
    add_missing = "-a" if _as_bool(cfg["add_missing"], True) else ""
    tp_size = _as_int(cfg["vllm_tp_size"], 1)
    tp_args = f" --tensor-parallel-size {tp_size}" if tp_size > 1 else ""
    skip_eval = _as_bool(cfg["skip_eval"], False)

    return compose_slime(include_eval=not skip_eval).substitute(
        EXPERIMENT_ID=request.experiment_id,
        WORKER_ID=request.worker_id,
        BENCHMARK=request.benchmark,
        TRAINING_MODE=request.training_mode,
        TRAINING_CONFIG_PATH=str(request.training_config_path),
        TRAIN_DATA_PATH=str(request.train_data_path),
        WORKSPACE=str(workspace),
        CHECKPOINT_DIR=str(ckpt_dir),
        TENSORBOARD_DIR=str(workspace / "tensorboard"),
        EVAL_DIR=str(workspace / "eval"),
        RESULT_JSON=str(workspace / "result.json"),
        PIPELINE_LOG=str(workspace / "pipeline.log"),
        SLIME_PYTHON=_SLIME_PYTHON,
        SLIME_PREFLIGHT=_SLIME_PREFLIGHT,
        SLIME_LAUNCH_SFT=_SLIME_LAUNCH_SFT,
        SLIME_DATA_FILE=cfg["data_file"],
        SLIME_MODEL_DIR=cfg["model_dir"],
        SLIME_TORCH_DIST_DIR=cfg["torch_dist_dir"],
        SLIME_MODEL_SPEC=cfg["model_spec"],
        SLIME_OUTPUT_DIR=cfg["output_dir"],
        SLIME_CKPT_DIR=cfg["ckpt_dir"],
        SLIME_NUM_EPOCH=cfg["num_epoch"],
        SLIME_NUM_GPUS=cfg["num_gpus"],
        SLIME_TP_SIZE=cfg["tp_size"],
        SLIME_EP_SIZE=cfg["ep_size"],
        SLIME_HAS_NVLINK=cfg["has_nvlink"],
        SLIME_ROLLOUT_BATCH_SIZE=cfg["rollout_batch_size"],
        SLIME_GLOBAL_BATCH_SIZE=cfg["global_batch_size"],
        SLIME_SEQ_LENGTH=cfg["seq_length"],
        SLIME_MAX_TOKENS_PER_GPU=cfg["max_tokens_per_gpu"],
        SLIME_CP_SIZE=cfg["cp_size"],
        SLIME_HF_DIR=cfg["hf_dir"],
        SLIME_CONVERT_TO_HF=_SLIME_CONVERT_TO_HF,
        SLIME_MEGATRON_LM=_SLIME_MEGATRON_LM,
        SLIME_VOCAB_ARGS=vocab_args,
        SLIME_ADD_MISSING=add_missing,
        VLLM_PORT=str(port),
        VLLM_GPU_MEM_UTIL=str(serve_gpu_util),
        VLLM_MAX_MODEL_LEN=str(serve_max_model_len),
        VLLM_TP_SIZE_ARGS=tp_args,
        VLLM_ENV=vllm_env,
        VLLM_SERVE_GUARD=_VLLM_SERVE_GUARD,
        LLAMA_FACTORY_ENV=_LLAMA_FACTORY_ENV,
        EVAL_SCRIPT_DIR=eval_cfg["eval_dir"],
        EVAL_PY=eval_cfg["eval_py"],
        METRIC_KEY=eval_cfg["metric_key"],
        EVAL_ARGS=eval_args,
        JUDGE_EXPORTS=judge_exports,
        MAX_TOKENS=str(eval_cfg["max_tokens"]),
        MAX_WORKERS=str(eval_cfg["max_workers"]),
        LIMIT=str(limit),
        # stage-0 去污 (decontamination): slime 用 slime conda env 解释器.
        DECONTAM_SCRIPT=_DECONTAM_SCRIPT,
        BENCH_ID=str(request.bench_id or ""),
        BMK_ROOT=str(ea.get("bmk_root") or BENCHMARK_ROOT),
    )


def build_pipeline_script(
    request: SubmitRequest,
    workspace: Path,
) -> tuple[Path, dict[str, str]]:
    """Render pipeline.sh into ``workspace/`` and return (path, key_paths).

    ``key_paths`` 里带上了 checkpoint / tensorboard / eval / result_json /
    pipeline_log 的绝对路径, WorkerService 层需要把它们回填到 SubmitResponse。
    """
    workspace.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = workspace / "checkpoints"
    tensorboard_dir = workspace / "tensorboard"
    eval_dir = workspace / "eval"
    result_json = workspace / "result.json"
    pipeline_script = workspace / "pipeline.sh"
    pipeline_log = workspace / "pipeline.log"

    if request.training_mode == "slime_sft":
        ckpt_dir = Path(str((request.slime_config or {}).get("ckpt_dir") or checkpoint_dir))
        rendered = _render_slime(request, workspace, ckpt_dir)
        checkpoint_dir = ckpt_dir
        # stage-1.5 转换后的 HF 模型目录 (LF 无此产物, 给空串).
        hf_model_dir = str((request.slime_config or {}).get("hf_dir") or workspace / "hf_model")
    else:
        rendered = _render_llama_factory(request, workspace)
        hf_model_dir = ""

    pipeline_script.write_text(rendered, encoding="utf-8")
    pipeline_script.chmod(pipeline_script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)

    return pipeline_script, {
        "checkpoint_dir": str(checkpoint_dir),
        "tensorboard_dir": str(tensorboard_dir),
        "eval_dir": str(eval_dir),
        "result_json_path": str(result_json),
        "pipeline_script_path": str(pipeline_script),
        "pipeline_log_path": str(pipeline_log),
        "hf_model_dir": hf_model_dir,
    }
