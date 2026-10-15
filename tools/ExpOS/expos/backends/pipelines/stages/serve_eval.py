"""Stage-2 vLLM serve + stage-3 eval bash fragments, and the python helpers
that render them (eval dispatch, vLLM port/serve args, dataset staging).

llama-factory 路径使用这些片段; slime 本期 (preflight + 训练) 不进入 serve/eval。
"""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Any, Mapping

from ..constants import (
    _BENCHMARK_EVAL,
    _DEFAULT_EVAL_ALL_LIMIT,
    _DEFAULT_LIMIT,
    _DEFAULT_SERVE_GPU_MEM_UTIL,
    _DEFAULT_SERVE_MAX_MODEL_LEN,
    _DEFAULT_SERVE_MAX_MODEL_LEN_LARGE,
    _SERVE_MAX_MODEL_LEN_PROMPT_SLACK,
    _VLLM_PORT_BASE,
    _VLLM_PORT_MOD,
)
from ..recommendations import is_large_model
from ...models import SubmitRequest

logger = logging.getLogger(__name__)

_LONG_OUTPUT_BENCHMARKS = {"ifbench", "supergpqa", "arc-agi-1", "arc-agi-2"}
_LONG_OUTPUT_SERVE_MAX_MODEL_LEN = 131072


# --- stage 2: vLLM serve ------------------------------------------------------
# eos/stop 消费语义: vLLM 的 stop 集合 = tokenizer.eos_token_id (每次请求无条件
# 加进 _all_stop_token_ids) ∪ generation_config 的 eos ids (sampling_params.py::
# update_from_generation_config; 即使传 ``--generation-config vllm`` 也会 union,
# 因为它只中和 sampling 默认 temperature/top_p/max_new_tokens, 不阻止 eos union)。
# 因此模型在训练里被教会发的 turn-ender (== 训练产物 tokenizer.eos, replace_eos)
# 一出现 vllm 就停。default 模板 -> 产物 tokenizer/config/generation_config 三者
# 自洽 (全 base eos); qwen3/qwen3_nothink -> tokenizer 是 <|im_end|>、config 是
# base eos, 分叉仍无害 (tokenizer.eos 无条件在 stop 集)。真正导致 length 退化的是
# 训练侧: LoRA 已下线 (2026-08-20); 全参 SFT 容量/数据/epochs 不足时模型在长
# few-shot 里不发任何 stop token (0819 六模板实证: full-SFT 三模板 200/200 stop)。
FRAGMENT_STAGE2 = r"""
# --- stage 2: vLLM serve ----------------------------------------------------
log "stage-2: launching vLLM on port $$VLLM_PORT"
log "stage-2: serving $$SERVE_DIR"

# FlashInfer may JIT-compile kernels during vLLM startup and requires ninja.
# Check it before launching the background server so a broken serve env fails
# with a precise error instead of surfacing later as a generic dead process.
(
    # shellcheck disable=SC1090
    source "${VLLM_ENV}"
    command -v ninja >/dev/null
) || {
    FAILURE_REASON="stage-2: vLLM environment is missing required executable 'ninja'"
    log "$$FAILURE_REASON"
    exit 1
}

# serve 前对齐守卫 (H7): 用 serve env 自己的 python 校验 checkpoint 的 rope
# 配置格式能否被该 env 的 transformers 正确读取。0819 根因: 训练 (transformers
# 5.8) 写新式 rope_parameters, 旧 env (4.57) 读不懂 -> 静默回落 rope_theta 差
# 100x。守卫在 vllm serve 启动前拦截, 不兼容即 fail loudly, 不静默 serve 错模型。
if [[ ! -f "${VLLM_SERVE_GUARD}" ]]; then
    log "stage-2: WARN serve 对齐守卫缺失 (${VLLM_SERVE_GUARD}), 跳过 (部署滞后不卡轮)"
else
    (
        # shellcheck disable=SC1090
        source "${VLLM_ENV}"
        python "${VLLM_SERVE_GUARD}" --serve-dir "$$SERVE_DIR"
    ) >> "$$WORKSPACE/vllm.log" 2>&1 || {
        FAILURE_REASON="stage-2: checkpoint 与 serve env 不兼容 (rope 配置格式, 见 vllm.log)"
        log "$$FAILURE_REASON"
        exit 1
    }
    log "stage-2: serve 对齐守卫通过 (rope 配置与 serve env 兼容)"
fi

# 单独开 subshell activate vllm env, 避免与 llama-factory env 变量污染。
# ``${VLLM_TP_SIZE_ARGS}``: 大模型 (如 slime 35a3b MoE) 需要多卡 serve, render 时
# 渲染成 `` --tensor-parallel-size N``; dense 单卡路径为空串 (LF 不变)。
# ``${VLLM_PARSER_ARGS}`` (2026-08-28, 用户要求): tool-call-parser +
# enable-auto-tool-choice + reasoning-parser=qwen3, render 时按模型家族注入
# (qwen3-8b->hermes, qwen3.5/35a3b->qwen3_coder)。
(
    # shellcheck disable=SC1090
    source "${VLLM_ENV}"
    # The venv may have been copied from another host, leaving ``bin/vllm``
    # with a stale absolute shebang. Invoke its Python module directly so the
    # configured environment remains relocatable.
    VLLM_PYTHON="$$(dirname "${VLLM_ENV}")/python"
    # Minimal runtime images do not necessarily contain nvcc. vLLM 0.27 enables
    # FlashInfer's sampler by default and otherwise attempts a CUDA JIT during
    # warm-up, killing the server after the model has already loaded. Use the
    # built-in native sampler, which does not need a CUDA toolkit at runtime.
    export VLLM_USE_FLASHINFER_SAMPLER=0
    exec "$$VLLM_PYTHON" -m vllm.entrypoints.cli.main serve \
        "$$SERVE_DIR" --port "$$VLLM_PORT" --generation-config vllm \
        --gpu-memory-utilization "$$VLLM_GPU_MEM_UTIL" --max-model-len "$$VLLM_MAX_MODEL_LEN"${VLLM_TP_SIZE_ARGS}${VLLM_PARSER_ARGS}
) >> "$$WORKSPACE/vllm.log" 2>&1 &
VLLM_PID=$$!
log "stage-2: vllm PID=$$VLLM_PID (log: $$WORKSPACE/vllm.log)"

# 轮询等待 /v1/models 可用, 上限 ${VLLM_READY_TIMEOUT}s (默认 600s)。
# 35B MoE 冷启动 (权重加载 + torch.compile + CUDA graph 捕获) 实测可 >600s,
# 超时会误杀 —— 由 render 注入 VLLM_READY_TIMEOUT (vllm_ready_timeout 配置)。
BASE_URL="http://127.0.0.1:$$VLLM_PORT/v1"
READY=0
READY_ITERS=$$(( ${VLLM_READY_TIMEOUT} / 5 ))
for i in $$(seq 1 $$READY_ITERS); do
    if ! kill -0 "$$VLLM_PID" 2>/dev/null; then
        FAILURE_REASON="stage-2: vllm process died before serving (see vllm.log)"
        log "$$FAILURE_REASON"
        exit 1
    fi
    if curl -s -o /dev/null -w '%{http_code}' "$$BASE_URL/models" | grep -q '^200$$'; then
        READY=1
        log "stage-2: vllm ready after $$((i * 5))s"
        break
    fi
    sleep 5
done
if [[ "$$READY" -ne 1 ]]; then
    FAILURE_REASON="stage-2: vllm did not become ready within ${VLLM_READY_TIMEOUT}s"
    log "$$FAILURE_REASON"
    exit 1
fi
"""

# --- stage 3: benchmark eval ---------------------------------------------------
FRAGMENT_STAGE3 = r"""
# --- stage 3: benchmark eval -------------------------------------------------
${JUDGE_EXPORTS}
log "stage-3: ${EVAL_PY} eval starts (limit=$$LIMIT)"
# Re-activate llama_factory_env for eval (has datasets module; vllm_env does NOT)
# shellcheck disable=SC1090
source "${LLAMA_FACTORY_ENV}"
cd "${EVAL_SCRIPT_DIR}"
log "stage-3: using $$(which python) for eval"
STAGE3_STATUS=0
python ${EVAL_PY} \
    --model "$$SERVE_DIR" \
    --base-url "$$BASE_URL" \
    --api-key "sk-123" \
    ${EVAL_ARGS} \
    --max-tokens "$$MAX_TOKENS" \
    --temperature 0.0 \
    --max-workers "$$MAX_WORKERS" \
    --timeout 600 \
    --limit "$$LIMIT" \
    --run-dir "$$EVAL_DIR" \
    2>&1 | tee -a "$$STAGE_LOG" || STAGE3_STATUS=$$?
if [[ "$$STAGE3_STATUS" -ne 0 ]]; then
    FAILURE_REASON="stage-3 (${EVAL_PY}) exited with code $$STAGE3_STATUS"
    log "$$FAILURE_REASON"
    exit "$$STAGE3_STATUS"
fi

# 提取 metric_key (accuracy / pass_at_1 ...)
if [[ -f "$$EVAL_DIR/metrics.json" ]]; then
    SCORE="$$(/usr/bin/env python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$$EVAL_DIR/metrics.json" "$$METRIC_KEY")"
    EVAL_STATUS="completed"
    STATUS="completed"
    log "stage-3: extracted $${METRIC_KEY}=$$SCORE"
else
    FAILURE_REASON="stage-3: eval finished but metrics.json missing"
    log "$$FAILURE_REASON"
    exit 1
fi

log "pipeline: completed successfully."
exit 0
"""


def _benchmark_eval_cfg(benchmark: str) -> dict[str, Any]:
    """Return the stage-3 eval config for ``benchmark``, or raise loudly."""
    cfg = _BENCHMARK_EVAL.get(benchmark)
    if cfg is None:
        raise ValueError(
            f"unsupported benchmark {benchmark!r}; supported: {sorted(_BENCHMARK_EVAL)}"
        )
    return cfg


def _judge_exports(
    benchmark: str,
    cfg: dict[str, Any],
    creds: Mapping[str, Any] | None = None,
) -> str:
    """For judge-based benchmarks, bake OPENAI_BASE_URL/OPENAI_API_KEY into
    pipeline.sh so the eval script's openai SDK can reach the LLM judge.

    Generation still talks to the locally-served vLLM via ``BASE_URL``; these
    two exports only steer the judge calls. ``creds`` (run-config judge
    credentials, e.g. ``evaluation.eval_all.judge``) wins over the renderer's
    environment; if neither supplies anything, emit nothing (the GPU-machine
    env may provide them).
    """
    if not cfg.get("judge_env"):
        return ""
    creds = creds or {}
    base = str(creds.get("base_url") or "").strip() or os.environ.get("OPENAI_BASE_URL", "").strip()
    key = str(creds.get("api_key") or "").strip() or os.environ.get("OPENAI_API_KEY", "").strip()
    if not base and not key:
        logger.warning(
            "benchmark %r needs an LLM judge, but neither OPENAI_BASE_URL nor "
            "OPENAI_API_KEY is set in the run config or the driver env — judge "
            "calls will fail unless the GPU machine provides them.",
            benchmark,
        )
        return ""
    lines = ["# judge-based eval: point the openai SDK at the LLM judge sidecar"]
    if base:
        lines.append(f'export OPENAI_BASE_URL="{base}"')
    if key:
        lines.append(f'export OPENAI_API_KEY="{key}"')
    return "\n".join(lines)


def _pick_vllm_port(experiment_id: str, *, override: int | None = None) -> int:
    """Pick a vLLM serve port in [_VLLM_PORT_BASE, _VLLM_PORT_BASE + _VLLM_PORT_MOD).

    ``override`` (来自 ``eval_all.vllm_port``, 提交方 CLI 传入) 优先 —— 同一实验
    重跑/恢复时端口稳定可预期。未指定时随机挑一个: 并行 pipeline 各自随机, 避免
    并发 combo 撞同一端口。旧实现按 experiment_id digest 决定, 但并发 combo 共用
    ``exp_0001`` → 端口一样 → stage-2 vLLM EADDRINUSE 崩溃 → 健康检查连到邻 combo
    的 server → 404 model-tag 不匹配 (2026-08-23 slime-8b-nothink 实证)。

    ``experiment_id`` 保留仅用于可读日志/回溯 (gpu-backend 按 GPU 排队, 不同实验
    之间本就不需要同一端口)。
    """
    if override is not None:
        return int(override)
    return _VLLM_PORT_BASE + random.randrange(_VLLM_PORT_MOD)


def _serve_args(request: SubmitRequest, eval_cfg: dict[str, Any]) -> tuple[float, int]:
    """Compute stage-2 vLLM serve ``--gpu-memory-utilization`` / ``--max-model-len``.

    ``request.eval_all`` (来自 state["eval_all"] / config evaluation.eval_all) 可覆盖
    默认值; 否则按 benchmark 的 eval ``max_tokens`` 推导 max-model-len。
    """
    ea = request.eval_all or {}
    try:
        gpu_util = float(ea.get("serve_gpu_memory_utilization") or _DEFAULT_SERVE_GPU_MEM_UTIL)
    except (TypeError, ValueError):
        gpu_util = _DEFAULT_SERVE_GPU_MEM_UTIL
    if not (0.0 < gpu_util <= 1.0):
        logger.warning("serve_gpu_memory_utilization=%r invalid; using %.1f", gpu_util, _DEFAULT_SERVE_GPU_MEM_UTIL)
        gpu_util = _DEFAULT_SERVE_GPU_MEM_UTIL

    try:
        max_len = int(ea.get("serve_max_model_len") or 0)
    except (TypeError, ValueError):
        max_len = 0
    if max_len <= 0:
        eval_max_tokens = int(eval_cfg.get("max_tokens") or 0)
        if eval_max_tokens > 0:
            max_len = eval_max_tokens + _SERVE_MAX_MODEL_LEN_PROMPT_SLACK
        elif (request.bench_id or "").lower() in _LONG_OUTPUT_BENCHMARKS:
            # These benchmark defaults request 65K--100K output. Keep their
            # serving context at the shared 128K protocol instead of falling
            # back to a model-size heuristic.
            max_len = _LONG_OUTPUT_SERVE_MAX_MODEL_LEN
        else:
            # eval_all 模式拿不到具体 max_tokens (config max_tokens:0 由
            # benchmark suite 决定) -> 按模型尺寸兜底, 与训练 seq_length 一致:
            # 大模型 (35B/MoE, 原生 256K) 用 262144, 小模型 (4B/8B, 原生 32K)
            # 用 32768。serve 的上下文上限 = 模型原生上限, 不被压成固定 32K。
            model_dir = (request.slime_config or {}).get("model_dir") or ""
            max_len = (
                _DEFAULT_SERVE_MAX_MODEL_LEN_LARGE
                if is_large_model(model_dir)
                else _DEFAULT_SERVE_MAX_MODEL_LEN
            )
    return gpu_util, max_len


def _run_llama_factory_data_dir(workspace: Path) -> Path:
    """本次 run 在 ``workspace/public`` 里的 llama-factory 数据目录.

    ``workspace`` 形如 ``<run>/workspace/experiment_os/experiments/<exp_id>``。
    从 workspace 向上遍历, 找到第一个包含 ``public/backend/llama_factory/llama_factory/data``
    的目录 —— 即本次 run 自带的 llama-factory 副本。agent 必须把自定义数据集注册进
    这份 run 副本的 ``data/`` (dataset_info.json + 数据文件), 绝不允许写 source tree
    (``backend/backend/llama_factory/llama_factory/data``)。找不到副本时只告警、
    仍然用推导路径 —— 让训练以 "dataset not found" 大声失败, 而不是悄悄回退到
    source tree 继续污染。
    """
    for base in (workspace, *workspace.parents):
        candidate = (
            base / "public" / "backend" / "llama_factory" / "llama_factory" / "data"
        )
        if candidate.is_dir():
            return candidate
    fallback = (
        workspace.parents[2] / "public" / "backend" / "llama_factory" / "llama_factory" / "data"
        if len(workspace.parents) >= 3
        else Path("public/backend/llama_factory/llama_factory/data")
    )
    logger.warning(
        "run llama-factory data dir not found under %s (public seed failed?); "
        "using %s — relative dataset names in train YAML will fail loudly.",
        workspace,
        fallback,
    )
    return fallback


def _stage_decontaminated_dataset(request: SubmitRequest, workspace: Path) -> Path:
    """Create an experiment-local dataset registry pointing at the clean JSONL."""
    source_dir = _run_llama_factory_data_dir(workspace)
    # Template-only tests deliberately use nonexistent placeholders. Real submits
    # arrive here with WorkerService's checked, decontaminated artifact.
    if not Path(request.train_data_path).is_file():
        return source_dir
    info_path = source_dir / "dataset_info.json"
    if not info_path.is_file():
        raise ValueError(f"dataset registry not found: {info_path}")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to resolve the training dataset") from exc
    train_cfg = yaml.safe_load(Path(request.training_config_path).read_text(encoding="utf-8"))
    dataset_names = train_cfg.get("dataset") if isinstance(train_cfg, dict) else None
    if isinstance(dataset_names, str):
        dataset_names = [name.strip() for name in dataset_names.split(",") if name.strip()]
    if not isinstance(dataset_names, list) or len(dataset_names) != 1:
        raise ValueError("decontamination requires exactly one named training dataset")
    dataset_name = str(dataset_names[0])
    registry = json.loads(info_path.read_text(encoding="utf-8"))
    if dataset_name not in registry:
        raise ValueError(f"dataset {dataset_name!r} missing from {info_path}")
    entry = dict(registry[dataset_name])
    entry["file_name"] = "decontaminated_train.jsonl"
    staged_dir = workspace / "llama_factory_data"
    staged_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(request.train_data_path, staged_dir / entry["file_name"])
    (staged_dir / "dataset_info.json").write_text(
        json.dumps({dataset_name: entry}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return staged_dir
