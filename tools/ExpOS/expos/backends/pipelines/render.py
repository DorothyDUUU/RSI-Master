"""Render ``pipeline.sh`` for a submit request, dispatching on training_mode.

- ``llama_factory_sft`` -> 无条件拦截 (守卫 (c), 2026-08-27): 本集群只跑
  slime_sft, llama-factory 训练请求整体拒绝 (Co-Placement OOM 修复)。
- ``slime_sft`` -> 数据预检 (格式校验) + 训练委托 ``launch_sft.sh`` (env 覆盖)
  + stage-1.5 torch_dist->HF 转换 + vLLM serve + eval (与 llama-factory 共用
  stage-2/3)。``slime_config`` 的 ``model_dir`` / ``torch_dist_dir`` 必填, 缺了
  render 期直接 ValueError; ``skip_eval: true`` 可退回只训练不评测。
"""

from __future__ import annotations

import logging
import re
import stat
from pathlib import Path
from typing import Any

from ..models import SubmitRequest
from .recommendations import (
    is_large_model,
    model_param_count,
    parallel_defaults,
)
from .constants import (
    _BENCHMARK_EVAL,
    _DEFAULT_EVAL_ALL_LIMIT,
    _DEFAULT_LIMIT,
    _LLAMA_FACTORY_CWD,
    _LLAMA_FACTORY_ENV,
    _LLAMA_FACTORY_PREFLIGHT,
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
from .stages.serve_eval import (
    _benchmark_eval_cfg,
    _judge_exports,
    _pick_vllm_port,
    _serve_args,
    _stage_decontaminated_dataset,
)

logger = logging.getLogger(__name__)


def _reject_llama_factory(request: SubmitRequest) -> None:
    """守卫 (c): llama-factory 训练请求整体拦截 (2026-08-27 决策).

    Co-Placement OOM 修复的第一点: 本集群只跑 slime_sft, llama-factory 全下线。
      - 35B/MoE 走 LF 必 OOM (R3 实证: fp32 upcast ~417GB 物理爆卡);
      - 8B 也必须走 slime 单卡 (已真机验证 1 张 GPU 塞的下), LF 不再用于任何模型。
    统一卡数语义 (35B→2, 8B→1) 让 gpu-backend 调度可预测共置, 不会因混跑
    LF/slime 的不同显存占用而 OOM。无条件拒绝, 不给 worker 任何 '试一把' 的机会。
    """
    raise ValueError(
        "llama-factory 训练已整体下线 (Co-Placement OOM 修复, 2026-08-27): 本集群"
        "只跑 slime_sft —— 35B MoE 用 gpu_nums=2 (TP=2×EP=1×CP=1), 8B 用 "
        "gpu_nums=1 (TP=1×EP=1×CP=1)。请改用 training_mode=slime_sft"
    )


def _reject_bad_slime_gpu_nums(request: SubmitRequest) -> None:
    """守卫 (d): slime 卡数严格语义 (2026-08-28 决策, 2 卡 35B 定性不可能).

    本集群 slime 允许的确定性配置, 让 gpu-backend 调度可预测:
      - 大模型 (>=13B / MoE, 35a3b): ``gpu_nums == 4`` (TP=2×EP=2×CP=1) 或
        ``gpu_nums == 8`` (TP=2×EP=4×CP=1, 单节点 32K 真机已验证);
        4 卡配置已验证
        4×H20 塞的下, loss 健康下降, 0.94/0.96)。2 卡 (TP=1×EP=2×CP=1) 也放行:
        0828 已定性 2 卡 pod 的 440GiB RAM 数学上放不下 35B fp32-master 全精度,
        GPU 侧 f=0.85 已修但 CPU 侧越线 —— 真训练必须 4 卡, 2 卡仅保留给
        L2 内存/共置验证;
      - 小模型 (dense 8B): 必须 ``gpu_nums == 1`` (TP=1×EP=1×CP=1, 已验证 1 卡)。
    其它卡数 (2 卡小模型 / 1 卡大模型 / 大于 8 卡) 一律在入队前拒绝。附带校验并行度一致性: 显式 (或按
    模型默认) 的 TP×EP×CP 必须 == gpu_nums, 否则 launch_sft.sh 运行时断言失败。
    """
    gpu_nums = int(request.gpu_nums or 1)
    slime_cfg = request.slime_config or {}
    model_path = str(slime_cfg.get("model_dir") or "")
    b = model_param_count(model_path)
    large = is_large_model(model_path)
    if large:
        if gpu_nums not in (2, 4, 8):
            raise ValueError(
                f"slime 大模型 ({model_path}, ~{b}B) 只能用 gpu_nums=8 "
                f"(TP=2×EP=4×CP=1), 4 (TP=2×EP=2×CP=1) 或 "
                f"2 (TP=1×EP=2×CP=1, 内存验证): 当前 {gpu_nums} 卡被拒。"
                "8B 用 gpu_nums=1"
            )
    elif gpu_nums == 2 and slime_cfg.get("allow_small_model_2gpu"):
        # 专项 opt-in (2026-09-02, 新机器 L20Z 完整 submit 验证): 默认策略小模型
        # 只能 1 卡; 仅当提交方显式声明 allow_small_model_2gpu=true 才放行 2 卡
        # TP2。背景: 4B dense 在 80GB 单卡贴边 (DP=1 全量 fp32 optimizer ~64GB/rank,
        # 长样本 step-2 OOM), 2×80GB TP2 (~32GB/rank) 是 L20Z 上的稳态解 (实证)。
        # 共享集群平台不传该键 → 行为不变。
        pass
    elif gpu_nums != 1:
        raise ValueError(
            f"slime 小模型 ({model_path}) 只能用 gpu_nums=1 (TP=1×EP=1×CP=1): "
            f"当前 {gpu_nums} 卡被拒 —— 多卡对小模型是浪费且挤占配额槽位。"
            "35B MoE 用 gpu_nums=4 或 8"
        )
    # 并行度一致性: TP×EP×CP 必须 == gpu_nums (缺省按模型尺寸 + 卡数的确定性默认)。
    tp_d, ep_d, cp_d = parallel_defaults(model_path, gpu_nums)
    tp = int(slime_cfg.get("tp_size") or tp_d)
    ep = int(slime_cfg.get("ep_size") or ep_d)
    cp = int(slime_cfg.get("cp_size") or cp_d)
    if tp * ep * cp != gpu_nums:
        raise ValueError(
            f"slime 并行度 TP×EP×CP = {tp}×{ep}×{cp} = {tp * ep * cp} != "
            f"gpu_nums {gpu_nums}: launch_sft.sh 会断言失败。大模型 8 卡用 "
            "TP=2×EP=4×CP=1, 4 卡用 TP=2×EP=2×CP=1 "
            "(2 卡用 TP=1×EP=2×CP=1), 8B 1 卡用 TP=1×EP=1×CP=1"
        )


def _torchrun_launch_block(request: SubmitRequest) -> str:
    """torchrun 触发路径注入启动块 (FORCE_TORCHRUN + MASTER_PORT 重置).

    **FORCE_TORCHRUN (deepspeed)**: LLaMA-Factory 启动契约
    (``hparams/parser.py:353``): 配置声明 deepspeed 且 ``parallel_mode !=
    DISTRIBUTED`` 时直接 ``ValueError: Please use FORCE_TORCHRUN=1 to launch
    DeepSpeed training.`` 而 launcher 只在 FORCE_TORCHRUN=1 **或** 可见 GPU>1 时才
    自动包 torchrun —— 我们 stage-1 直接 ``llamafactory-cli train``, 所以**单卡 +
    deepspeed 配置**会被这个检查拦死 (2026-08-25 exp_0008/0009 双坑之二)。注入
    ``export FORCE_TORCHRUN=1`` + ``export NPROC_PER_NODE=<gpu_nums>`` 让 launcher
    自包 torchrun, ``parallel_mode`` 变 DISTRIBUTED, 检查通过。多卡 (gpu_nums>=2)
    显式 nproc=gpu_nums 与 CUDA_VISIBLE_DEVICES 分配一致; 单卡放行但打 WARN ——
    不并行, 只有 ZeRO-offload 省显存的意义。

    **MASTER_PORT 重置 (R1, 2026-08-26)**: pod env 全局钉死 MASTER_PORT=23456,
    launcher 读 ``os.getenv("MASTER_PORT", find_available_port())``, 每个
    torchrun/deepspeed job 都绑同一端口 —— 并发 job EADDRINUSE 秒挂 (0826 批次 30
    例) 或 NCCL cross-talk 0-step 死锁 (7 例)。torchrun 触发路径 (deepspeed 任意
    gpu_nums / 无 deepspeed 但 gpu_nums>1 隐式 auto-wrap) 一律先 ``unset
    MASTER_PORT`` 再 export 每实验随机端口 (20000-39999)。单卡 + 无 deepspeed
    (不触发 torchrun) -> 空串, 现状零改动。
    """
    try:
        import yaml

        cfg = yaml.safe_load(Path(request.training_config_path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — 读不到/不可解析 → 跳过 (stage-0 预检覆盖)
        cfg = None
    has_deepspeed = isinstance(cfg, dict) and bool(cfg.get("deepspeed"))
    gpu_nums = int(request.gpu_nums or 1)

    # R1: 每实验随机 MASTER_PORT, 覆盖所有 torchrun 触发路径.
    master_port_reset = (
        "# R1 (2026-08-26): pod env 全局注入 MASTER_PORT=23456, 并发 torchrun/deepspeed\n"
        "# job 全绑同端口 -> EADDRINUSE / NCCL cross-talk. 每实验随机重置 (20000-39999).\n"
        "unset MASTER_PORT\n"
        "export MASTER_PORT=$((20000+RANDOM%20000))\n"
    )

    if not has_deepspeed:
        if gpu_nums <= 1:
            # 单卡 + 无 deepspeed: 不触发 torchrun, 无 MASTER_PORT 冲突, 零改动.
            return ""
        return (
            "# 无 deepspeed + gpu_nums>1: launcher 对可见 GPU>1 隐式 auto-wrap torchrun,\n"
            "# 同样读 MASTER_PORT —— 重置避免撞 pod 注入的 23456。\n"
            + master_port_reset
        )
    if gpu_nums <= 1:
        return (
            "# deepspeed 配置 + 单卡: FORCE_TORCHRUN=1 让 launcher 自包 torchrun, 否则\n"
            "# parser.py:353 直接 ValueError。单卡不并行 —— 只有 ZeRO-offload 省显存\n"
            "# 的意义; 想真多卡训练请提交 gpu_nums>=2。\n"
            'log "WARN stage-1: training YAML has deepspeed: but gpu_nums=1 -> torchrun '
            'nproc=1 (ZeRO-offload only, no parallelism)"\n'
            "export FORCE_TORCHRUN=1\n"
            "export NPROC_PER_NODE=1\n"
            + master_port_reset
        )
    return (
        "# deepspeed 多卡: FORCE_TORCHRUN=1 显式走 torchrun, 不依赖隐式 auto-wrap;\n"
        "# NPROC_PER_NODE=gpu_nums 与 gpu_backend 注入的 CUDA_VISIBLE_DEVICES 一致。\n"
        f'log "stage-1: deepspeed multi-GPU training, torchrun nproc={gpu_nums}"\n'
        "export FORCE_TORCHRUN=1\n"
        f"export NPROC_PER_NODE={gpu_nums}\n"
        + master_port_reset
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


def estimate_max_mem_quota(request: SubmitRequest) -> int | None:
    """估算 job 每卡最低可用显存 (MB), 供 gpu_backend 调度器放置 (quota).

    R2 (2026-08-26): 放置谓词原来只认 ``vram_usage_percent < 95%``, 不认绝对剩余
    显存 —— 调度器把 35B job 放到只剩 17MB free 的卡上 OOM (0826 批次 27 例 OOM
    里大部分)。这里给每个 job 一个保守的每卡 free-MB 下限, 调度器
    ``free_memory_mb >= quota`` 不满足就不放, 天然挡住「打包到高占用卡」。

    估算口径 = 训练峰值/卡。卡的总显存由调度器掌握，不能在客户端按 96GB 常量
    反推 ``serve_gpu_memory_utilization``；否则迁移到 80GB 卡时会生成永远无法满足的
    quota。默认独占调度负责保证 serve 获得整卡，vLLM 再按实际卡容量应用 utilization。
      - 训练 (llama_factory 全参 SFT): 单卡 (gpu_nums=1) 权重 bf16 + fp32 优化器 +
        梯度 + 激活, 8B 实测 ~60GB、35B ~72GB (R3 物理边界) -> 取 72GB; 多卡
        deepspeed 按卡分摊。
      - slime (Megatron 并行): 权重/梯度/优化器按 DP/PP/TP 分摊到 gpu_nums 卡,
        每卡 ~20-45GB。

    ``eval_all.max_mem_quota`` 可显式覆盖 (已知模型显存需求的场景)。
    """
    ea = request.eval_all or {}
    gpu_nums = max(1, int(request.gpu_nums or 1))
    if request.training_mode == "slime_sft":
        train_tier = 45 * 1024  # Megatron 并行按卡分摊
    elif gpu_nums <= 1:
        train_tier = 72 * 1024  # 全参 SFT 单卡 (8B~60GB / 35B~72GB)
    else:
        train_tier = 45 * 1024  # deepspeed 多卡按卡分摊
    quota = train_tier

    override = ea.get("max_mem_quota")
    if override is not None:
        try:
            return max(int(override), 0)
        except (TypeError, ValueError):
            logger.warning("eval_all.max_mem_quota=%r ignored (not an int)", override)
    return quota


_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def run_id_from_workspace(workspace: str | Path) -> str:
    """从 experiment workspace 路径推导 run 目录名 (eval_all trial-name 前缀用).

    标准布局: ``<run>/workspace/experiment_os/experiments/<experiment_id>``。
    向上找名为 ``experiments`` 的祖先目录 (其父目录必须是 ``experiment_os``),
    再往上三级 (experiment_os -> workspace -> run) 就是 run 目录。非标准布局
    (单测临时目录 / 直接调 render 的场景) 返回 ``""``, 调用方回退到纯
    ``experiment_id`` —— 保持老行为, 只给真实 run 加前缀。
    """
    ws = Path(workspace)
    for anc in ws.parents:
        if anc.name == "experiments" and anc.parent.name == "experiment_os":
            run_dir = anc.parent.parent.parent
            if run_dir.is_absolute() and run_dir.name:
                return run_dir.name
    return ""


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
        # trial_name = <run_id>-<experiment_id>: 用 run 目录名作前缀, 避免不同
        # run 的同名 exp_XXXX 在 benchmark suite 的 data/output/<id>/<trial> 撞目录
        # (2026-08-30 跨 run eval 撞名 bug)。run_id 非法/推导不出时回退纯 exp_id。
        trial_name = f"{run_id_from_workspace(request.experiment_workspace)}-{request.experiment_id}"
        if _SAFE_NAME.fullmatch(trial_name):
            eval_args += f' --trial-name "{trial_name}"'
        else:
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
        effective_judge_env = bool(
            ea.get("judge_env", _BENCHMARK_EVAL.get(bench_id, {}).get("judge_env", False))
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


def _tool_call_parser(model_dir: str) -> str:
    """vLLM 0.27 ``--tool-call-parser`` 按模型家族选值 (2026-08-28).

    用户要求: Qwen3 系 (qwen3-8b dense) -> ``hermes``, Qwen3.5 系 (35A3B MoE)
    -> ``qwen3_coder``。依 model_dir 目录名判定: 含 qwen3.5 / qwen3.6 /
    35b / a3b 视为新系 -> qwen3_coder, 其余 -> hermes。
    """
    base = Path(model_dir).name.lower() if model_dir else ""
    if any(k in base for k in ("qwen3.5", "qwen3.6", "qwen3_5", "qwen3_6", "35b", "a3b")):
        return "qwen3_coder"
    return "hermes"


def _vllm_parser_args(
    model_dir: str,
    override: str | None = None,
    *,
    include_reasoning_parser: bool = True,
) -> str:
    """stage-2 vLLM serve 的解析器 CLI 参数 (2026-08-28, 用户要求的三个 flag).

    ``--tool-call-parser <p> --enable-auto-tool-choice --reasoning-parser qwen3``。
    ``--enable-auto-tool-choice`` 依赖 ``--tool-call-parser`` (vLLM cli_args 校验,
    只传前者会报错)。``override`` (slime_config.vllm_tool_call_parser) 优先,
    否则按模型家族推断 (_tool_call_parser)。空串时无参数 (老管线不传, 逐字节一致)。
    """
    parser = (override or "").strip() or _tool_call_parser(model_dir)
    args = f" --tool-call-parser {parser} --enable-auto-tool-choice"
    if include_reasoning_parser:
        # 2026-09-04 (main-agent): the hardcoded qwen3 reasoning parser routes a
        # NON-reasoning base model's entire completion into reasoning_content,
        # leaving content EMPTY -> every slime_sft lexam eval scored 0.0 (parser
        # artifact, verified rev_0001). It only helps for actual thinking models
        # (emit <think> tags). Gate on an explicit opt-in so base models (this run
        # = Qwen3-4B-Base) are served plain and the grader reads real content.
        import os as _os
        if _os.environ.get("EXPOS_VLLM_REASONING_PARSER") == "1":
            args += " --reasoning-parser qwen3"
    return args


def _render_llama_factory(request: SubmitRequest, workspace: Path) -> str:
    """Render the LLaMA-Factory path after run-contract validation."""
    # Backward-compatible safety default: direct/backend callers that do not
    # carry an explicit run contract remain blocked. A configured run must
    # explicitly include llama_factory_sft in allowed_training_modes.
    if not request.allowed_training_modes:
        _reject_llama_factory(request)
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

    staged_dataset_dir = _stage_decontaminated_dataset(request, workspace)
    ea = request.eval_all or {}
    # serve env (经 eval_all 透传): 默认 _VLLM_ENV = vllm_env_0819 (transformers
    # 5.15.1), 与训练 env (llama_factory_env 5.8.0) 同属 5.x, tokenizer 格式兼容,
    # 训练产物原样 serve, 无 tokenizer 恢复步骤. ``vllm_env`` 可指向其它 env.
    # stage-0b 模板-eos 预检默认开启; 0819 后已是**咨询性** (qwen3/qwen3_nothink
    # 分叉不再拒绝, 只打印说明), 无需为 replace_eos 模板跳过.
    # ``skip_template_preflight: true`` 仍保留为兜底逃生口 (预检脚本缺失/临时问题).
    vllm_env = str(ea.get("vllm_env") or "").strip() or _VLLM_ENV
    skip_preflight = bool(ea.get("skip_template_preflight"))
    # torchrun 触发路径注入启动块: deepspeed (单卡 WARN / 多卡 nproc=gpu_nums) 或
    # 无 deepspeed 但 gpu_nums>1 (隐式 auto-wrap), 并重置 pod 注入的 MASTER_PORT
    # (R1, 2026-08-26); 单卡无 deepspeed 为空串, 脚本与旧版逐字节一致。
    lf_torchrun_force = _torchrun_launch_block(request)
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
        LF_TORCHRUN_FORCE=lf_torchrun_force,
        VLLM_GPU_MEM_UTIL=str(serve_gpu_util),
        VLLM_MAX_MODEL_LEN=str(serve_max_model_len),
        # 单卡 dense 路径无 TP 拆分; 只有 slime 大模型 (35a3b) 才传 TP 参数.
        VLLM_TP_SIZE_ARGS="",
        # 解析器 flag 共享同一 FRAGMENT_STAGE2, 必须提供; LF 已整体下线 (守卫 (c)
        # 先 raise), 这里仅占位保证 substitute 不 KeyError。hermes 是 qwen3 默认。
        VLLM_PARSER_ARGS=_vllm_parser_args(
            "", include_reasoning_parser=request.bench_id != "arenahardwriting"
        ),
        # vLLM 就绪轮询上限 (秒); LF dense 冷启动远低于 600s, 保持默认。
        VLLM_READY_TIMEOUT="600",
        LLAMA_FACTORY_ENV=_LLAMA_FACTORY_ENV,
        LLAMA_FACTORY_CWD=_LLAMA_FACTORY_CWD,
        DATASET_DIR=str(staged_dataset_dir),
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
    # 并行度默认 = 模型尺寸 + 卡数的确定性配置 (大 4 卡 TP2×EP2×CP1 / 大 2 卡
    # TP1×EP2×CP1 / 小 1 卡 TP1×EP1×CP1), 与守卫 (d) 同一事实源。worker 不传时
    # 渲染出合法配置, 不会撞 launch 断言。
    _gpu_nums = int(request.gpu_nums or slime_cfg.get("num_gpus") or 1)
    tp_d, ep_d, cp_d = parallel_defaults(model_dir, _gpu_nums)
    # rollout/global batch、seq 等训练超参 (可配)。默认沿用 launch_sft.sh 内置值,
    # 但注意: 默认 rollout_batch_size=128 要求数据 >128 行, 小数据集 (infra-test
    # slime_rows=64) 必须显式传, 否则 num_rollout_per_epoch=0 -> AssertionError。
    # serve/eval 相关 (可选, 有默认): hf_dir 是 stage-1.5 转换落点;
    # vocab_size 去 Megatron embedding padding (0 = 不传); vllm_tp_size 大模型
    # MoE 多卡 serve 的 TP 拆分 (1 = 单卡); skip_eval 退回只训练不评测。
    # 后备默认按尺寸给 context length: 小模型 (4B/8B) 32K, 大模型 (35B) 256K。
    # (Qwen3.5-35B-A3B 原生 max_position_embeddings=262144; 训练声明 256K 是
    # 截断窗口, 短数据实际批 token 很小, 4 卡 pod 验证过能跑起)。
    _seq_default = 262144 if is_large_model(model_dir) else 32768
    return {
        "model_dir": model_dir,
        "torch_dist_dir": torch_dist_dir,
        "model_spec": model_spec,
        "output_dir": str(slime_cfg.get("output_dir") or workspace),
        "ckpt_dir": str(slime_cfg.get("ckpt_dir") or ckpt_dir),
        "data_file": str(slime_cfg.get("data_file") or request.train_data_path),
        "num_epoch": str(slime_cfg.get("num_epoch") or 3),
        "num_gpus": str(int(request.gpu_nums or slime_cfg.get("num_gpus") or 1)),
        "tp_size": str(slime_cfg.get("tp_size") or tp_d),
        "ep_size": str(slime_cfg.get("ep_size") or ep_d),
        "has_nvlink": str(slime_cfg.get("has_nvlink") or 0),
        "rollout_batch_size": str(slime_cfg.get("rollout_batch_size") or 128),
        "global_batch_size": str(slime_cfg.get("global_batch_size") or 128),
        "seq_length": str(slime_cfg.get("seq_length") or _seq_default),
        "max_tokens_per_gpu": str(slime_cfg.get("max_tokens_per_gpu") or _seq_default),
        "cp_size": str(slime_cfg.get("cp_size") or 1),
        "hf_dir": str(slime_cfg.get("hf_dir") or workspace / "hf_model"),
        "vocab_size": str(slime_cfg.get("vocab_size") or 0),
        "add_missing": str(slime_cfg.get("add_missing", True)),
        "skip_eval": str(slime_cfg.get("skip_eval", False)),
        "vllm_tp_size": str(slime_cfg.get("vllm_tp_size") or 1),
        # vLLM 就绪轮询上限 (秒): 35B MoE + 262144 max-model-len 冷启动实测 >600s
        # (权重加载 ~101s + torch.compile ~53s + CUDA graph 捕获, 600s 死在 graph
        # 捕获中; 2026-08-28 35a3b-256k 实证)。默认 2400 避免误杀 (2026-08-28)。
        "vllm_ready_timeout": str(slime_cfg.get("vllm_ready_timeout") or 2400),
        # stage-2 的 tool-call-parser 覆盖 (可选): 缺省按模型家族推断
        # (qwen3-8b->hermes, qwen3.5/35a3b->qwen3_coder)。显式指定优先级最高。
        "vllm_tool_call_parser": str(slime_cfg.get("vllm_tool_call_parser") or ""),
    }


def _render_slime(request: SubmitRequest, workspace: Path, ckpt_dir: Path) -> str:
    """slime 路径: 预检 + 训练 + torch_dist->HF 转换 + (默认) serve/eval.

    与 llama-factory 共用 stage-2/3 与评测派发 (_build_eval_config), 单次 substitute。
    ``slime_config.skip_eval=true`` 时退回只训练+转换 (eval_status=skipped)。
    """
    _reject_bad_slime_gpu_nums(request)
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
    # stage-2 vLLM 解析器 flag (2026-08-28): 按模型家族推断 tool-call-parser
    # (qwen3-8b->hermes, qwen3.5/35a3b->qwen3_coder), 可被 slime_config 覆盖。
    # The pinned ArenaHardWriting evaluator reads only message.content. Keep
    # Qwen thoughts and answer together there; otherwise vLLM can return
    # content=None and the official evaluator crashes after generation.
    parser_args = _vllm_parser_args(
        cfg["model_dir"],
        cfg["vllm_tool_call_parser"],
        include_reasoning_parser=request.bench_id != "arenahardwriting",
    )
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
        VLLM_PARSER_ARGS=parser_args,
        VLLM_READY_TIMEOUT=cfg["vllm_ready_timeout"],
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

    if request.allowed_training_modes and request.training_mode not in request.allowed_training_modes:
        raise ValueError(
            f"training_mode={request.training_mode!r} is not allowed by this run; "
            f"allowed_training_modes={request.allowed_training_modes}"
        )

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
