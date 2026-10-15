"""Declarative, extensible training recommendations per model-size tier.

按模型尺寸档位 (4b/8b/35b) 的默认训练配置 (含 ``model_spec``) + 可扩展预检规则,
让 slime_sft 提交自动拿到与模型尺寸匹配的生效默认, 并跑对应的预检。

设计:
  - **内置默认 = 唯一来源**: 三档推荐配置以模块常量 ``DEFAULT_RECOMMENDATIONS_CFG``
    内置 (2026-09-03 起; 原 config.yaml 的 ``expos_training_recommendations`` 块已移除,
    只留注释)。若 config 仍显式提供该块, 则**整体覆盖**内置默认 (escape hatch, 不复用
    逐份重复维护)。
  - **档位派生 (seed 期, worker 不感知)**: run 的 ``base_model`` 固定 → run 启动时
    ``resolve_tier_for_model`` 从模型路径词边界匹配档位 → ``resolve_run_defaults`` 把该
    档位 ``default_config.slime`` 权威地 seed 成 ``state["slime_config"]`` (带
    ``model_tier``)。worker 裸提交即命中档位默认 + 预检, 无需也不应声明 model_tier。
    提交层若仍显式给了 ``model_tier`` (历史遗留) 则沿用旧语义参与匹配。
  - **声明式 + 注册式规则**: 规则本体实现并注册进 ``_RULE_REGISTRY`` —— 加规则 =
    实现规则工厂并注册; 在内置/覆盖 config 的 ``profile.preflight_rules`` 引用规则名。
    引用未知规则名时 ``load_recommendations`` 立即 raise, 写错在启动/提交时就暴露。
  - **两种拦截**: ``HARD_REFUSE`` 直接 raise ValueError (入队前拒绝);
    ``WARNING_REFUSE`` 收集 warning 但放行 (走 worker_service 返回里对 agent 可见
    的通道, 与 ``slime_config_warnings`` 同款)。
  - **对 agent 不可见**: recommendations 只进 ``state["training_recommendations"]``,
    不 seed 进 public/ 或 prompts; 但它导入的 ``training_mode`` (worker 提交的显式
    值) 对 agent 可见。

本模块只依赖标准库 (预检规则读 parquet 用 pyarrow, 缺/读失败则跳过规则)。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

#: 大模型阈值: >=13B (或 MoE 标记) 视为大模型。与 render.py 的守卫共用 (单一事实源,
#: render 从本模块 import)。llama-factory 全参 SFT 在大模型上物理不可行 (R3,
#: 2026-08-26): fp32 upcast ~417GB 物理爆卡。
_LARGE_MODEL_MIN_BILLIONS = 13
#: 模型路径里的 MoE 标记 (小写匹配): 出现即视为大模型 (独立于 B 数)。
_MOE_MARKERS = ("a3b", "moe")
_MODEL_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)b")

#: 档位默认 ``model_spec`` (launch_sft.sh 的架构/env 声明文件) 路径前缀, 来自
#: slime 安装 (仓库之外), 由 ``$SLIME_MODEL_SPEC_PREFIX`` 提供。
#: 空 MODEL_SPEC 会让 launch_sft.sh 落到内置 **35B MoE 默认架构** (hidden 2048/layers
#: 40/rope 1e7), 与 Qwen3-4B-Base (2560/36/1e6) 不匹配 -> hf_validate 炸 / 静默训错,
#: 所以三档都必须在默认里显式带 model_spec。
_MODEL_SPEC_PREFIX = os.environ.get("SLIME_MODEL_SPEC_PREFIX", "")


def _model_spec(name: str) -> str:
    """Join the slime model-spec dir with *name*; "" when the prefix is unset."""
    return f"{_MODEL_SPEC_PREFIX}/{name}" if _MODEL_SPEC_PREFIX else ""


_QWEN3_4B_SPEC = _model_spec("qwen3-4B.sh")
_QWEN3_8B_SPEC = _model_spec("qwen3-8B.sh")
_QWEN3_5_35B_A3B_SPEC = _model_spec("qwen3.5-35B-A3B.sh")


def model_param_count(model_path: str) -> float | None:
    """从模型路径/HF id 粗估参数量 (B 数): 'Qwen3-8B-Base' -> 8.0, '35B-A3B' -> 13.0.

    找不到 B 数 -> None (未知, 调用方不拦截)。MoE 标记 (``a3b`` / ``moe``) 无论
    B 数一律视为大模型 (返回阈值) —— 本仓库只有 35a3b 一种 MoE, 且 MoE 不该走 LF。
    """
    if not model_path:
        return None
    low = model_path.lower()
    for marker in _MOE_MARKERS:
        if marker in low:
            return float(_LARGE_MODEL_MIN_BILLIONS)
    m = _MODEL_SIZE_RE.search(low)
    return float(m.group(1)) if m else None


def is_large_model(model_path: str) -> bool:
    """模型是否为"大模型" (>=13B 或 MoE 标记): render 守卫与档位匹配共用的阈值."""
    b = model_param_count(model_path)
    return b is not None and b >= _LARGE_MODEL_MIN_BILLIONS


def parallel_defaults(model_path: str, gpu_nums: int) -> tuple[int, int, int]:
    """模型尺寸 + 卡数对应的确定性并行配置 (唯一合法 (tp, ep, cp)).

    大模型 (>=13B / MoE, 本仓库只有 35a3b):
      8 卡 -> TP=2×EP=4×CP=1 (2×4×1=8, 2026-09-15 单节点 L20Z 32K 实证);
      4 卡 -> TP=2×EP=2×CP=1 (2×2×1=4, 2026-08-23 真机验证 0.94/0.96);
      2 卡 -> TP=1×EP=2×CP=1 (1×2×1=2, EP 拆专家; 2026-08-27 实证 TP=2×EP=1
      时专家不随 TP 减 -> 全模型复制 OOM)。
    小模型 (dense 4B/8B):
      2 卡 (L20Z 2×80GB TP2 opt-in, 2026-09-02 实证的 4B 稳态) ->
        TP=2×EP=1×CP=1 (2×1×1=2);
      1 卡 -> TP=1×EP=1×CP=1 (1×1×1=1)。
    """
    if is_large_model(model_path):
        if gpu_nums >= 8:
            return (2, 4, 1)
        return (2, 2, 1) if gpu_nums >= 4 else (1, 2, 1)
    return (2, 1, 1) if gpu_nums >= 2 else (1, 1, 1)


class PreflightKind(str, Enum):
    """预检规则拦截语义. ``HARD_REFUSE`` 直接拒绝实验 (raise ValueError);
    ``WARNING_REFUSE`` 提示 warning 但允许放行。"""

    HARD_REFUSE = "HARD_REFUSE"
    WARNING_REFUSE = "WARNING_REFUSE"


@dataclass
class TrainingRecommendation:
    """一个模型尺寸档位的训练推荐 (默认训练配置 + 预检规则)."""

    profile: str
    title: str
    #: 直接字段匹配: ``match`` 直接写档位字符串 (如 "8b"/"35b"), 与
    #: ``slime_config.model_tier`` 精确比较。该 model_tier 由 run seed 期
    #: ``resolve_tier_for_model`` 从 base_model 自动派生 (worker 不感知); 提交层若仍
    #: 显式传了 model_tier (历史遗留) 则沿用旧语义参与匹配。
    match_tiers: list[str] = field(default_factory=list)
    supported_training_modes: list[str] = field(default_factory=list)
    default_training_mode: str = ""
    allowed_gpu_nums: list[int] = field(default_factory=list)
    default_config: dict[str, Any] = field(default_factory=dict)
    preflight_rules: list["PreflightRule"] = field(default_factory=list)

    def matches(self, model_tier: str) -> bool:
        """提交的档位字符串是否命中本档位 (直接字段, 精确相等, 大小写不敏感)."""
        if not model_tier or not self.match_tiers:
            return False
        tier = model_tier.strip().lower()
        return any(t.strip().lower() == tier for t in self.match_tiers)


@dataclass
class PreflightContext:
    """一次提交的预检上下文 (apply_preflight 时填充)."""

    #: slime_config.model_tier —— seed 期按 base_model 档位自动派生 (4b/8b/35b),
    #: worker 无需声明; 提交层显式覆盖时沿用旧语义参与匹配。
    model_tier: str
    model_path: str  # slime_config.model_dir (仅用于错误/警告消息里的模型身份)
    training_mode: str
    gpu_nums: int
    slime_config: dict[str, Any]
    train_data_path: str
    recommendations: list[TrainingRecommendation]
    profile: TrainingRecommendation | None = None


@dataclass
class PreflightRule:
    """一条预检规则. ``predicate(ctx)`` 返回 None = 放行, 返回 str = 触发消息.

    ``kind == HARD_REFUSE`` 时触发即 raise ValueError(message);
    ``kind == WARNING_REFUSE`` 时触发消息进 apply_preflight 的返回列表。
    """

    name: str
    kind: PreflightKind
    description: str
    predicate: Callable[[PreflightContext], str | None]


#: 规则注册器: name -> 工厂 (接收 config 的 params dict, 返回 PreflightRule)。
#: 可扩展入口 —— 新增规则 = 实现工厂 + 注册 + config 引用。
_RULE_REGISTRY: dict[str, Callable[[dict[str, Any]], PreflightRule]] = {}


def register_rule(name: str, factory: Callable[[dict[str, Any]], PreflightRule]) -> None:
    """注册一个规则工厂 (可扩展入口). 重复注册同 name 直接 raise, 防 config 分叉."""
    if name in _RULE_REGISTRY:
        raise ValueError(f"preflight rule already registered: {name}")
    _RULE_REGISTRY[name] = factory


def _rule(name: str, kind: PreflightKind, description: str) -> Callable:
    """装饰器: 把 predicate 函数注册成同名规则工厂 (params 只取 kind/description 覆盖)."""

    def deco(fn: Callable[[PreflightContext], str | None]) -> Callable:
        def factory(params: dict[str, Any]) -> PreflightRule:
            return PreflightRule(
                name=name,
                kind=PreflightKind(params.get("kind", kind)),
                description=str(params.get("description", description)),
                predicate=fn,
            )

        register_rule(name, factory)
        return fn

    return deco


def _model_label(ctx: PreflightContext) -> str:
    """错误/警告消息里的模型身份: 档位字符串 + model_dir."""
    tier = ctx.model_tier.strip() if ctx.model_tier else "<no tier>"
    return f"{ctx.model_path or '<unknown>'} [{tier}]"


# ── 内置规则 ────────────────────────────────────────────────────────────────


@_rule(
    "training_mode_supported",
    PreflightKind.HARD_REFUSE,
    "training_mode 必须在该档位支持列表内",
)
def _r_training_mode_supported(ctx: PreflightContext) -> str | None:
    prof = ctx.profile
    if prof is None or ctx.training_mode in prof.supported_training_modes:
        return None
    modes = ", ".join(prof.supported_training_modes)
    return (
        f"training_mode={ctx.training_mode} 不被档位 '{prof.profile}' "
        f"({prof.title}) 支持: 该模型尺寸只支持 [{modes}]。"
        f"{'请改用 ' + prof.default_training_mode if prof.default_training_mode else ''}"
    )


@_rule(
    "gpu_nums_must_match",
    PreflightKind.HARD_REFUSE,
    "gpu_nums 必须在该档位允许列表内",
)
def _r_gpu_nums_must_match(ctx: PreflightContext) -> str | None:
    prof = ctx.profile
    if prof is None or ctx.gpu_nums in prof.allowed_gpu_nums:
        return None
    allowed = "、".join(str(n) for n in prof.allowed_gpu_nums)
    return (
        f"档位 '{prof.profile}' ({prof.title}) 只允许 gpu_nums={allowed}: "
        f"当前 {ctx.gpu_nums} 卡被拒 ({_model_label(ctx)})。"
        "请按档位要求显式传 --gpu-nums"
    )


@_rule(
    "parallelism_matches_gpu_nums",
    PreflightKind.HARD_REFUSE,
    "TP×EP×CP 必须 == gpu_nums",
)
def _r_parallelism_matches_gpu_nums(ctx: PreflightContext) -> str | None:
    if ctx.profile is None:
        return None
    tp_d, ep_d, cp_d = parallel_defaults(ctx.model_path, ctx.gpu_nums)
    tp = int(ctx.slime_config.get("tp_size") or tp_d)
    ep = int(ctx.slime_config.get("ep_size") or ep_d)
    cp = int(ctx.slime_config.get("cp_size") or cp_d)
    if tp * ep * cp == ctx.gpu_nums:
        return None
    return (
        f"slime 并行度 TP×EP×CP = {tp}×{ep}×{cp} = {tp * ep * cp} != gpu_nums "
        f"{ctx.gpu_nums}: launch_sft.sh 会断言失败。大模型 8 卡用 TP=2×EP=4×CP=1, "
        "4 卡用 TP=2×EP=2×CP=1 (2 卡用 TP=1×EP=2×CP=1), "
        "8B 1 卡用 TP=1×EP=1×CP=1"
    )


def _estimate_row_count(path: str) -> int | None:
    """粗估训练数据行数 (slime 数据是 parquet). pyarrow 不可用/读失败 -> None (放行).

    独立函数便于测试 monkeypatch 注入行数。
    """
    if not path:
        return None
    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]

        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:  # noqa: BLE001 — 读不到就当未知, 放行
        return None


@_rule(
    "torch_dist_dir_required",
    PreflightKind.WARNING_REFUSE,
    "缺 slime_config.torch_dist_dir 时提示需先转换",
)
def _r_torch_dist_dir_required(ctx: PreflightContext) -> str | None:
    if ctx.slime_config.get("torch_dist_dir"):
        return None
    return (
        f"training_mode=slime_sft 未指定 slime_config.torch_dist_dir "
        f"({_model_label(ctx)}): 训练 --load 需要先跑 convert_hf_to_torch_dist.sh "
        "转换出的 Megatron torch_dist 产物, 缺了 GPU 上会 ERROR。新模型请先转换再提交。"
    )


@_rule(
    "data_rows_vs_rollout_batch",
    PreflightKind.WARNING_REFUSE,
    "数据行数 < rollout_batch_size 时提示 (否则 launch num_rollout_per_epoch=0 断言失败)",
)
def _r_data_rows_vs_rollout_batch(ctx: PreflightContext) -> str | None:
    data_file = str(ctx.slime_config.get("data_file") or ctx.train_data_path or "")
    rollout = int(ctx.slime_config.get("rollout_batch_size") or 128)
    rows = _estimate_row_count(data_file)
    if rows is None or rows >= rollout:
        return None
    return (
        f"训练数据约 {rows} 行 < rollout_batch_size {rollout} ({_model_label(ctx)}) —— "
        "launch_sft.sh 会算出 num_rollout_per_epoch=0 然后断言失败。"
        "小数据集请显式配置 slime_config.rollout_batch_size (≤ 行数)。"
    )


def _estimate_row_tokens(row: Any) -> int | None:
    """估算一行 slime 数据的内容 token 数 (无 tokenizer 的保守估计).

    slime 数据的 ``messages`` 列每行是 [{role, content}, ...]。逐条 content 累计:
    非 ASCII (CJK/数学符号, 实际 ~1 tok/char) 按 1 tok/char, ASCII (英文/代码,
    实际 ~3.5 chars/tok) 按 len/3.5 —— 无 tokenizer 时足够保守地判断"seq_length 会不会
    截掉长行"。结构异常/未知 -> None (放行, 不拿半对数据吓 worker)。
    """
    if not isinstance(row, list):
        return None
    total = 0
    for msg in row:
        if not isinstance(msg, dict):
            return None
        content = msg.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):  # multimodal parts (只取 text)
            text = "".join(
                str(p.get("text", "")) if isinstance(p, dict) else str(p)
                for p in content
            )
        else:
            text = ""
        ascii_n = sum(1 for ch in text if ch.isascii())
        total += int((len(text) - ascii_n) + ascii_n / 3.5)
    return total


def _estimate_seq_overrun(path: str, seq: int) -> tuple[int, int, int] | None:
    """读训练 parquet 估 (max_est_tokens, n_over_seq, n_rows); 失败 -> None (放行).

    只拉 ``messages`` 列并分 batch 流式聚合, 不整列载入内存。与 ``_estimate_row_count``
    同风格: pyarrow 缺/读失败/无 messages 列都静默放行 (best-effort 预检)。
    """
    if not path or not seq:
        return None
    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return None
    try:
        pf = pq.ParquetFile(path)
        if "messages" not in pf.schema.names:
            return None
        n_rows = int(pf.metadata.num_rows)
        max_est = 0
        n_over = 0
        for batch in pf.iter_batches(columns=["messages"], batch_size=2048):
            for row in batch.column("messages").to_pylist():
                est = _estimate_row_tokens(row)
                if est is None:
                    return None  # 结构未知 -> 整体放行
                if est > max_est:
                    max_est = est
                if est > seq:
                    n_over += 1
        return max_est, n_over, n_rows
    except Exception:  # noqa: BLE001
        return None


@_rule(
    "seq_length_may_truncate",
    PreflightKind.WARNING_REFUSE,
    "seq_length 小到可能截掉训练长行时提示 (截断窗口, 非阻塞; 默认 32768/262144)",
)
def _r_seq_length_may_truncate(ctx: PreflightContext) -> str | None:
    # seq_length 只是截断窗口 (长尾可被截); 风险在 worker 为省显存把 seq 调太小, 静默
    # 截掉长 CoT/长文行尾部。默认与 render/档位同源: 大模型 262144, 小模型 32768。
    data_file = str(ctx.slime_config.get("data_file") or ctx.train_data_path or "")
    try:
        seq = int(ctx.slime_config.get("seq_length"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        seq = 0
    if seq <= 0:
        seq = 262144 if is_large_model(ctx.model_path) else 32768
    over = _estimate_seq_overrun(data_file, seq)
    if over is None or over[1] == 0:
        return None
    max_est, n_over, n_rows = over
    return (
        f"训练数据约 {n_over}/{n_rows} 行内容长度估算 > seq_length={seq} "
        f"({_model_label(ctx)}), 最长行 ~{max_est} tok: seq_length 是截断窗口, 超长行"
        "尾部会被截掉 —— 长 CoT/长上下文数据请调大 seq_length (及 max_tokens_per_gpu), "
        "别靠调小 seq 省显存而静默丢长尾。"
    )


@_rule(
    "large_model_two_gpu_memory",
    PreflightKind.WARNING_REFUSE,
    "35B 档位 + gpu_nums==2 时提示真训练必须 4 卡",
)
def _r_large_model_two_gpu_memory(ctx: PreflightContext) -> str | None:
    if ctx.profile is None or ctx.gpu_nums != 2:
        return None
    return (
        f"gpu_nums=2 跑 35B 全精度训练 ({_model_label(ctx)}): 2 卡 pod 440GiB RAM "
        "数学上放不下 35B fp32-master 全精度 (2026-08-28 定性, memory "
        "2gpu-35b-full-precision-infeasible): 真训练必须 gpu_nums=4 (TP=2×EP=2×CP=1), "
        "2 卡仅保留给内存/共置验证。"
    )


# ── 加载 / 匹配 / 执行 ──────────────────────────────────────────────────────


#: 内置档位推荐默认 (2026-09-03 起唯一来源; config.yaml 不再重复维护, 只留注释)。
#: 语义要点:
#:   - run seed 期按 base_model 自动命中档位 (``resolve_tier_for_model``), worker 不感知、
#:     不声明 model_tier; config 若仍显式提供 ``expos_training_recommendations`` 则整体覆盖
#:     这份内置默认 (escape hatch)。
#:   - 三档都带 ``model_spec`` (launch_sft.sh 无默认 spec 会落到内置 **35B MoE 架构**,
#:     hidden 2048/layers 40/rope 1e7, 与 Qwen3-4B-Base 的 2560/36/1e6 不匹配 -> 静默训错)。
#:   - **4B 默认 2 卡** (L20Z 2×80GB TP2 稳态, 2026-09-02 实证; 单卡 80GB 贴边)。
#:     ``allowed_gpu_nums=[1,2]`` -> 单卡保留为显式 opt-in; default slime 带
#:     ``allow_small_model_2gpu: true`` 才过得去 render 守卫 (小模型 2 卡默认拒)。
#:   - 档位 slime 默认**不再烘焙 tp/ep/cp** (并行以 ``parallel_defaults`` 为唯一事实源:
#:     4b@2卡 -> TP2, 35b@4 -> TP2×EP2; 烘焙会掩盖并行默认, 卡数一变就撞守卫)。
#:     35b 保留 serve 侧 ``vllm_tp_size``/``vllm_ready_timeout`` (render 默认 vllm_tp_size=1)。
#:   - 每档 preflight_rules 含 ``seq_length_may_truncate`` (WARNING: 提醒 seq 别调太小截掉长行)。
DEFAULT_RECOMMENDATIONS_CFG: dict[str, Any] = {
    "profiles": {
        "4b": {
            "title": "Qwen3-4B (小尺寸 dense, 只走 slime_sft; 默认 2 卡 TP2)",
            "match": "4b",
            "supported_training_modes": ["slime_sft"],
            "default_training_mode": "slime_sft",
            "allowed_gpu_nums": [1, 2],
            "default_config": {
                "gpu_nums": 2,
                "slime": {
                    "model_spec": _QWEN3_4B_SPEC,
                    "allow_small_model_2gpu": True,
                    "seq_length": 32768,
                    "max_tokens_per_gpu": 32768,
                    "rollout_batch_size": 64,
                    "global_batch_size": 64,
                },
            },
            "preflight_rules": [
                {"name": "training_mode_supported", "kind": "HARD_REFUSE"},
                {"name": "gpu_nums_must_match", "kind": "HARD_REFUSE"},
                {"name": "parallelism_matches_gpu_nums", "kind": "HARD_REFUSE"},
                {"name": "torch_dist_dir_required", "kind": "WARNING_REFUSE"},
                {"name": "data_rows_vs_rollout_batch", "kind": "WARNING_REFUSE"},
                {"name": "seq_length_may_truncate", "kind": "WARNING_REFUSE"},
            ],
        },
        "8b": {
            "title": "Qwen3-8B (中尺寸 dense, 只走 slime_sft)",
            "match": "8b",
            "supported_training_modes": ["slime_sft"],
            "default_training_mode": "slime_sft",
            "allowed_gpu_nums": [1],
            "default_config": {
                "gpu_nums": 1,
                "slime": {
                    "model_spec": _QWEN3_8B_SPEC,
                    "seq_length": 32768,
                    "max_tokens_per_gpu": 32768,
                    "rollout_batch_size": 64,
                    "global_batch_size": 64,
                },
            },
            "preflight_rules": [
                {"name": "training_mode_supported", "kind": "HARD_REFUSE"},
                {"name": "gpu_nums_must_match", "kind": "HARD_REFUSE"},
                {"name": "parallelism_matches_gpu_nums", "kind": "HARD_REFUSE"},
                {"name": "torch_dist_dir_required", "kind": "WARNING_REFUSE"},
                {"name": "data_rows_vs_rollout_batch", "kind": "WARNING_REFUSE"},
                {"name": "seq_length_may_truncate", "kind": "WARNING_REFUSE"},
            ],
        },
        "35b": {
            "title": "Qwen3.5-35B-A3B (大尺寸 MoE, 只走 slime_sft, 支持 4/8 卡真训练)",
            "match": ["35b", "a3b"],
            "supported_training_modes": ["slime_sft"],
            "default_training_mode": "slime_sft",
            "allowed_gpu_nums": [2, 4, 8],
            "default_config": {
                "gpu_nums": 4,
                "slime": {
                    "model_spec": _QWEN3_5_35B_A3B_SPEC,
                    "seq_length": 262144,
                    "max_tokens_per_gpu": 262144,
                    "rollout_batch_size": 64,
                    "global_batch_size": 64,
                    "vllm_tp_size": 4,  # 2026-08-29: 任务独占 4 卡, serve 用满 4 卡 TP4
                    "vllm_ready_timeout": 2400,
                },
            },
            "preflight_rules": [
                {"name": "training_mode_supported", "kind": "HARD_REFUSE"},
                {"name": "gpu_nums_must_match", "kind": "HARD_REFUSE"},
                {"name": "parallelism_matches_gpu_nums", "kind": "HARD_REFUSE"},
                {"name": "torch_dist_dir_required", "kind": "WARNING_REFUSE"},
                {"name": "data_rows_vs_rollout_batch", "kind": "WARNING_REFUSE"},
                {"name": "seq_length_may_truncate", "kind": "WARNING_REFUSE"},
                {"name": "large_model_two_gpu_memory", "kind": "WARNING_REFUSE"},
            ],
        },
    },
}


def load_recommendations(cfg: dict[str, Any] | None) -> list[TrainingRecommendation]:
    """从 config 的 ``expos_training_recommendations.profiles.*`` 解析档位列表.

    未知规则名 / 缺 kind 等配置错误立即 raise (提交时暴露, 不等 GPU)。空/缺失
    config -> [] (无推荐, 不预检)。
    """
    if not cfg:
        return []
    profiles_cfg = (cfg.get("profiles") or {}) if isinstance(cfg, dict) else {}
    recommendations: list[TrainingRecommendation] = []
    for profile_name, pc in profiles_cfg.items():
        if not isinstance(pc, dict):
            raise ValueError(
                f"expos_training_recommendations.profiles.{profile_name} must be a mapping"
            )
        match = pc.get("match") or {}
        match_tiers = _as_tiers_list(match.get("tiers") if isinstance(match, dict) else match)
        rules: list[PreflightRule] = []
        for rule_cfg in pc.get("preflight_rules") or []:
            if not isinstance(rule_cfg, dict):
                raise ValueError(
                    f"preflight rule entry under {profile_name} must be a mapping: {rule_cfg}"
                )
            name = rule_cfg.get("name")
            factory = _RULE_REGISTRY.get(name)
            if factory is None:
                known = ", ".join(sorted(_RULE_REGISTRY))
                raise ValueError(
                    f"unknown preflight rule name {name!r} under profile {profile_name}; "
                    f"registered rules: {known}"
                )
            rules.append(factory(rule_cfg))
        recommendations.append(
            TrainingRecommendation(
                profile=str(profile_name),
                title=str(pc.get("title", profile_name)),
                match_tiers=match_tiers,
                supported_training_modes=[
                    str(m) for m in (pc.get("supported_training_modes") or [])
                ],
                default_training_mode=str(pc.get("default_training_mode") or ""),
                allowed_gpu_nums=[int(n) for n in (pc.get("allowed_gpu_nums") or [])],
                default_config=dict(pc.get("default_config") or {}),
                preflight_rules=rules,
            )
        )
    return recommendations


def effective_recommendations(
    cfg: dict[str, Any] | None,
) -> list[TrainingRecommendation]:
    """解析**生效**的档位列表: cfg 有值 -> 走 ``load_recommendations(cfg)`` (整体覆盖
    内置默认); 空/缺 -> 内置 ``DEFAULT_RECOMMENDATIONS_CFG`` (不返回 []).

    与 ``load_recommendations`` 的唯一区别是空时的回退对象 —— 让任一 run 只靠
    base_model 就自动启用档位默认 + 预检。worker_service 预检 / run seed 都用这个,
    不用手动 ``or DEFAULT_RECOMMENDATIONS_CFG``。
    """
    return load_recommendations(cfg if cfg else DEFAULT_RECOMMENDATIONS_CFG)


def effective_recommendations_cfg(cfg: dict[str, Any] | None) -> dict[str, Any]:
    """生效的推荐配置 **dict** (seed 进 ``state["training_recommendations"]`` 自描述用).

    只回答"state 里存哪份配置", 不改档位解析 (解析一律走 ``effective_recommendations``)。
    cfg 有值 (run config 显式覆盖) 原样返回; 空/缺 -> 内置 ``DEFAULT_RECOMMENDATIONS_CFG``,
    让 state 自描述出当前生效的三档, worker_service 无需再手动回退。
    """
    return dict(cfg) if cfg else DEFAULT_RECOMMENDATIONS_CFG


def resolve_profile(
    model_tier: str, recommendations: list[TrainingRecommendation]
) -> TrainingRecommendation | None:
    """按声明顺序匹配第一个命中的档位 (直接字段: ``slime_config.model_tier`` == 档位
    ``match`` 声明的字符串); 不命中 -> None (不预检、不用默认)."""
    for rec in recommendations:
        if rec.matches(model_tier):
            return rec
    return None


def resolve_tier_for_model(
    model_path: str, recommendations: list[TrainingRecommendation]
) -> str | None:
    """从模型路径/HF id 顺序匹配第一个命中的档位 (seed 期按 base_model 自动派生).

    对小写路径做**词边界**匹配 (目标串前后都不是字母/数字): 'Qwen3-4B-Base' 命中
    "4b" ('4b' 前是 '-' 边界); 'Qwen3-14B' 里的 "14b" 不会误中 "4b" ('4b' 前是 '1');
    'Qwen3.5-35B-A3B' 命中 "35b" (也吃 "a3b")。返回命中档位的 profile 名 (与
    ``resolve_profile`` 匹配用的 model_tier 同串); 无命中 -> None (无档位尺寸, 沿用
    旧行为, 不预检不用默认)。
    """
    if not model_path or not recommendations:
        return None
    low = model_path.lower()
    for rec in recommendations:
        for tier in rec.match_tiers:
            t = tier.strip().lower()
            if not t:
                continue
            if re.search(r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])", low):
                return rec.profile
    return None


def resolve_run_defaults(
    base_model: str,
    run_slime_block: dict[str, Any] | None,
    recs_cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    """run seed 期把 base_model 命中档位的默认 slime 配置权威地并进 run 级默认.

    结果写 ``state["slime_config"]`` (worker 每次 slime_sft 提交的基准), 末尾带
    ``model_tier`` 使 submit 期 preflight 自动武装。合并优先级: 档位 profile 已定义的
    键全归 profile (seq/batch/model_spec/allow 2gpu 等尺寸键一律按 base_model 档位定,
    不再被 run 级遗留默认污染, 例: config.yaml 旧的 tp2/ep4 被 profile/并行默认顶掉);
    run 块只贡献 profile **没定义**的键 (model_dir/num_epoch/torch_dist_dir 等 run 专属)。
    base_model 无档位命中 -> 原样透传 run 块 (旧行为, 1.7B 等无档位尺寸不受影响)。
    """
    run_slime_block = dict(run_slime_block or {})
    recs = effective_recommendations(recs_cfg)
    tier = resolve_tier_for_model(base_model, recs)
    if tier is None:
        return run_slime_block
    profile = resolve_profile(tier, recs)
    if profile is None:  # 理论不可达: tier 从 recs 派生必命中; 兜底走旧行为
        return run_slime_block
    profile_slime = dict(profile.default_config.get("slime") or {})
    merged = dict(profile_slime)
    merged.update({k: v for k, v in run_slime_block.items() if k not in profile_slime})
    merged["model_tier"] = tier
    return merged


def apply_preflight(
    profile: TrainingRecommendation,
    ctx: PreflightContext,
) -> list[str]:
    """跑档位的全部预检规则, 返回 WARNING 消息列表 (放行).

    HARD_REFUSE 命中 -> ``raise ValueError(message)`` (入队前拒绝, 消息对 agent
    友好); WARNING_REFUSE 命中 -> 消息进返回列表。
    """
    ctx.profile = profile
    warnings: list[str] = []
    for rule in profile.preflight_rules:
        try:
            message = rule.predicate(ctx)
        except Exception as exc:  # noqa: BLE001 — 规则自身异常不该放行危险提交
            message = (
                f"{rule.name} predicate 异常 ({exc}); "
                "规则被跳过, 请检查 config/实现"
            )
        if message is None:
            continue
        if rule.kind == PreflightKind.HARD_REFUSE:
            raise ValueError(message)
        warnings.append(message)
    return warnings


def _as_tiers_list(value: Any) -> list[str]:
    """解析 ``match``: 单个字符串或字符串列表 -> list[str] (档位名); 缺失 -> [].

    支持 ``match: "8b"`` 或 ``match: ["8b", "8B"]``。
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if v]
    return [str(value).strip()] if str(value).strip() else []
