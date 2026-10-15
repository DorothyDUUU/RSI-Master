"""Pydantic v2 models used by the ExpOS backend HTTP API.

设计要点:
    - 请求/响应字段与 :class:`~expos.service.worker_service.SubmitBackend`
      协议直接对齐, 方便 client/server 两端复用同一份 schema。
    - ``SubmitResponse`` 至少要携带 ``status`` 和 ``score``, 其余字段用于
      ExperimentResult / ArtifactRecord 采集。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

#: benchmark ids supported by the ExpOS backend pipeline (must mirror the
#: benchmark_suite in configs/rsi_master/config.yaml).
#: 注意: 这是一个 *文档性* 别名, 不再用于字段类型约束 —— ``SubmitRequest.benchmark``
#: 放宽为 ``str``, 因为当 ``bench_id`` (benchmark suite 命名空间) 设置时, run 的
#: benchmark 可以是这 7 个之外的值 (如 ``aime2025`` / ``gpqa``)。
BENCHMARK_IDS = [
    "gsm8k",
    "aime_2025",
    "arena_hard_writing",
    "bfcl",
    "gpqa_main",
    "healthbench_easy",
    "human_eval",
]


class SubmitRequest(BaseModel):
    """POST /submit body."""

    experiment_id: str
    worker_id: str
    benchmark: str = "gsm8k"
    train_data_path: str
    training_config_path: str
    training_mode: Literal["llama_factory_sft", "slime_sft"] = "llama_factory_sft"
    experiment_workspace: str
    # 需要的 GPU 卡数。调度交给 gpu-backend: 它按剩余显存选卡并把
    # CUDA_VISIBLE_DEVICES 注入训练环境, 提交方**不要**设 CUDA_VISIBLE_DEVICES。
    gpu_nums: int = 1
    # 独占整卡到任务 tmux 会话结束 (gpu-backend --need-exclusive-gpu)。训练+serve+eval
    # 是长挂任务: 默认共享下 20 分钟租约到期后卡会对其他非独占任务开放, 可能争显存。
    # pipeline 提交一律独占 (default True) —— 只有显式关掉才退化为共享。
    need_exclusive_gpu: bool = True
    # 训练过程对外暴露的字段, 便于 driver 组装 pipeline.sh 时打注释。
    objective: str = ""
    hypothesis: str = ""
    changed_variable: str = ""
    parent_experiment_id: str | None = None
    # ── eval_all (统一评测) ──────────────────────────────────────────────────
    # ``bench_id`` 非空时, pipeline 的 stage-3 从 per-benchmark 分立评测切换到
    # benchmark suite 统一评测 (backend/backend/eval_all/eval_all.py)。bench_id 是
    # benchmark suite ``format/<id>/`` 目录名, 与 ``benchmark`` 不是同一命名空间
    # (如 aime2025 vs aime_2025)。
    bench_id: str | None = None
    # eval_all 的评测 knob (limit / max_tokens / max_workers / split / bmk_root),
    # 从 ExperimentOS ``state["eval_all"]`` 透传而来。
    eval_all: dict[str, Any] = Field(default_factory=dict)
    # ── slime_sft (Megatron-LM) ──────────────────────────────────────────────
    # ``training_mode="slime_sft"`` 时, pipeline stage-1 委托 launch_sft.sh 训练,
    # 这些键映射到 launch_sft.sh 的 env (model_dir / torch_dist_dir 必填, render
    # 期校验; 其余见 render._resolve_slime_config 的默认值)。
    slime_config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("training_mode", mode="before")
    @classmethod
    def _reject_lora_mode(cls, v: Any) -> Any:
        """LoRA 已下线: 对 ``llama_factory_lora_sft`` 给友好中文报错。

        mode="before" 先于 Literal 校验跑, 让 API/agent 拿到可读的「已下线」提示,
        而不是泛化 ValidationError。其它值原样放行, 交由 Literal 收窄。
        """
        if v == "llama_factory_lora_sft":
            raise ValueError(
                "`llama_factory_lora_sft` 已下线 (2026-08-20, LoRA 停用): "
                "训练只能走全参 SFT —— `training_mode=llama_factory_sft` + "
                "训练 YAML `finetuning_type: full`"
            )
        return v


class SubmitResponse(BaseModel):
    """POST /submit response — mirrors SubmitBackend.submit(...) return dict.

    Pipeline driver 是异步的: 提交到 gpu_backend 后立即返回 ``status="running"``
    并附上 ``gpu_script_id`` / ``pipeline_script_path`` / ``pipeline_log_path``,
    让 agent 用 gpu-backend CLI 自行轮询和查看日志。真正的 ``score`` 由 pipeline
    bash 在结束时写到 ``experiment_workspace/result.json``, WorkerService 之后
    会用一个独立的 refresh 步骤把它回收。
    """

    status: Literal["completed", "failed", "pending", "running"] = "pending"
    score: float | None = None
    benchmark: str = ""
    training_mode: str = ""
    trial_id: str = ""
    trial_path: str = ""
    train_data_path: str = ""
    # gpu_backend 侧的产出 —— 三段 pipeline 都跑在同一个 SCRIPT_ID 下面。
    gpu_script_id: str = ""
    gpu_session_id: str = ""
    gpu_log_path: str = ""
    pipeline_script_path: str = ""
    checkpoint_dir: str = ""
    tensorboard_dir: str = ""
    eval_dir: str = ""
    result_json_path: str = ""
    summary: str = ""
    failure_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        # WorkerService 只关心 dict, 保持宽松以便 GPU backend 后续追加字段。
        return self.model_dump()


class TrialInfo(BaseModel):
    """GET /trials/{trial_id} response — for polling long-running submits."""

    trial_id: str
    experiment_id: str
    worker_id: str
    status: Literal["queued", "pending", "running", "completed", "failed"] = "queued"
    submitted_at: float
    finished_at: float | None = None
    request: SubmitRequest
    response: SubmitResponse | None = None
    failure_reason: str = ""


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded"] = "healthy"
    driver: str = "stub"
    active_trials: int = 0
