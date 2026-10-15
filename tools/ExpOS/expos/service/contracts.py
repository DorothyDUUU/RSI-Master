"""Structured contracts for the ExperimentOS v2 graph-based workflow.

Pydantic 模型集合，定义 ExperimentOS 中所有落盘对象的结构化协议:
    - DAG 节点 (WorkerNode / ExperimentNode / ReviewerNode)
    - 工作单据 (WorkOrder / ExperimentResult / ReviewReport)
    - 数据池记录 (DataPoolRecord / DataMemoryRecommendation)
    - 事件流 (OSEvent)

Every artifact written to disk MUST be validated through one of these models so
that CLI commands and agent tools observe a stable schema.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field


# ─── 数据流基础模型 ────────────────────────────────────────────────────────────


class ArtifactRecord(BaseModel):
    """Registered artifact metadata.

    工件内容仍留在磁盘 workspace, ExperimentOS 仅记录路径、摘要以便审计。
    """

    type: str = Field(description="Artifact kind, e.g. train_data, config, eval, log, checkpoint")
    path: str = Field(description="Absolute filesystem path to the artifact")
    sha256: str = Field(default="", description="Digest computed during registration")
    summary: str = Field(default="", description="Short human-readable description")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional structured artifact metadata")


class DataMemoryRecommendation(BaseModel):
    """Reviewer recommendation about a train_data artifact.

    仅作为建议, Main Agent 可自主决定是否 promote / reuse。
    """

    action: Literal["promote", "keep_candidate", "discard", "revise", "inspect_more", "none"] = "none"
    target_data_id: str | None = None
    target_artifact_path: str | None = None
    source_experiment_id: str | None = None
    reason: str = ""
    reuse_instruction: str | None = None
    what_to_preserve: list[str] = Field(default_factory=list)
    what_to_change: list[str] = Field(default_factory=list)
    leakage_risk: str | None = None
    confidence: float | None = None


class DataReusePlan(BaseModel):
    """Worker's explicit decision about whether to reuse prior train_data."""

    checked_reusable_data: bool = False
    reuse_mode: Literal["none", "reuse", "extend", "filter", "ablate", "revise"] = "none"
    selected_data_ids: list[str] = Field(default_factory=list)
    selected_paths: list[str] = Field(default_factory=list)
    rejected_data_ids: list[str] = Field(default_factory=list)
    source_review_ids: list[str] = Field(default_factory=list)
    reason: str = ""
    what_to_preserve: list[str] = Field(default_factory=list)
    what_to_change: list[str] = Field(default_factory=list)


# ─── 实验/工单/审计 ────────────────────────────────────────────────────────────


class WorkOrder(BaseModel):
    """A single experiment prescription authored by a Worker via submit."""

    schema_version: str = "rsi_master.work_order.v1"
    experiment_id: str
    worker_id: str
    parent_experiment_id: str | None = None
    control_experiment_id: str | None = None
    comparison_group: str = ""
    benchmark: str = "aime_2025"
    objective: str = ""
    hypothesis: str = ""
    changed_variable: str = ""
    fixed_variables: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    deliverables: list[str] = Field(default_factory=list)
    data_reuse_plan: DataReusePlan = Field(default_factory=DataReusePlan)
    training_config_path: str = ""
    # plain str (不作 Literal 约束): 历史 run 的 work_order.json 可能已落盘
    # ``llama_factory_lora_sft``, 收窄 Literal 会让反序列化失败; 活体请求面
    # (SubmitRequest / WorkerSubmitParams) 已用收窄 Literal + 友好 validator 拦截。
    training_mode: str = "llama_factory_sft"
    train_data_path: str = ""
    created_at: float = Field(default_factory=time.time)


class ExperimentResult(BaseModel):
    """Outcome payload produced by the training submit pipeline."""

    schema_version: str = "rsi_master.result.v1"
    experiment_id: str
    worker_id: str
    status: Literal["completed", "failed", "pending", "running", "aborted"] = "pending"
    eval_status: Literal["completed", "failed", "pending", "skipped"] = "pending"
    benchmark: str = "aime_2025"
    training_mode: str = ""
    score: float | None = None
    data_summary: str = ""
    effective_config: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[ArtifactRecord] = Field(default_factory=list)
    submit_result: dict[str, Any] = Field(default_factory=dict)
    failure_reason: str = ""
    notes: str = ""
    created_at: float = Field(default_factory=time.time)


class ReviewReport(BaseModel):
    """Reviewer audit output — a structured verdict, evidence, and recommendation."""

    schema_version: str = "rsi_master.review.v1"
    reviewer_id: str
    worker_id: str
    experiment_id: str
    verdict: Literal["promote", "iterate", "discard", "unknown"] = "unknown"
    primary_bottleneck: str = ""
    confidence: float = 0.0
    evidence: list[str] = Field(default_factory=list)
    evidence_table: list[dict[str, Any]] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    leakage_risk: Literal["low", "medium", "high", "unknown"] = "unknown"
    inspected_samples: int = 0
    inspection_manifest: dict[str, Any] = Field(default_factory=dict)
    what_result_proves: list[str] = Field(default_factory=list)
    what_result_does_not_prove: list[str] = Field(default_factory=list)
    next_experiment_gradient: dict[str, Any] = Field(default_factory=dict)
    data_memory_recommendation: DataMemoryRecommendation = Field(default_factory=DataMemoryRecommendation)
    blocking_issues: list[str] = Field(default_factory=list)
    reviewer_notes: str = ""
    created_at: float = Field(default_factory=time.time)


# ─── DAG 节点模型 ─────────────────────────────────────────────────────────────


NodeStatus = Literal["pending", "running", "finished", "failed", "aborted"]


class WorkerNode(BaseModel):
    """A worker node in the workflow DAG.

    Worker 相当于 CPU 中的时钟周期激活单元, 每一次 main-agent 决定"下一步"
    就会实例化一个 WorkerNode; 每个 WorkerNode 可以再启动多个 experiments
    以及多个 reviewers。
    """

    node_type: Literal["worker"] = "worker"
    worker_id: str
    parent_worker_id: str | None = None
    child_worker_ids: list[str] = Field(default_factory=list)
    experiment_ids: list[str] = Field(default_factory=list)
    reviewer_ids: list[str] = Field(default_factory=list)
    status: NodeStatus = "pending"
    objective: str = ""
    best_score: float | None = None
    best_experiment_id: str | None = None
    workspace: str = ""
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


class ExperimentNode(BaseModel):
    """A single experiment executed inside a worker's submit call."""

    node_type: Literal["experiment"] = "experiment"
    experiment_id: str
    worker_id: str
    parent_experiment_id: str | None = None
    reviewer_ids: list[str] = Field(default_factory=list)
    status: NodeStatus = "pending"
    training_mode: str = "full_sft"
    benchmark: str = ""
    score: float | None = None
    workspace: str = ""
    submit_trial_path: str = ""
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


class ReviewerNode(BaseModel):
    """A reviewer node — a reviewer may audit multiple experiments (non 1:1)."""

    node_type: Literal["reviewer"] = "reviewer"
    reviewer_id: str
    worker_id: str
    reviewed_experiment_ids: list[str] = Field(default_factory=list)
    objective: str = Field(default="", description="What the reviewer should focus on / verify.")
    status: NodeStatus = "pending"
    verdict_summary: dict[str, Any] = Field(default_factory=dict)
    workspace: str = ""
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


# ─── Data Pool ────────────────────────────────────────────────────────────────


class DataPoolRecord(BaseModel):
    """Metadata for a data_pool entry stored under data_pool/{data_id}/config.json."""

    schema_version: str = "rsi_master.data_pool.v1"
    data_id: str
    source_worker_id: str | None = None
    source_experiment_id: str | None = None
    benchmark: str = ""
    artifact_type: str = "train_data"
    artifact_path: str = ""
    sha256: str = ""
    num_rows: int | None = None
    score: float | None = None
    promotion_status: Literal["raw", "candidate", "promoted", "discarded"] = "candidate"
    memory_recommendation: DataMemoryRecommendation | None = None
    review_verdict: str | None = None
    reused_by_experiments: list[str] = Field(default_factory=list)
    what_to_preserve: list[str] = Field(default_factory=list)
    what_to_change: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    descriptions: str = Field(
        default="",
        description="Free-text description: data source, composition, and intended use.",
    )
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


# ─── 事件流 ────────────────────────────────────────────────────────────────────


class OSEvent(BaseModel):
    """One structured line inside events.jsonl."""

    schema_version: str = "rsi_master.event.v1"
    type: str
    worker_id: str | None = None
    experiment_id: str | None = None
    reviewer_id: str | None = None
    data_id: str | None = None
    tool: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    time: float = Field(default_factory=time.time)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def dump_model(model: BaseModel) -> dict[str, Any]:
    """Serialize a Pydantic model (v1/v2 compatible)."""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()
