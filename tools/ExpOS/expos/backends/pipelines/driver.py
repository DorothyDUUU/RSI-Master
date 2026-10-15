"""Trial driver: render pipeline.sh, submit it to gpu_backend (async).

``run_trial`` 是 SubmitBackend 契约的实现: 组装 pipeline.sh 提交到 gpu_backend
后**立即**返回 ``status="running"`` + SCRIPT_ID, 最终 metrics 由 pipeline bash
的 EXIT trap 写进 workspace/result.json。
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..models import SubmitRequest, SubmitResponse
from .render import build_pipeline_script, estimate_max_mem_quota

logger = logging.getLogger(__name__)


def run_trial(request: SubmitRequest) -> SubmitResponse:
    """Build pipeline.sh, submit it to gpu_backend, return an async receipt.

    We do NOT wait for training to finish; ``status="running"`` means the
    pipeline is queued/started. Final metrics land in ``result.json`` when
    the bash script's EXIT trap fires.
    """
    workspace = Path(request.experiment_workspace)
    try:
        pipeline_script, paths = build_pipeline_script(request, workspace)
    except (OSError, ValueError) as exc:
        # ValueError: 渲染期配置错误 (如 slime_sft 缺 model_dir / torch_dist_dir)
        return SubmitResponse(
            status="failed",
            failure_reason=f"failed to render pipeline.sh: {exc}",
            benchmark=request.benchmark,
            training_mode=request.training_mode,
            trial_path=str(workspace),
            train_data_path=request.train_data_path,
        )

    # Lazy import so ExpOS backend 不硬依赖 gpu_backend 包 (方便 stub 模式)。
    # gpu_backend 包内部用 `from ..server.models import ...` 相对导入, 要求
    # 自身作为 `backend.gpu_backend` 顶级子包 —— 因此这里也走 `backend.` 前缀,
    # 只有 repo root 在 PYTHONPATH 上时才能 import (符合项目其他调用点的约定).
    try:
        from backend.gpu_backend.client import BackendClient
        from backend.gpu_backend.config import get_backend_url
    except ImportError as exc:
        return SubmitResponse(
            status="failed",
            failure_reason=(
                f"backend.gpu_backend package not importable: {exc}. Make sure "
                "the repo root is on PYTHONPATH."
            ),
            benchmark=request.benchmark,
            training_mode=request.training_mode,
            trial_path=str(workspace),
            train_data_path=request.train_data_path,
            pipeline_script_path=paths["pipeline_script_path"],
            checkpoint_dir=paths["checkpoint_dir"],
            tensorboard_dir=paths["tensorboard_dir"],
            eval_dir=paths["eval_dir"],
            result_json_path=paths["result_json_path"],
        )

    try:
        client = BackendClient(base_url=get_backend_url(), timeout=60.0)
        resp = client.submit_script(
            command_path=str(pipeline_script),
            timeout=0,  # 无超时 —— 训练 + 评测可能很长, 交给 gpu_backend 侧的策略。
            run_in_background=True,
            gpu_nums=request.gpu_nums,
            # R2 (2026-08-26): 每卡最低可用显存 (MB) 放置需求, 调度器对
            # free_memory_mb < quota 的卡直接跳过 —— 防止 35B job 被放到只剩 17MB
            # free 的卡上 OOM。quota 由 render 期按训练峰值/serve util 估算。
            max_mem_quota=estimate_max_mem_quota(request),
            need_exclusive_gpu=request.need_exclusive_gpu,
            priority="high",
            queue_owner=f"{workspace.parent.parent.resolve()}::{request.worker_id}",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("gpu_backend submit_script failed")
        return SubmitResponse(
            status="failed",
            failure_reason=f"gpu_backend submit_script failed: {exc}",
            benchmark=request.benchmark,
            training_mode=request.training_mode,
            trial_path=str(workspace),
            train_data_path=request.train_data_path,
            pipeline_script_path=paths["pipeline_script_path"],
            checkpoint_dir=paths["checkpoint_dir"],
            tensorboard_dir=paths["tensorboard_dir"],
            eval_dir=paths["eval_dir"],
            result_json_path=paths["result_json_path"],
        )

    status_str = str(getattr(resp.status, "value", resp.status))
    # gpu_backend 侧: 提交成功后一般是 QUEUED / RUNNING; 都翻译成 running。
    if status_str in {"queued", "running", "pending"}:
        submit_status = "running"
    elif status_str == "completed":
        submit_status = "completed"
    elif status_str == "rejected":
        submit_status = "deferred"
    else:
        submit_status = "failed"

    summary = (resp.message if submit_status == "deferred" else (
        f"pipeline queued on gpu_backend as SCRIPT_ID={resp.script_id} "
        f"(priority=high, background=True). Metrics will be written to "
        f"{paths['result_json_path']} when the bash script's EXIT trap fires."
    ))
    return SubmitResponse(
        status=submit_status,  # type: ignore[arg-type]
        score=None,
        benchmark=request.benchmark,
        training_mode=request.training_mode,
        trial_path=str(workspace),
        train_data_path=request.train_data_path,
        gpu_script_id=str(resp.script_id or ""),
        gpu_session_id=str(resp.session_id or ""),
        gpu_log_path=str(resp.log_path or ""),
        pipeline_script_path=paths["pipeline_script_path"],
        checkpoint_dir=paths["checkpoint_dir"],
        tensorboard_dir=paths["tensorboard_dir"],
        eval_dir=paths["eval_dir"],
        result_json_path=paths["result_json_path"],
        summary=summary,
        failure_reason=(resp.message or "") if submit_status == "failed" else "",
    )
