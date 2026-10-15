"""FastAPI server exposing the ExpOS training/eval backend.

对应 ``backend/server/main.py`` 的角色: 一个可独立启动的 HTTP 服务, 让
``expos.WorkerService`` 通过 HTTP 而非 Python import 触发训练。

Endpoint 布局::

    POST /submit                提交一次训练+评测 (SubmitRequest -> SubmitResponse)
    GET  /trials/{trial_id}     查询 trial 状态 (长任务时轮询)
    GET  /health                健康检查
    GET  /                      版本信息

当前实现 (driver=stub):
    - handler 把请求登记到本地 trials 表, 返回 status="pending".
    - 真正的训练调用 (原来的 ``evomaster.posttrain.submit_training_eval``)
      会在 ``backend/gpu_backend`` 迁移完成后接入 ``_run_trial`` 位置。

启动方式::

    python -m expos.backends server start
    # 或直接:
    uvicorn expos.backends.server:app --host 0.0.0.0 --port 8990
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

from .config import load_backend_config
from .models import (
    HealthResponse,
    SubmitRequest,
    SubmitResponse,
    TrialInfo,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


# ── Trial store ─────────────────────────────────────────────────────────────


class TrialStore:
    """In-memory trial registry.

    上线时可以换成基于 :mod:`expos.service.io_utils` 的持久化实现; 这里保持
    简单, 因为 server 目前只做记录并返回 pending。
    """

    def __init__(self):
        self._trials: dict[str, TrialInfo] = {}
        self._lock = threading.Lock()

    def register(self, request: SubmitRequest) -> TrialInfo:
        trial_id = f"trial_{uuid.uuid4().hex[:12]}"
        info = TrialInfo(
            trial_id=trial_id,
            experiment_id=request.experiment_id,
            worker_id=request.worker_id,
            status="queued",
            submitted_at=time.time(),
            request=request,
        )
        with self._lock:
            self._trials[trial_id] = info
        return info

    def get(self, trial_id: str) -> TrialInfo | None:
        with self._lock:
            return self._trials.get(trial_id)

    def update(self, trial_id: str, **fields: Any) -> TrialInfo:
        with self._lock:
            info = self._trials[trial_id]
            data = info.model_dump()
            data.update(fields)
            new_info = TrialInfo(**data)
            self._trials[trial_id] = new_info
            return new_info

    def count(self) -> int:
        with self._lock:
            return len(self._trials)


# ── Trial execution stub ────────────────────────────────────────────────────


def _run_trial(request: SubmitRequest, config_driver: str) -> SubmitResponse:
    """Dispatch a submit request to the configured backend driver.

    支持的 driver:
        - ``pipelines`` (推荐): 按 training_mode 分发的 pipeline package
          (:mod:`expos.backends.pipelines`), 组装 pipeline.sh 走 gpu_backend
          异步提交, 立即返回 ``status="running"`` + SCRIPT_ID。
        - ``pipeline_llama_factory_gsm8k``: 旧名, 现在是指向 ``pipelines`` 的
          re-export 壳, 行为一致 (向后兼容, 老 run 的 config 不用改)。
        - ``stub``: 只落盘 workspace, 返回 pending. 保留给单元测试。
    """
    # 保底: 保证 experiment workspace 存在
    try:
        Path(request.experiment_workspace).mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # noqa: BLE001
        return SubmitResponse(
            status="failed",
            failure_reason=f"failed to prepare workspace: {exc}",
            benchmark=request.benchmark,
            training_mode=request.training_mode,
        )

    if config_driver in {"pipelines", "pipeline_llama_factory_gsm8k"}:
        # 延迟 import, 避免 gpu_backend 依赖污染 stub 模式的启动。
        from .pipelines import run_trial as run_pipeline
        return run_pipeline(request)

    if config_driver == "stub":
        return SubmitResponse(
            status="pending",
            benchmark=request.benchmark,
            training_mode=request.training_mode,
            trial_path=request.experiment_workspace,
            train_data_path=request.train_data_path,
            failure_reason="",
            summary=(
                "backend driver=stub — training + eval is not wired. "
                "This response is a receipt of the submit call only."
            ),
        )

    return SubmitResponse(
        status="failed",
        failure_reason=f"unknown backend driver: {config_driver}",
        benchmark=request.benchmark,
        training_mode=request.training_mode,
    )


# ── FastAPI app ─────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_backend_config()
    app.state.config = cfg
    app.state.trials = TrialStore()
    logger.info(
        "ExpOS backend server up (driver=%s, url=%s)", cfg.driver, cfg.base_url
    )
    yield
    logger.info("ExpOS backend server shutting down")


app = FastAPI(
    title="ExpOS Backend",
    description="Training + eval HTTP backend for ExperimentOS workers",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": "expos-backend", "version": "0.1.0"}


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    cfg = app.state.config
    return HealthResponse(driver=cfg.driver, active_trials=app.state.trials.count())


@app.post("/submit", response_model=SubmitResponse)
async def submit(request: SubmitRequest) -> SubmitResponse:
    """Register the trial and (in stub mode) return a pending receipt."""
    trials: TrialStore = app.state.trials
    cfg = app.state.config
    info = trials.register(request)
    response = _run_trial(request, cfg.driver)
    trials.update(
        info.trial_id,
        status="pending" if response.status == "pending" else response.status,
        response=response,
        finished_at=None if response.status in {"queued", "running"} else time.time(),
    )
    # 把 server 端 trial_id 回填, 方便客户端后续 GET /trials/{id}
    if not response.trial_id:
        response = response.model_copy(update={"trial_id": info.trial_id})
    return response


@app.get("/trials/{trial_id}", response_model=TrialInfo)
async def trial_info(trial_id: str) -> TrialInfo:
    info = app.state.trials.get(trial_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"trial not found: {trial_id}")
    return info


@app.get("/queue")
def queue_snapshot() -> dict[str, Any]:
    # Resolve on the same host/config as the pipeline driver, not the Agent CPU.
    from backend.gpu_backend.client import BackendClient
    from backend.gpu_backend.config import get_backend_url
    with BackendClient(base_url=get_backend_url(), timeout=10) as client:
        return client.get_queue()


@app.post("/queue/{script_id}/cancel")
def cancel_queued(script_id: str) -> dict[str, Any]:
    from backend.gpu_backend.client import BackendClient
    from backend.gpu_backend.config import get_backend_url
    with BackendClient(base_url=get_backend_url(), timeout=10) as client:
        return client.cancel_queued_script(script_id)


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    cfg = load_backend_config()
    uvicorn.run(app, host=cfg.host, port=cfg.port)
