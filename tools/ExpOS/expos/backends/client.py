"""HTTP client for the ExpOS backend — also implements SubmitBackend.

对应 ``backend/gpu_backend/client.py``: 一个同步的 httpx 客户端, 直接实现
:class:`~expos.service.worker_service.SubmitBackend` 协议, 可以被注入到
``WorkerService`` 里。

典型用法::

    from expos.backends import build_default_client
    from expos.service.experiment_os import ExperimentOS
    from expos.service.worker_service import WorkerService

    client = build_default_client()
    worker = WorkerService(os_=ExperimentOS(root), submit_backend=client)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import httpx

from .config import BackendConfig, load_backend_config
from .models import SubmitRequest, SubmitResponse, TrialInfo

logger = logging.getLogger(__name__)


class BackendError(RuntimeError):
    """Raised when the backend HTTP call fails outright (network/HTTP 5xx)."""


class SubmitBackendClient:
    """Synchronous HTTP client for the ExpOS backend.

    也是 :class:`~expos.service.worker_service.SubmitBackend` 协议的实现:
    ``client.submit(...)`` 直接以 HTTP 触发 ``POST /submit``。
    """

    def __init__(self, config: BackendConfig | None = None):
        self.config = config or load_backend_config()
        self._client = httpx.Client(
            base_url=self.config.base_url,
            timeout=httpx.Timeout(
                self.config.submit_timeout,
                connect=self.config.connect_timeout,
            ),
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "SubmitBackendClient":  # pragma: no cover
        return self

    def __exit__(self, *_: Any) -> None:  # pragma: no cover
        self.close()

    # ── SubmitBackend protocol ───────────────────────────────────────────────

    def submit(
        self,
        *,
        experiment_id: str,
        worker_id: str,
        benchmark: str,
        train_data_path: str,
        training_config_path: str,
        training_mode: str,
        experiment_workspace: Path,
        objective: str = "",
        hypothesis: str = "",
        changed_variable: str = "",
        parent_experiment_id: str | None = None,
        bench_id: str | None = None,
        eval_all: dict[str, Any] | None = None,
        gpu_nums: int = 1,
        need_exclusive_gpu: bool = True,  # 默认独占 (2026-08-29): 整卡到会话结束
        max_mem_quota: int | None = None,
        slime_config: dict[str, Any] | None = None,
        allowed_training_modes: list[str] | None = None,
    ) -> dict[str, Any]:
        """Fire ``POST /submit`` and return the response as a plain dict.

        网络/服务端故障会被降级成一条 ``status="failed"`` 记录, 让
        WorkerService 层照常写入 ExperimentResult 而不是抛异常终止 worker。

        ``need_exclusive_gpu``: 默认 True (2026-08-29 用户要求)——每个任务独占
        整卡到 tmux 会话结束, 长挂训练+serve+eval 不被抢占; 需要共享时才显式传 False
        (max_mem_quota 配额估算仍生效, 只是独占优先)。
        ``max_mem_quota``: 每卡最低可用显存 (MB) 放置需求; render 期估算。

        ``slime_config``: ``training_mode="slime_sft"`` 时透传给后端 pipeline
        (键映射到 launch_sft.sh 的 env)。
        """
        request = SubmitRequest(
            experiment_id=experiment_id,
            worker_id=worker_id,
            benchmark=benchmark,
            train_data_path=str(train_data_path),
            training_config_path=str(training_config_path),
            training_mode=training_mode,  # type: ignore[arg-type]
            experiment_workspace=str(experiment_workspace),
            objective=objective,
            hypothesis=hypothesis,
            changed_variable=changed_variable,
            parent_experiment_id=parent_experiment_id,
            bench_id=bench_id,
            eval_all=eval_all or {},
            gpu_nums=gpu_nums,
            need_exclusive_gpu=need_exclusive_gpu,
            max_mem_quota=max_mem_quota,
            slime_config=slime_config or {},
            allowed_training_modes=allowed_training_modes or [],
        )
        try:
            resp = self._client.post("/submit", json=request.model_dump())
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "backend /submit HTTP %s: %s", exc.response.status_code, exc.response.text
            )
            return SubmitResponse(
                status="failed",
                failure_reason=(
                    f"backend HTTP {exc.response.status_code}: {exc.response.text[:400]}"
                ),
                benchmark=benchmark,
                training_mode=training_mode,
                trial_path=str(experiment_workspace),
            ).to_dict()
        except httpx.RequestError as exc:
            logger.error("backend /submit request error: %s", exc)
            return SubmitResponse(
                status="failed",
                failure_reason=f"backend network error: {exc}",
                benchmark=benchmark,
                training_mode=training_mode,
                trial_path=str(experiment_workspace),
            ).to_dict()
        return SubmitResponse(**resp.json()).to_dict()

    # ── auxiliary endpoints ──────────────────────────────────────────────────

    def get_queue(self) -> dict[str, Any]:
        resp = self._client.get("/queue", timeout=self.config.status_timeout)
        resp.raise_for_status()
        return resp.json()

    def cancel_queued_script(self, script_id: str) -> dict[str, Any]:
        resp = self._client.post(f"/queue/{script_id}/cancel", timeout=self.config.status_timeout)
        resp.raise_for_status()
        return resp.json()

    def health(self) -> dict[str, Any]:
        resp = self._client.get("/health", timeout=self.config.status_timeout)
        resp.raise_for_status()
        return resp.json()

    def trial_info(self, trial_id: str) -> TrialInfo:
        resp = self._client.get(
            f"/trials/{trial_id}", timeout=self.config.status_timeout
        )
        if resp.status_code == 404:
            raise BackendError(f"trial not found: {trial_id}")
        resp.raise_for_status()
        return TrialInfo(**resp.json())


def build_default_client(config_path: str | Path | None = None) -> SubmitBackendClient:
    """Convenience factory used by :mod:`expos.cli.context`."""
    return SubmitBackendClient(load_backend_config(config_path))
