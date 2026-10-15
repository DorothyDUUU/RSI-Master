"""Shared CLI context — resolves the ExperimentOS root once per invocation.

命令行工具通过 ``--root <path>`` 或环境变量 ``EXPOS_ROOT`` 指定 ExperimentOS
数据目录。SubmitBackend 走 HTTP: 每次 CLI 调用会从
``expos/backends/config.yaml`` 读取 URL, 并把
:class:`~expos.backends.SubmitBackendClient` 注入到 :class:`WorkerService`。
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from ..service.data_pool import DataPoolService
from ..service.experiment_os import ExperimentOS
from ..service.reviewer_service import ReviewerService
from ..service.state_service import StateService
from ..service.worker_service import SubmitBackend, WorkerService

logger = logging.getLogger(__name__)


ENV_ROOT = "EXPOS_ROOT"


@dataclass
class CLIContext:
    root: Path
    os: ExperimentOS
    state: StateService
    worker: WorkerService
    reviewer: ReviewerService
    data_pool: DataPoolService


def resolve_root(namespace: argparse.Namespace) -> Path:
    """Return the ExperimentOS root path from --root or ``$EXPOS_ROOT``."""
    root = getattr(namespace, "root", None) or os.environ.get(ENV_ROOT)
    if not root:
        raise SystemExit(
            "expos: --root is required (or set the EXPOS_ROOT environment variable)"
        )
    return Path(root).expanduser().resolve()


def resolve_submit_backend(
    explicit: SubmitBackend | None = None,
    *,
    backend_config: str | Path | None = None,
) -> SubmitBackend | None:
    """Pick a :class:`SubmitBackend` for this CLI invocation.

    优先级:
        1. Python 调用方显式注入的 backend (测试 / 自定义 stub);
        2. 默认从 ``expos.backends.config.yaml`` 构造 HTTP 客户端;
           若 HTTP 客户端构造失败 (依赖缺失), 返回 ``None`` 让
           :class:`WorkerService` 走 pending dry-run 记录, 不阻塞 CLI。
    """
    if explicit is not None:
        return explicit
    try:
        from ..backends import build_default_client
    except Exception as exc:  # noqa: BLE001 — 环境缺 httpx 时降级
        logger.warning("expos.backends unavailable, falling back to None backend: %s", exc)
        return None
    try:
        return build_default_client(backend_config)
    except Exception as exc:  # noqa: BLE001 — 配置加载失败降级
        logger.warning("failed to build backend client, using pending fallback: %s", exc)
        return None


def build_context(
    namespace: argparse.Namespace,
    *,
    submit_backend: SubmitBackend | None = None,
) -> CLIContext:
    """Instantiate all services for one CLI invocation."""
    root = resolve_root(namespace)
    os_ = ExperimentOS(root)
    data_pool = DataPoolService(os_)
    backend = resolve_submit_backend(
        submit_backend,
        backend_config=getattr(namespace, "backend_config", None),
    )
    return CLIContext(
        root=root,
        os=os_,
        state=StateService(os_),
        worker=WorkerService(os_, submit_backend=backend, data_pool=data_pool),
        reviewer=ReviewerService(os_, data_pool=data_pool),
        data_pool=data_pool,
    )
