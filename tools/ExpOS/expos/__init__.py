"""ExpOS — file-backed ExperimentOS with a graph-based workflow and a CLI.

顶层包只做门面导出, 让下游 (playground / agent-loop) 可以用::

    from expos import ExperimentOS, StateService, WorkerService, ReviewerService, DataPoolService

而不需要感知 service / cli 目录结构。CLI frontend 在 :mod:`expos.cli` 中。
"""

from .service.contracts import (
    ArtifactRecord,
    DataMemoryRecommendation,
    DataPoolRecord,
    DataReusePlan,
    ExperimentNode,
    ExperimentResult,
    OSEvent,
    ReviewerNode,
    ReviewReport,
    WorkOrder,
    WorkerNode,
    dump_model,
)
from .service.data_pool import DataPoolService
from .service.events import EventStream
from .service.experiment_os import ExperimentOS
from .service.graph import WorkflowGraph
from .service.reviewer_service import ReviewerService
from .service.state_service import StateService
from .service.worker_service import SubmitBackend, WorkerService

# HTTP-based training/eval backend. FastAPI server + httpx client.
from .backends import SubmitBackendClient, build_default_client

__all__ = [
    "ArtifactRecord",
    "DataMemoryRecommendation",
    "DataPoolRecord",
    "DataReusePlan",
    "DataPoolService",
    "EventStream",
    "ExperimentNode",
    "ExperimentOS",
    "ExperimentResult",
    "OSEvent",
    "ReviewReport",
    "ReviewerNode",
    "ReviewerService",
    "StateService",
    "SubmitBackend",
    "SubmitBackendClient",
    "WorkOrder",
    "WorkerNode",
    "WorkerService",
    "WorkflowGraph",
    "build_default_client",
    "dump_model",
]
