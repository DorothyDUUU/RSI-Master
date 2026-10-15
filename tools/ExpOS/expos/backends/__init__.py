"""FastAPI-based training/eval backend for ExpOS.

参考 ``backend/server`` + ``backend/gpu_backend`` 的解耦模式:

    +--------------+  httpx  +----------------+
    | WorkerService| ------> | server.py      |
    |  (via Client)|         |  FastAPI       |
    +--------------+         +--------|-------+
                                       |
                                       v
                               (real GPU backend — TODO)

模块布局:
    server.py    FastAPI 应用, 声明 /submit / /trials/{trial_id} 等 endpoint
                 目前只把请求登记下来并返回 pending; 实际训练逻辑会在下一轮迁移
                 到 ``backend/gpu_backend`` 后调用它
    client.py    httpx.Client, 同时也是 SubmitBackend 协议实现
    config.py    读取本包目录下的 config.yaml, 优先级: 显式路径 > 包内 default
    config.yaml  Server 的 host/port + timeout
    cli.py       ``python -m expos.backends`` — 启动 server / 客户端调 submit
    models.py    Pydantic v2 请求/响应模型

CLI 层 (`expos.cli`) 在启动 WorkerService 时, 会通过
:func:`build_default_client` 拿到一个 :class:`SubmitBackendClient`, 从而实现
Service / Backend 解耦。
"""

from .client import SubmitBackendClient, build_default_client
from .config import BackendConfig, load_backend_config
from .models import SubmitRequest, SubmitResponse, TrialInfo

__all__ = [
    "BackendConfig",
    "SubmitBackendClient",
    "SubmitRequest",
    "SubmitResponse",
    "TrialInfo",
    "build_default_client",
    "load_backend_config",
]
