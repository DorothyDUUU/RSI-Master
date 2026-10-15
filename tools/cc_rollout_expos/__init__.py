"""cc_rollout_expos — CLI-form ExpOS entry point for cc_rollout (cc_with_expos).

顶层独立包 (tools/cc_rollout_expos), 不依赖 playground/。cc_with_expos seed 时
把本包复制进 ``workspace/public/cc_rollout_expos/``, 由 ``workspace/expos``
launcher ``python -m cc_rollout_expos`` 执行 —— workspace 里的副本就是被解析
的副本 (launcher 的 PYTHONPATH 把 workspace/public 放在最前, tools/ 不在其上)。

包装 ``expos.cli.main.main`` 并注入 :class:`SubmitPipelineBackend`, 让 ``worker
submit`` 走 serverless submit 训练管线 (``backend.backend.submit_pipeline``
``run_trial`` → gpu_backend :8989) 而不是 ExpOS FastAPI HTTP 服务 (:8990)。
其余子命令与框架自带 CLI 行为完全一致。

import 依赖: ``expos`` (tools/ExpOS) 与 ``backend`` (repo root) 都在 launcher
烘焙的 PYTHONPATH 上。
"""

from .expos_cli import main

__all__ = ["main"]
__version__ = "0.1.0"
