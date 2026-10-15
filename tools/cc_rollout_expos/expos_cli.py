"""CLI-form ExpOS entry point for cc_with_expos (单 CC 兼任三角色).

包装真实的 ``expos.cli.main.main``, 注入 :class:`SubmitPipelineBackend`, 让
``worker submit`` 走 serverless submit 训练管线 (``submit_pipeline.run_trial``
→ gpu_backend :8989) 而不是 ExpOS FastAPI HTTP 服务 (:8990)。其余子命令
(init / dump-state / state add-new-worker / worker add-data / reviewer
inspect-experiment ...) 与框架自带 CLI 行为完全一致 —— 相同工具接口, 文件系统即
DAG (EXPOS_ROOT 下的 state/worker/experiments/reviews/data_pool)。

顶层独立包 (tools/cc_rollout_expos, 不依赖 playground/): cc_with_expos seed 时
复制进 ``workspace/public/cc_rollout_expos/``, ``workspace/expos`` launcher
(seed.py) 设好 PYTHONPATH (workspace/public 最前 + repo root + tools/ExpOS)
和 EXPOS_ROOT 后 ``python -m cc_rollout_expos "$@"``。
"""

from __future__ import annotations

import sys
from typing import Sequence

from .expos_cli_backend import SubmitPipelineBackend


def main(
    argv: Sequence[str] | None = None,
    *,
    submit_backend: object | None = None,
) -> int:
    """Dispatch one expos CLI invocation with the submit pipeline backend.

    *submit_backend* 是注入缝 (测试里传 FakeSubmitBackend, hermetic); 默认用
    :class:`SubmitPipelineBackend` (serverless submit 训练管线)。
    """
    # 延迟导入: tools/ExpOS 在 launcher 的 PYTHONPATH 上; 纯读命令 (dump-state /
    # events tail) 即使本机没有 gpu_backend 依赖也要能用。
    from expos.cli.main import main as expos_main

    return expos_main(
        argv,
        submit_backend=submit_backend or SubmitPipelineBackend(),  # type: ignore[arg-type]
    )


if __name__ == "__main__":
    raise SystemExit(main())
