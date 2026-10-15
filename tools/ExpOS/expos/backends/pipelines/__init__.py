"""Pipeline drivers — package 化替代原单文件 ``pipeline_llama_factory_gsm8k.py``.

按 ``training_mode`` 分发到不同框架的组装器 (目前: llama-factory SFT / slime SFT),
为后续支持更多 infra 预留扩展点。对外暴露的 API 与旧模块一致::

    from expos.backends.pipelines import build_pipeline_script, run_trial

旧的 driver 名 ``pipeline_llama_factory_gsm8k`` 仍被接受 (server.py 双名兼容),
该模块瘦身成了指向本包的 re-export 壳。
"""

from __future__ import annotations

from .driver import run_trial
from .render import build_pipeline_script

# server 端 `_run_trial` 接受的 driver 名集合。
DRIVER_NAMES = {"pipelines", "pipeline_llama_factory_gsm8k"}

__all__ = ["build_pipeline_script", "run_trial", "DRIVER_NAMES"]
