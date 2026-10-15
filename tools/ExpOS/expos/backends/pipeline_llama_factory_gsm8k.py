"""Async pipeline driver (legacy shim): llama-factory SFT + vLLM serve + eval.

**本文件现在是 re-export 壳** —— 真正的实现已迁到
:mod:`expos.backends.pipelines` (package, 按 ``training_mode`` 分发:

- ``llama_factory_sft`` (全参 SFT, 唯一 llama_factory 模式; LoRA 已下线):
  LLaMA-Factory SFT → vLLM serve → 评测 (与旧单文件行为逐字节一致);
- ``slime_sft``: slime (Megatron-LM) 数据预检 + ``launch_sft.sh`` 训练,
  serve/eval 本期跳过。

保留本模块名与 driver 字符串 ``pipeline_llama_factory_gsm8k`` 是为了向后兼容
旧 config / 旧 import 路径; ``server._run_trial`` 同时接受 ``pipelines`` 与旧名。
"""

from __future__ import annotations

from .pipelines import build_pipeline_script, run_trial

#: server 端接受的 driver 名集合 (见 :func:`expos.backends.server._run_trial`)。
DRIVER_NAMES = {"pipelines", "pipeline_llama_factory_gsm8k"}

__all__ = ["build_pipeline_script", "run_trial", "DRIVER_NAMES"]
