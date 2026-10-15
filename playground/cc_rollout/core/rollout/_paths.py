"""Shared path resolution for the cc_rollout rollout package.

移植自 ``playground/cc_branching/core/rollout/_paths.py``。本包住在
``playground/cc_rollout/core/rollout/``, 仓库根 = 上溯 4 层。
为稳妥, 解析顺序: 环境变量 ``RSI_MASTER_ROOT``(部署环境可显式指定) >
基于文件位置的推断。

rsi_master 根下各依赖:
    <root>/backend           -> gpu_backend / server (PYTHONPATH namespace 源)
    <root>/backend/backend   -> eval_all / slime / submit_pipeline / llama_factory ...
    <root>/public            -> vllm_env / llama_factory_env / datasets
    <root>/runs              -> 默认运行目录 (run.py 约定)
"""

from __future__ import annotations

import os
from pathlib import Path

# 本文件: .../playground/cc_rollout/core/rollout/_paths.py
_ROLLOUT_DIR = Path(__file__).resolve().parent
# rollout -> core -> cc_rollout -> playground -> rsi_master 根
_FALLBACK_ROOT = _ROLLOUT_DIR.parents[3]


def rsi_master_root() -> Path:
    """The rsi_master repo root (env override first)."""
    env = os.environ.get("RSI_MASTER_ROOT")
    if env:
        return Path(env).resolve()
    return _FALLBACK_ROOT


def backend_root() -> Path:
    return rsi_master_root() / "backend"


def public_root() -> Path:
    return rsi_master_root() / "public"


def runs_root() -> Path:
    """Default run root.  ``RSI_MASTER_RUN_ROOT`` overrides (playground adapter
    sets it to the run.py-managed run_dir when present)."""
    env = os.environ.get("RSI_MASTER_RUN_ROOT")
    if env:
        return Path(env).resolve()
    return rsi_master_root() / "runs"
