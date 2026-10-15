"""cc_rollout playground core — registers the playground with evomaster."""

from __future__ import annotations

import sys
from pathlib import Path

# 保证 `tools/ExpOS` 里的 expos 包可以被顶层脚本 (run.py) 直接 import ——
# cc_with_expos 模式的 seed 要调 ExperimentOS.initialize, expos_cli 包装
# `expos.cli.main` (v2 同款注入, 见 agent_loop/__init__.py)。
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_EXPOS_DIR = _PROJECT_ROOT / "tools" / "ExpOS"
if _EXPOS_DIR.exists() and str(_EXPOS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPOS_DIR))

from .playground import ClaudeCodeRolloutPlayground  # noqa: E402  (sys.path 注入先)

__all__ = ["ClaudeCodeRolloutPlayground"]
