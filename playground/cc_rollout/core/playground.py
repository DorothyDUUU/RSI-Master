"""cc_rollout playground adapter — bridges rsi_master run.py into the rollout loop.

仿 ``playground/cc_branching/core/playground.py``: 一个类通过 ``run.py --agent
cc_rollout`` 派发, 不做 BasePlayground 子类 (配置格式不匹配 EvoMasterConfig 的
pydantic schema)。run.py 只需要一个注册类提供 ``set_run_dir(run_dir)`` +
``run(task_description, ...)`` —— 其余 (seeding / agent launch / summary) 全部
委托给 ``rollout.orchestrator.run``。

流程:
1. run.py 调 ``set_run_dir(run_dir)`` (默认 ``runs/cc_rollout_<ts>``)。
2. run.py 调 ``run(task_description=..., images=...)``。
3. adapter ``_materialize``: 复制 rollout config (``configs/cc_rollout/*.yaml``),
   注入 ``run.task = task_description``, 把 run root 指到 run.py 管理的 run_dir
   (``RSI_MASTER_RUN_ROOT``), 委托 ``orchestrator.run`` — 它按 ``config.mode``
   决定 cc_naive / cc_with_expos。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from evomaster.core import register_playground

from .rollout._paths import rsi_master_root, runs_root


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


@register_playground("cc_rollout")
class ClaudeCodeRolloutPlayground:
    """Single-CC post-training rollout as a rsi_master playground."""

    def __init__(self, config_dir: str | Path | None = None, config_path: str | Path | None = None):
        self.run_dir: Path | None = None
        self._run_root: Path | None = None
        self._config_dir = Path(config_dir).resolve() if config_dir is not None else None
        self._config_path = Path(config_path).resolve() if config_path is not None else None

    # -- run.py lifecycle --------------------------------------------------

    def set_run_dir(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir).resolve()

    @property
    def rollout_config_path(self) -> Path:
        """The rollout-format config file (run.py --config, else configs/cc_rollout/config.yaml)."""
        if self._config_path is not None:
            return self._config_path
        if self._config_dir is not None:
            cfg = self._config_dir / "config.yaml"
            if cfg.exists():
                return cfg
        return rsi_master_root() / "configs" / "cc_rollout" / "config.yaml"

    def _materialize(self, task_description: str, task_id: str) -> Path:
        """Copy config → run_dir, inject run.task + sidecar token, return the path."""
        src = self.rollout_config_path
        if not src.exists():
            raise FileNotFoundError(f"cc_rollout config not found: {src}")
        cfg = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
        if not isinstance(cfg, dict):
            raise TypeError(f"cc_rollout config root must be a mapping: {src}")

        run_cfg = _as_dict(cfg.get("run"))
        run_cfg["task"] = str(task_description)
        cfg["run"] = run_cfg

        # Sidecar auth token: config ships empty (never commit the secret);
        # inject from env when present so the claude -p agent can authenticate.
        cc = _as_dict(cfg.get("claude"))
        if not cc.get("auth_token"):
            cc["auth_token"] = os.environ.get("CC_ROLLOUT_AUTH_TOKEN", "")
        cfg["claude"] = cc

        run_root = self.run_dir if self.run_dir is not None else runs_root()
        run_root.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("RSI_MASTER_RUN_ROOT", str(run_root))

        out = run_root / "config.rollout.yaml"
        out.write_text(
            "# Auto-materialized by cc_rollout playground adapter.\n"
            f"# Base config: {src}\n"
            + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return out

    def run(
        self,
        task_description: str,
        task_id: str = "exp_001",
        images: list[str] | None = None,
        on_step=None,
    ) -> dict:
        """Entry point invoked by run.py — delegates to the rollout orchestrator."""
        from .rollout import orchestrator

        config_path = self._materialize(task_description, task_id)
        run_root = orchestrator.run(config_path, run_name=None)
        return {
            "task_id": task_id,
            "status": "completed",
            "run_root": str(run_root),
            "playground": "cc_rollout",
        }
