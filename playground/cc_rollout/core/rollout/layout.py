"""Run-dir layout: the workspace vs the claude-invisible configs.

cc_rollout 是**单 agent、无 branching** 的 rollout, 所以每个 run root 下的布局
比 cc_branching 简单 —— 没有 ``branches/<id>`` / ``config/<id>`` 两层, 而是三个
平级目录 (对应 docs/README.md 的 ``workspace / logs/configs`` 要求)::

    runs/<run_id>_<timestamp>/
    ├── workspace/             # WORKSPACE — the only thing the agent sees.
    │   ├── README.md           #   protocol (system prompt, --append-system-prompt)
    │   ├── BENCHMARK.md        #   eval spec (from benchmark root format/<bench_id>/)
    │   ├── GLOBAL_MEMORY.md    #   cross-agent scratchpad stub
    │   ├── public/             #   seeded backend snapshots + CLI launchers
    │   ├── FINISH              #   agent submits: `echo <experiment_id> > FINISH`
    │   └── (cc_with_expos) experiment_os/  SUBMIT_GUIDE.md  expos
    ├── configs/                # CONFIG — system-owned, invisible to the agent.
    │   ├── .claude-config/     #   session transcript (CLAUDE_CONFIG_DIR)
    │   ├── run.json            #   {mode, session_id, task, ...}
    │   └── settings.json       #   permission policy (passed via --settings)
    └── logs/                   # agent.log (claude -p stdout/stderr tee)
"""

from __future__ import annotations

from pathlib import Path


def workspace_dir_for(run_root: Path | str) -> Path:
    """``run_root`` → the agent's workspace dir."""
    return Path(run_root).resolve() / "workspace"


def config_dir_for(workspace_dir: Path | str) -> Path:
    """Map the workspace dir to its sibling claude-invisible configs dir.

    ``<run_root>/workspace`` → ``<run_root>/configs``.  Used by the runner /
    orchestrator / seed to resolve where system state lives without ever putting
    it in the workspace.
    """
    workspace_dir = Path(workspace_dir).resolve()
    return workspace_dir.parent / "configs"


def logs_dir_for(workspace_dir: Path | str) -> Path:
    """Map the workspace dir to its sibling logs dir (agent.log etc.)."""
    workspace_dir = Path(workspace_dir).resolve()
    return workspace_dir.parent / "logs"
