"""cc_rollout playground: 一个 CC 一路 Rollout 到底的后训练 playground。

两种 rollout 方式 (config ``mode:``):
- **cc_naive** — 只 initialize workspace (backend copy + README/BENCHMARK/
  GLOBAL_MEMORY), 启动一个 CC 直接 Rollout 到底, 不做 branching。
- **cc_with_expos** — 同 naive, 再 seed 一份 CLI 形式的 ExpOS
  (experiment_os + SUBMIT_GUIDE.md + ``./expos`` launcher): 一个 CC 兼任
  MainAgent/Worker/Reviewer 三角色, submit 走 serverless submit 训练管线。

核心代码在 ``playground/cc_rollout/core/rollout/``, 经 ``run.py --agent
cc_rollout`` 派发。
"""

__version__ = "0.1.0"
