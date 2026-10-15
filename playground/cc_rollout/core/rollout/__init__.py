"""cc_rollout rollout 脚手架: 单 CC 一路 Rollout 到底。

两种模式 (见 :mod:`seed`):
- ``cc_naive`` — initialize workspace 后一个 CC 直接 Rollout, 不做 branching;
- ``cc_with_expos`` — 额外 seed 一份 CLI 形式 ExpOS (experiment_os /
  SUBMIT_GUIDE.md / ``./expos`` launcher), 一个 CC 兼任三角色, submit 走
  serverless submit 训练管线 (``expos_cli`` → ``SubmitPipelineBackend`` →
  ``backend.backend.submit_pipeline``)。

Reuses ``backend/gpu_backend`` for GPU execution (localhost:8989, no ssh tunnel)
和 ``backend/backend/eval_all`` for eval。单个 Claude-Code agent 在私有
workspace 里探索; 结束条件 = agent 写 ``FINISH`` / 预算耗尽 / deadline 兜底。
"""

__version__ = "0.1.0"
