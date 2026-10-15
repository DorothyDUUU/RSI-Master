"""SubmitPipelineBackend — cc_with_expos 注入的 submit backend。

实现 ``expos.service.worker_service.SubmitBackend`` 协议, 但 submit **不经过**
ExpOS FastAPI HTTP 服务 (:8990), 而是在进程内直接调用 repo 的
``backend.backend.submit_pipeline`` ``run_trial`` —— 与 ExpOS FastAPI server
用同一份 driver / 渲染 / 提交逻辑, 只是省掉 HTTP 一跳。

设计要点 (playground/cc_rollout/docs/expos_cli.md):
- 工具接口与当前 ExpOS 完全相同: ``expos.cli.main.main(argv, submit_backend=...)``
  原样复用子命令树, 一个 CC 通过 ``./expos`` CLI 兼任 MainAgent / Worker /
  Reviewer 三角色。
- 实现逻辑有变化: submit 走 serverless submit 训练管线 (run_trial → 渲染
  pipeline.sh → 提交 gpu_backend :8989 → 异步返回 ``status=running`` +
  ``gpu_script_id``)。WorkerService 侧的落盘 (work_order.json / 占位 result.json /
  ExperimentNode 注册 / 进程内去污染) 原样发生 —— 工具只做对应文件系统的修改,
  不影响任何 main-agent-loop。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class SubmitPipelineBackend:
    """Implements the :class:`SubmitBackend` Protocol against ``run_trial``.

    路径规范化 / 去污染 / 契约校验都由 WorkerService 在调用本 backend 之前完成;
    这里只把 submit kwargs 翻译成 :class:`SubmitRequest` 并进程内执行。

    ``need_exclusive_gpu`` 是协议之外的 kwarg (``WorkerService.submit`` 调用点
    总是传, 默认 True); ``slime_config`` 只在 slime_sft 时到达。多余的 ``**extra``
    兜底未来协议演进, 绝不因为调用点多传了字段而崩。
    """

    def submit(
        self,
        *,
        experiment_id: str,
        worker_id: str,
        benchmark: str,
        train_data_path: str,
        training_config_path: str,
        training_mode: str,
        experiment_workspace: Path,
        objective: str = "",
        hypothesis: str = "",
        changed_variable: str = "",
        parent_experiment_id: str | None = None,
        bench_id: str | None = None,
        eval_all: dict[str, Any] | None = None,
        gpu_nums: int = 1,
        slime_config: dict[str, Any] | None = None,
        allowed_training_modes: list[str] | None = None,
        need_exclusive_gpu: bool = True,
        **extra: Any,
    ) -> dict[str, Any]:
        # repo-root modules: ``backend/backend/submit_pipeline`` 需要 repo root 在
        # PYTHONPATH (expos launcher 已烘焙)。延迟导入, 让纯读 CLI 命令不依赖
        # gpu_backend 侧的 pydantic/driver 也能跑。
        from backend.backend.submit_pipeline.driver import run_trial
        from backend.backend.submit_pipeline.models import SubmitRequest

        request = SubmitRequest(
            experiment_id=experiment_id,
            worker_id=worker_id,
            benchmark=benchmark or "gsm8k",
            train_data_path=str(train_data_path),
            training_config_path=str(training_config_path),
            training_mode=training_mode,
            experiment_workspace=str(experiment_workspace),
            gpu_nums=int(gpu_nums),
            need_exclusive_gpu=bool(need_exclusive_gpu),
            objective=objective,
            hypothesis=hypothesis,
            changed_variable=changed_variable,
            parent_experiment_id=parent_experiment_id,
            bench_id=bench_id,
            eval_all=eval_all or {},
            slime_config=slime_config or {},
        )
        # `allowed_training_modes` 已在 WorkerService 侧生效 (run_contract gate),
        # 不需要传给 run_trial。
        return run_trial(request).to_dict()
