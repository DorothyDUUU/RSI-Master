"""State-level service — CLI backbone for global DAG operations.

对应 CLI 命令:
    state add-new-worker --parent-worker-id ...
    state get-current-worker
    state inspect-worker --worker-id ...

设计原则:
    - 只在 experiment_os 上做**结构化调用**, 不直接读写文件
    - 所有返回都是纯 JSON-safe dict, CLI 层负责渲染
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import dump_model
from .data_pool import DataPoolService
from .experiment_os import ExperimentOS


class StateService:
    """CLI-facing service for the global workflow DAG (state.json)."""

    def __init__(self, os_: ExperimentOS):
        self.os = os_

    # ── state.add_new_worker(parent_worker_id) ────────────────────────────────

    def add_new_worker(
        self,
        *,
        parent_worker_id: str | None = None,
        objective: str = "",
    ) -> dict[str, Any]:
        """Instantiate a new worker under the given parent.

        Returns a dict containing the new ``worker_id``, its workspace path, and
        a hint reminding the agent to update HISTORY.md / TASK.md.
        """
        state = self.os.load_state()
        graph = self.os.load_graph()
        # 容错: 部分 LLM 会把 null 序列化成字符串 "null"/"None"/"" 传进来,
        # 统一归一化成 None (视为 root worker), 避免误判为 unknown parent.
        if isinstance(parent_worker_id, str) and parent_worker_id.strip().lower() in {"null", "none", ""}:
            parent_worker_id = None
        # 校验 parent 存在, 否则视为 root
        if parent_worker_id is not None and parent_worker_id not in graph.workers:
            raise ValueError(f"unknown parent_worker_id: {parent_worker_id}")

        node = self.os.register_worker(parent_worker_id=parent_worker_id, objective=objective)

        workspace = Path(node.workspace)
        return {
            "worker_id": node.worker_id,
            "parent_worker_id": node.parent_worker_id,
            "workspace": str(workspace),
            "history_path": str(workspace / "HISTORY.md"),
            "task_path": str(workspace / "TASK.md"),
            "hint": (
                "Update HISTORY.md and TASK.md with the concrete objective and "
                "constraints inherited from main-agent before starting work."
            ),
            "run_contract": state.get("run_contract", {}),
        }

    # ── state.get_current_worker() ────────────────────────────────────────────

    def get_current_worker(self) -> dict[str, Any]:
        """Return an overview of the DAG in the shape asked for by redesign.md.

        Contains:
            - current_worker_id
            - loop status
            - each worker's id / parent / status / best_score / experiments / reviewers
        """
        state = self.os.load_state()
        graph = self.os.load_graph()
        workers_summary = []
        for wid, worker in graph.workers.items():
            workers_summary.append(
                {
                    "worker_id": wid,
                    "parent_worker_id": worker.parent_worker_id,
                    "status": worker.status,
                    "objective": worker.objective,
                    "best_score": worker.best_score,
                    "best_experiment_id": worker.best_experiment_id,
                    "experiment_ids": list(worker.experiment_ids),
                    "reviewer_ids": list(worker.reviewer_ids),
                    "child_worker_ids": list(worker.child_worker_ids),
                }
            )
        return {
            "current_worker_id": state.get("current_worker_id"),
            "loop": state.get("loop", {}),
            "task_description": state.get("task_description", ""),
            "benchmark": state.get("benchmark", ""),
            "best": state.get("best", {}),
            "counters": state.get("counters", {}),
            "root_worker_ids": graph.root_worker_ids,
            "workers": workers_summary,
        }

    # ── state.inspect_worker(worker_id) ───────────────────────────────────────

    def inspect_worker(self, worker_id: str, *, event_limit: int = 200) -> dict[str, Any]:
        """Return everything that happened for ``worker_id``.

        Includes: node metadata, owned experiment nodes, owned reviewer nodes,
        and the tail of matching entries from ``events.jsonl``.
        """
        graph = self.os.load_graph()
        worker = graph.workers.get(worker_id)
        if worker is None:
            raise KeyError(f"unknown worker_id: {worker_id}")

        # 读之前先把磁盘上已完成实验同步回 DAG, 避免看到 stale 的 running/None。
        self.os.sync_experiment_results(worker_id=worker_id)

        experiments = [dump_model(exp) for exp in graph.experiments_for_worker(worker_id)]
        reviewers = [dump_model(rev) for rev in graph.reviewers_for_worker(worker_id)]
        events = self.os.events.filter(worker_id=worker_id, limit=event_limit)
        return {
            "worker": dump_model(worker),
            "experiments": experiments,
            "reviewers": reviewers,
            "recent_events": events,
        }

    def sync(self, *, worker_id: str | None = None) -> dict[str, Any]:
        """Reconcile disk results back into state + events (experiments + data pool).

        - experiments: scan ``experiments/{id}/result.json`` and sync any terminal
          result the DAG never learned about (fixes orphan experiments).
        - data pool: rebuild ``data_pool_index`` + global ``data.jsonl`` from disk.

        Returns which experiments were synced and the resulting data pool ids.
        """
        synced_experiments = self.os.sync_experiment_results(worker_id=worker_id)
        data_entries = DataPoolService(self.os).reconcile()
        return {
            "synced_experiments": synced_experiments,
            "data_pool_entries": data_entries,
        }
