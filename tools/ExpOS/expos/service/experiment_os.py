"""ExperimentOS v2 — file-backed OS with a graph-based workflow.

新版目录结构 (由 :meth:`ExperimentOS.initialize` 建立)::

    experiment_os/
      README.md
      TASK.md
      state/
        state.json                    # DAG 状态 + 全局摘要
        events.jsonl                  # 结构化事件流 (append-only)
      worker/{worker_id}/
        HISTORY.md
        TASK.md
        config.json
      experiments/{experiment_id}/
        config.json                   # WorkOrder + 状态
        result.json                   # 落盘的 ExperimentResult (submit 完成后)
        artifacts/                    # submit 落盘产物 (checkpoint/eval/inspect)
      reviews/{reviewer_id}/
        config.json                   # ReviewerNode meta
        reports/{experiment_id}.json  # 单条 ReviewReport
      data_pool/{data_id}/
        README.md
        config.json
        (原始数据 / 脚本 / 备份 ...)

主要不变量:
- 所有会改变 DAG 结构或节点状态的写操作都必须调用 :meth:`save_state` 与
  :meth:`_emit` (由 :class:`EventStream`), 让 CLI/审计有唯一事实源。
- 服务模块只通过本类的方法访问文件系统, 不直接读写。
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from .contracts import (
    ExperimentNode,
    ReviewerNode,
    WorkerNode,
    dump_model,
)
from .events import EventStream
from .graph import WorkflowGraph
from .io_utils import ensure_dir, read_json, write_json


SCHEMA_VERSION = "rsi_master.state.v1"


class ExperimentOS:
    """File-backed operating system for the rsi_master workflow."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        # Sub-directory anchors — the layout described in the module docstring.
        self.state_dir = self.root / "state"
        self.state_path = self.state_dir / "state.json"
        self.worker_dir = self.root / "worker"
        self.experiments_dir = self.root / "experiments"
        self.reviews_dir = self.root / "reviews"
        self.data_pool_dir = self.root / "data_pool"
        self.readme_path = self.root / "README.md"
        self.task_path = self.root / "TASK.md"

        self.events = EventStream(self.root)
        self._lock = threading.RLock()

    # ── Bootstrap ────────────────────────────────────────────────────────────

    def initialize(
        self,
        *,
        task_description: str,
        benchmark: str = "aime_2025",
        run_contract: dict[str, Any] | None = None,
        eval_all: dict[str, Any] | None = None,
        slime_config: dict[str, Any] | None = None,
        training_recommendations: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Create the on-disk skeleton and seed ``state.json``.

        Idempotent unless ``overwrite=True``.

        ``eval_all`` (非空时) 会随初始 state 一起落盘为 ``state["eval_all"]``,
        ``WorkerService.submit`` 读它并把 ``bench_id`` 转发给 SubmitBackend,
        让 stage-3 走 benchmark suite 统一评测。未启用时存 ``{}``, 行为与旧版一致。

        ``slime_config`` (非空时) 随初始 state 落盘为 ``state["slime_config"]``
        (slime_sft 的 launch_sft.sh env 覆盖项, model_dir / torch_dist_dir 必填)。
        ``WorkerService.submit`` 读它并合并 worker 侧 ``--slime-config`` 覆盖项。

        ``training_recommendations`` (非空时) 随初始 state 落盘为
        ``state["training_recommendations"]`` (config 的 ``expos_training_recommendations``
        段)。``WorkerService.submit`` 读它做模型档位匹配 + 默认配置补缺 + 预检规则
        (HARD_REFUSE / WARNING_REFUSE)。对 agent 不可见。
        """
        with self._lock:
            for path in (
                self.state_dir,
                self.worker_dir,
                self.experiments_dir,
                self.reviews_dir,
                self.data_pool_dir,
            ):
                ensure_dir(path)

            fresh = overwrite or not self.state_path.exists()
            if fresh:
                initial = {
                    "schema_version": SCHEMA_VERSION,
                    "task_description": task_description,
                    "benchmark": benchmark,
                    "eval_all": eval_all or {},
                    "slime_config": slime_config or {},
                    "training_recommendations": training_recommendations or {},
                    "run_contract": run_contract or {},
                    "created_at": time.time(),
                    "updated_at": time.time(),
                    "loop": {"status": "running", "stop_reason": None},
                    "counters": {"worker": 0, "experiment": 0, "reviewer": 0, "data": 0},
                    "current_worker_id": None,
                    "best": {"experiment_id": None, "score": None, "worker_id": None},
                    "data_pool_index": [],
                    "graph": WorkflowGraph().to_state_dict(),
                }
                self.save_state(initial)
                # events.jsonl already exists via EventStream.__init__
                (self.state_dir / "events.jsonl").touch(exist_ok=True)
                self._write_default_docs(task_description=task_description, benchmark=benchmark)
                self.events.emit(
                    "os_initialized",
                    payload={"task_description": task_description, "benchmark": benchmark},
                )
            return self.load_state()

    def _write_default_docs(self, *, task_description: str, benchmark: str) -> None:
        """Seed README.md / TASK.md if the user hasn't written their own."""
        if not self.readme_path.exists():
            self.readme_path.write_text(
                "# ExperimentOS Workspace\n\n"
                "This directory is managed by rsi_master ExperimentOS.\n"
                "See the top-level docs at `docs/` for the CLI and agent-loop reference.\n",
                encoding="utf-8",
            )
        if not self.task_path.exists():
            self.task_path.write_text(
                f"# Task\n\n{task_description}\n\nBenchmark: `{benchmark}`\n",
                encoding="utf-8",
            )
    # ── State I/O ────────────────────────────────────────────────────────────

    def load_state(self) -> dict[str, Any]:
        """Load ``state.json``. Returns an empty dict if missing."""
        return read_json(self.state_path, default={}) or {}

    def save_state(self, state: dict[str, Any]) -> None:
        """Persist ``state`` atomically after stamping ``updated_at``."""
        state = dict(state)
        state["updated_at"] = time.time()
        with self._lock:
            write_json(self.state_path, state)
            from .research_context import ResearchContext, enabled
            if enabled(state):
                ResearchContext(self).refresh_files(state)

    def load_graph(self) -> WorkflowGraph:
        """Rebuild the in-memory :class:`WorkflowGraph` from disk."""
        state = self.load_state()
        return WorkflowGraph.from_state_dict(state.get("graph"))

    def commit_graph(self, graph: WorkflowGraph, state: dict[str, Any] | None = None) -> dict[str, Any]:
        """Persist a mutated graph back into ``state.json``."""
        state = state if state is not None else self.load_state()
        state["graph"] = graph.to_state_dict()
        self.save_state(state)
        return state

    # ── ID allocation ────────────────────────────────────────────────────────

    def _next_id(self, prefix: str, counter_key: str) -> str:
        # 并发创建节点时, read-modify-write 必须互斥, 否则两个线程可能拿到同一个 id.
        with self._lock:
            state = self.load_state()
            counters = state.setdefault("counters", {})
            counters[counter_key] = int(counters.get(counter_key, 0)) + 1
            state["counters"] = counters
            self.save_state(state)
            return f"{prefix}_{counters[counter_key]:04d}"

    def new_worker_id(self) -> str:
        return self._next_id("worker", "worker")

    def new_experiment_id(self) -> str:
        return self._next_id("exp", "experiment")

    def new_reviewer_id(self) -> str:
        return self._next_id("rev", "reviewer")

    def new_data_id(self, prefix: str = "data") -> str:
        return self._next_id(prefix, "data")

    # ── Node registration (structural) ───────────────────────────────────────

    def register_worker(
        self,
        *,
        parent_worker_id: str | None,
        objective: str = "",
    ) -> WorkerNode:
        """Create a new worker node, its workspace directory, and seed files."""
        with self._lock:
            worker_id = self.new_worker_id()
            workspace = self.worker_dir / worker_id
            ensure_dir(workspace)

            node = WorkerNode(
                worker_id=worker_id,
                parent_worker_id=parent_worker_id,
                objective=objective,
                workspace=str(workspace),
                status="pending",
            )
            self._write_worker_seed_files(node)

            state = self.load_state()
            graph = WorkflowGraph.from_state_dict(state.get("graph"))
            graph.add_worker(node)

            state["current_worker_id"] = worker_id
            self.commit_graph(graph, state)
            self.events.emit(
                "worker_registered",
                worker_id=worker_id,
                payload={"parent_worker_id": parent_worker_id, "objective": objective},
            )
            return node

    def _write_worker_seed_files(self, node: WorkerNode) -> None:
        workspace = Path(node.workspace)
        history_path = workspace / "HISTORY.md"
        if not history_path.exists():
            history_path.write_text(
                f"# Worker `{node.worker_id}` — history log\n\n"
                "Record everything you tried, kept, and discarded here. "
                "Reviewers and future workers rely on this.\n",
                encoding="utf-8",
            )
        task_path = workspace / "TASK.md"
        if not task_path.exists():
            parent_context = ""
            if node.parent_worker_id:
                parent_history = self.worker_dir / node.parent_worker_id / "HISTORY.md"
                parent_context = (
                    f"Parent history: `{parent_history}`\n\n"
                    "Before starting, read the parent history and preserve its "
                    "established results, rejected directions, reusable assets, "
                    "and unresolved questions.\n\n"
                )
            task_path.write_text(
                f"# Worker `{node.worker_id}` — task briefing\n\n"
                f"Parent worker: `{node.parent_worker_id or 'ROOT'}`\n\n"
                f"{parent_context}"
                "Fill this file in immediately after activation with the concrete "
                "objective, hypothesis, and any constraints inherited from main.\n",
                encoding="utf-8",
            )
        config_path = workspace / "config.json"
        if not config_path.exists():
            write_json(config_path, dump_model(node))

    def update_worker(self, worker_id: str, **updates: Any) -> WorkerNode:
        """Merge ``updates`` into a worker node and persist."""
        with self._lock:
            state = self.load_state()
            graph = WorkflowGraph.from_state_dict(state.get("graph"))
            worker = graph.workers.get(worker_id)
            if worker is None:
                raise KeyError(f"unknown worker_id: {worker_id}")
            payload = dump_model(worker)
            payload.update(updates)
            payload["updated_at"] = time.time()
            new_node = WorkerNode(**payload)
            graph.workers[worker_id] = new_node
            self.commit_graph(graph, state)
            write_json(Path(new_node.workspace) / "config.json", dump_model(new_node))
            self.events.emit("worker_updated", worker_id=worker_id, payload=updates)
            return new_node

    def is_worker_terminal(self, worker_id: str) -> bool:
        """Return True if the worker node has reached a terminal status.

        Reads ``state.json`` directly (a fresh disk read per call, so it can be
        polled cheaply from a worker's on_step hook) and returns True for
        ``finished`` / ``failed`` / ``aborted``. Unknown workers are treated as
        not-yet-terminal (False).
        """
        state = self.load_state()
        graph = WorkflowGraph.from_state_dict(state.get("graph"))
        worker = graph.workers.get(worker_id)
        return worker is not None and worker.status in {"finished", "failed", "aborted"}

    def register_experiment(self, node: ExperimentNode) -> ExperimentNode:
        """Persist a newly-created experiment and its workspace."""
        with self._lock:
            workspace = self.experiments_dir / node.experiment_id
            ensure_dir(workspace)
            ensure_dir(workspace / "artifacts")
            payload = dump_model(node)
            payload["workspace"] = str(workspace)
            node = ExperimentNode(**payload)
            write_json(workspace / "config.json", payload)

            state = self.load_state()
            graph = WorkflowGraph.from_state_dict(state.get("graph"))
            graph.add_experiment(node)
            self.commit_graph(graph, state)
            self.events.emit(
                "experiment_registered",
                worker_id=node.worker_id,
                experiment_id=node.experiment_id,
                payload={"training_mode": node.training_mode, "benchmark": node.benchmark},
            )
            return node

    def update_experiment(self, experiment_id: str, **updates: Any) -> ExperimentNode:
        with self._lock:
            state = self.load_state()
            graph = WorkflowGraph.from_state_dict(state.get("graph"))
            exp = graph.experiments.get(experiment_id)
            if exp is None:
                raise KeyError(f"unknown experiment_id: {experiment_id}")
            payload = dump_model(exp)
            payload.update(updates)
            payload["updated_at"] = time.time()
            new_exp = ExperimentNode(**payload)
            graph.experiments[experiment_id] = new_exp
            # Roll up score into worker.best / global best.
            if new_exp.score is not None:
                worker = graph.workers.get(new_exp.worker_id)
                if worker is not None:
                    if worker.best_score is None or new_exp.score > (worker.best_score or float("-inf")):
                        worker.best_score = new_exp.score
                        worker.best_experiment_id = new_exp.experiment_id
                best = state.get("best") or {}
                if best.get("score") is None or new_exp.score > (best.get("score") or float("-inf")):
                    state["best"] = {
                        "experiment_id": new_exp.experiment_id,
                        "worker_id": new_exp.worker_id,
                        "score": new_exp.score,
                    }
            self.commit_graph(graph, state)
            write_json(Path(new_exp.workspace) / "config.json", dump_model(new_exp))
            self.events.emit(
                "experiment_updated",
                worker_id=new_exp.worker_id,
                experiment_id=new_exp.experiment_id,
                payload=updates,
            )
            return new_exp

    def sync_experiment_results(self, worker_id: str | None = None) -> list[str]:
        """Reconcile ``experiments/{id}/result.json`` on disk back into state + events.

        ``worker_submit`` is fully async: the pipeline (running in tmux via
        gpu_backend) overwrites ``result.json`` when it finishes, but nothing
        ever writes that terminal status/score back into ``state.json``. Workers
        used to hand-hack ``update_experiment`` through a bash python one-liner;
        the ones that didn't left **orphan experiments** (disk completed/failed,
        DAG still ``running``/``None``).

        This scans ``result.json`` and, for every experiment whose DAG status is
        not terminal but whose disk result IS terminal (``completed``/``failed``),
        calls :meth:`update_experiment` (status + score) and emits an
        ``experiment_synced`` event. Idempotent: unchanged experiments are skipped.

        Args:
            worker_id: if given, only sync experiments owned by this worker.

        Returns:
            The list of experiment_ids that were actually synced.
        """
        synced: list[str] = []
        graph = self.load_graph()
        for eid, node in graph.experiments.items():
            if worker_id and node.worker_id != worker_id:
                continue
            workspace = Path(node.workspace)
            result = read_json(workspace / "result.json", default=None)
            if not isinstance(result, dict):
                result = {}
            disk_status = str(result.get("status") or "")
            if disk_status not in {"completed", "failed", "aborted"}:
                # 分数同步缺口 (a): 手动重 eval 只写 ``eval/metrics.json``, 不覆盖
                # ``result.json`` -> 旧的 sync 扫不到, 分数停在 running/None. 当
                # result.json 尚未终结时, 把 eval/metrics.json 的 accuracy 作为兜底
                # 来源 (result.json 一旦终结仍是权威源, 不会被这个兜底覆盖).
                metrics = read_json(workspace / "eval" / "metrics.json", default=None)
                if isinstance(metrics, dict) and isinstance(
                    metrics.get("accuracy"), (int, float)
                ):
                    result = {
                        "status": "completed",
                        "score": float(metrics["accuracy"]),
                        "_source": "eval/metrics.json fallback",
                    }
                    disk_status = "completed"
                else:
                    continue
            node_status = {"completed": "finished", "failed": "failed", "aborted": "aborted"}[disk_status]
            disk_score = result.get("score")
            score = float(disk_score) if isinstance(disk_score, (int, float)) else None
            # result.json is authoritative: re-eval / re-run pipelines overwrite
            # it on disk, so sync whenever status OR score diverges from the DAG
            # (even for already-terminal experiments). Unchanged -> skipped.
            changed = node.status != node_status or (
                score is not None and node.score != score
            )
            if not changed:
                continue
            self.update_experiment(eid, status=node_status, score=score)
            self.events.emit(
                "experiment_synced",
                worker_id=node.worker_id,
                experiment_id=eid,
                payload={"disk_status": disk_status, "score": score},
            )
            synced.append(eid)
        return synced

    def register_reviewer(self, node: ReviewerNode) -> ReviewerNode:
        with self._lock:
            workspace = self.reviews_dir / node.reviewer_id
            ensure_dir(workspace)
            ensure_dir(workspace / "reports")
            payload = dump_model(node)
            payload["workspace"] = str(workspace)
            node = ReviewerNode(**payload)
            write_json(workspace / "config.json", payload)

            state = self.load_state()
            graph = WorkflowGraph.from_state_dict(state.get("graph"))
            graph.add_reviewer(node)
            self.commit_graph(graph, state)
            self.events.emit(
                "reviewer_registered",
                worker_id=node.worker_id,
                reviewer_id=node.reviewer_id,
                payload={"reviewed_experiment_ids": node.reviewed_experiment_ids},
            )
            return node

    def update_reviewer(self, reviewer_id: str, **updates: Any) -> ReviewerNode:
        with self._lock:
            state = self.load_state()
            graph = WorkflowGraph.from_state_dict(state.get("graph"))
            reviewer = graph.reviewers.get(reviewer_id)
            if reviewer is None:
                raise KeyError(f"unknown reviewer_id: {reviewer_id}")
            payload = dump_model(reviewer)
            payload.update(updates)
            payload["updated_at"] = time.time()
            new_reviewer = ReviewerNode(**payload)
            graph.reviewers[reviewer_id] = new_reviewer
            self.commit_graph(graph, state)
            write_json(Path(new_reviewer.workspace) / "config.json", dump_model(new_reviewer))
            self.events.emit("reviewer_updated", reviewer_id=reviewer_id, payload=updates)
            return new_reviewer

    # ── Convenience readers ──────────────────────────────────────────────────

    def get_current_worker_id(self) -> str | None:
        return self.load_state().get("current_worker_id")

    def set_current_worker_id(self, worker_id: str | None) -> None:
        with self._lock:
            state = self.load_state()
            state["current_worker_id"] = worker_id
            self.save_state(state)
            self.events.emit(
                "current_worker_changed",
                worker_id=worker_id,
                payload={"current_worker_id": worker_id},
            )

    def stop(self, reason: str) -> None:
        with self._lock:
            state = self.load_state()
            loop = state.setdefault("loop", {})
            loop["status"] = "stopped"
            loop["stop_reason"] = reason
            self.save_state(state)
            self.events.emit("loop_stopped", payload={"reason": reason})
