"""Graph model for the ExperimentOS DAG.

DAG 里三类节点:
    - :class:`~contracts.WorkerNode`     (worker)
    - :class:`~contracts.ExperimentNode` (experiment)
    - :class:`~contracts.ReviewerNode`   (reviewer)

拓扑约束:
    Worker (parent) --spawn--> Worker (child)
    Worker          --owns--> Experiment
    Worker          --owns--> Reviewer
    Reviewer        --audits--> Experiment (多对多)

这里只做纯内存的图算法 (祖先/后代/邻接) 与 dict 序列化, 落盘由 :mod:`experiment_os`
统一负责。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable

from .contracts import ExperimentNode, ReviewerNode, WorkerNode, dump_model


@dataclass
class WorkflowGraph:
    """In-memory representation of the workflow DAG.

    Persists to ``state/state.json`` under the ``graph`` key. Rebuild the graph
    from that dict via :meth:`from_state_dict`.
    """

    workers: dict[str, WorkerNode] = field(default_factory=dict)
    experiments: dict[str, ExperimentNode] = field(default_factory=dict)
    reviewers: dict[str, ReviewerNode] = field(default_factory=dict)
    root_worker_ids: list[str] = field(default_factory=list)

    # ── (De)serialization ────────────────────────────────────────────────────

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "workers": {wid: dump_model(node) for wid, node in self.workers.items()},
            "experiments": {eid: dump_model(node) for eid, node in self.experiments.items()},
            "reviewers": {rid: dump_model(node) for rid, node in self.reviewers.items()},
            "root_worker_ids": list(self.root_worker_ids),
        }

    @classmethod
    def from_state_dict(cls, data: dict[str, Any] | None) -> "WorkflowGraph":
        graph = cls()
        if not isinstance(data, dict):
            return graph
        for wid, payload in (data.get("workers") or {}).items():
            graph.workers[wid] = WorkerNode(**payload)
        for eid, payload in (data.get("experiments") or {}).items():
            graph.experiments[eid] = ExperimentNode(**payload)
        for rid, payload in (data.get("reviewers") or {}).items():
            graph.reviewers[rid] = ReviewerNode(**payload)
        graph.root_worker_ids = list(data.get("root_worker_ids") or [])
        return graph

    # ── Mutation ─────────────────────────────────────────────────────────────

    def add_worker(self, node: WorkerNode) -> WorkerNode:
        self.workers[node.worker_id] = node
        if node.parent_worker_id is None:
            if node.worker_id not in self.root_worker_ids:
                self.root_worker_ids.append(node.worker_id)
        else:
            parent = self.workers.get(node.parent_worker_id)
            if parent is not None and node.worker_id not in parent.child_worker_ids:
                parent.child_worker_ids.append(node.worker_id)
        return node

    def add_experiment(self, node: ExperimentNode) -> ExperimentNode:
        self.experiments[node.experiment_id] = node
        worker = self.workers.get(node.worker_id)
        if worker is not None and node.experiment_id not in worker.experiment_ids:
            worker.experiment_ids.append(node.experiment_id)
        return node

    def add_reviewer(self, node: ReviewerNode) -> ReviewerNode:
        self.reviewers[node.reviewer_id] = node
        worker = self.workers.get(node.worker_id)
        if worker is not None and node.reviewer_id not in worker.reviewer_ids:
            worker.reviewer_ids.append(node.reviewer_id)
        for exp_id in node.reviewed_experiment_ids:
            exp = self.experiments.get(exp_id)
            if exp is not None and node.reviewer_id not in exp.reviewer_ids:
                exp.reviewer_ids.append(node.reviewer_id)
        return node

    def link_reviewer_to_experiment(self, reviewer_id: str, experiment_id: str) -> None:
        reviewer = self.reviewers.get(reviewer_id)
        experiment = self.experiments.get(experiment_id)
        if reviewer is not None and experiment_id not in reviewer.reviewed_experiment_ids:
            reviewer.reviewed_experiment_ids.append(experiment_id)
        if experiment is not None and reviewer_id not in experiment.reviewer_ids:
            experiment.reviewer_ids.append(reviewer_id)

    # ── Query ────────────────────────────────────────────────────────────────

    def worker_ancestors(self, worker_id: str) -> list[str]:
        """Return ancestor worker ids from immediate parent up to a root."""
        chain: list[str] = []
        current = self.workers.get(worker_id)
        while current is not None and current.parent_worker_id:
            chain.append(current.parent_worker_id)
            current = self.workers.get(current.parent_worker_id)
        return chain

    def worker_descendants(self, worker_id: str) -> list[str]:
        """BFS descendants of a worker (excluding *worker_id*)."""
        seen: list[str] = []
        queue: deque[str] = deque([worker_id])
        while queue:
            current = queue.popleft()
            for child in self.workers.get(current, WorkerNode(worker_id=current)).child_worker_ids:
                if child in seen:
                    continue
                seen.append(child)
                queue.append(child)
        return seen

    def experiments_for_worker(self, worker_id: str) -> list[ExperimentNode]:
        worker = self.workers.get(worker_id)
        if worker is None:
            return []
        return [self.experiments[eid] for eid in worker.experiment_ids if eid in self.experiments]

    def reviewers_for_worker(self, worker_id: str) -> list[ReviewerNode]:
        worker = self.workers.get(worker_id)
        if worker is None:
            return []
        return [self.reviewers[rid] for rid in worker.reviewer_ids if rid in self.reviewers]

    def iter_workers(self) -> Iterable[WorkerNode]:
        return self.workers.values()
