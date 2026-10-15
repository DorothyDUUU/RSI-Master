"""Reviewer-facing service — CLI backbone for a single reviewer's tools.

对应 redesign.md 中 Reviewer 的工具:
    reviewer.show_datapool()          (clone of worker.show_datapool)
    reviewer.inspect_data(data_id)    (clone of worker.inspect_data)
    reviewer.inspect_experiment(experiment_id)  (专属)
    reviewer.submit_review(...)       (由 CLI 层提供的落盘工具, 契约在这里)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import ReviewReport, dump_model
from .data_pool import DataPoolService
from .experiment_os import ExperimentOS
from .io_utils import read_json, write_json


class ReviewerService:
    """Bash-callable service scoped to a single reviewer."""

    def __init__(self, os_: ExperimentOS, *, data_pool: DataPoolService | None = None):
        self.os = os_
        self.data_pool = data_pool or DataPoolService(os_)

    # ── Data pool (identical to Worker) ──────────────────────────────────────

    def show_datapool(self) -> list[dict[str, Any]]:
        return self.data_pool.list_data()

    def inspect_data(self, data_id: str) -> dict[str, Any]:
        return self.data_pool.inspect_data(data_id)

    # ── Reviewer-only ────────────────────────────────────────────────────────

    def inspect_experiment(self, experiment_id: str) -> dict[str, Any]:
        """Parse everything under ``experiments/{experiment_id}/``.

        Returns:
            - node metadata (ExperimentNode)
            - the persisted WorkOrder + ExperimentResult
            - list of files under the experiment workspace
            - relevant event tail
        """
        graph = self.os.load_graph()
        exp = graph.experiments.get(experiment_id)
        if exp is None:
            raise KeyError(f"unknown experiment_id: {experiment_id}")

        # 审查前先把磁盘 result.json 同步回 DAG, reviewer 基于真实分数审查。
        self.os.sync_experiment_results()

        exp_dir = Path(exp.workspace)
        work_order = read_json(exp_dir / "work_order.json", default=None)
        result = read_json(exp_dir / "result.json", default=None)
        files: list[dict[str, Any]] = []
        for entry in sorted(exp_dir.rglob("*")):
            if not entry.is_file():
                continue
            files.append(
                {
                    "path": str(entry),
                    "relative": str(entry.relative_to(exp_dir)),
                    "size": entry.stat().st_size,
                }
            )
        events = self.os.events.filter(experiment_id=experiment_id, limit=200)
        return {
            "experiment": dump_model(exp),
            "work_order": work_order,
            "result": result,
            "files": files,
            "recent_events": events,
        }

    def inspect_reviewer(self, reviewer_id: str) -> dict[str, Any]:
        """Main-facing read of a reviewer: node status + where its reports live.

        Read-only — lets main-agent read a review's conclusions without guessing
        paths under ``reviews/<reviewer_id>/``. Returns the reviewer node, the
        per-experiment ``verdict_summary``, and the concrete paths of every
        report it has written (``reports/<eid>.json`` + ``REVIEW_REPORT.md``).
        """
        graph = self.os.load_graph()
        reviewer = graph.reviewers.get(reviewer_id)
        if reviewer is None:
            raise KeyError(f"unknown reviewer_id: {reviewer_id}")

        reviewer_dir = self.os.reviews_dir / reviewer_id
        reports_dir = reviewer_dir / "reports"
        reports: list[dict[str, Any]] = []
        if reports_dir.is_dir():
            for path in sorted(reports_dir.glob("*.json")):
                reports.append(
                    {
                        "experiment_id": path.stem,
                        "path": str(path),
                        "size": path.stat().st_size,
                    }
                )
        review_report = reviewer_dir / "REVIEW_REPORT.md"
        return {
            "reviewer_id": reviewer_id,
            "status": reviewer.status,
            "objective": reviewer.objective,
            "reviewed_experiment_ids": list(reviewer.reviewed_experiment_ids),
            "verdict_summary": dict(reviewer.verdict_summary),
            "reports": reports,
            "review_report_path": str(review_report) if review_report.is_file() else None,
            "review_report_exists": review_report.is_file(),
        }

    def submit_review(
        self,
        *,
        reviewer_id: str,
        experiment_id: str,
        report: ReviewReport,
    ) -> dict[str, Any]:
        """Persist a :class:`ReviewReport` to ``reviews/{reviewer_id}/reports/{eid}.json``.

        Also links the reviewer<->experiment edge in the DAG and, if the report
        includes a :class:`DataMemoryRecommendation` pointing at a data_id, records
        that recommendation on the data pool entry.
        """
        graph = self.os.load_graph()
        if reviewer_id not in graph.reviewers:
            raise KeyError(f"unknown reviewer_id: {reviewer_id}")
        if experiment_id not in graph.experiments:
            raise KeyError(f"unknown experiment_id: {experiment_id}")

        reviewer_dir = self.os.reviews_dir / reviewer_id
        reports_dir = reviewer_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / f"{experiment_id}.json"
        write_json(report_path, dump_model(report))

        # 链接 reviewer <-> experiment
        graph = self.os.load_graph()
        graph.link_reviewer_to_experiment(reviewer_id, experiment_id)
        state = self.os.load_state()
        self.os.commit_graph(graph, state)

        # 更新 reviewer 汇总
        reviewer = graph.reviewers[reviewer_id]
        summary = dict(reviewer.verdict_summary or {})
        summary[experiment_id] = {
            "verdict": report.verdict,
            "primary_bottleneck": report.primary_bottleneck,
            "confidence": report.confidence,
        }
        self.os.update_reviewer(reviewer_id, verdict_summary=summary, status="running")

        # 数据记忆推荐落到 data pool
        recommendation = report.data_memory_recommendation
        if recommendation and recommendation.target_data_id:
            try:
                self.data_pool.apply_memory_recommendation(
                    recommendation.target_data_id,
                    recommendation,
                    reviewer_id=reviewer_id,
                    review_verdict=report.verdict,
                )
            except KeyError:
                # data_id 已经消失, 记录事件即可
                self.os.events.emit(
                    "data_memory_recommendation_target_missing",
                    reviewer_id=reviewer_id,
                    data_id=recommendation.target_data_id,
                )

        self.os.events.emit(
            "review_registered",
            reviewer_id=reviewer_id,
            experiment_id=experiment_id,
            tool="reviewer.submit_review",
            payload={"verdict": report.verdict, "confidence": report.confidence},
        )
        return {
            "reviewer_id": reviewer_id,
            "experiment_id": experiment_id,
            "report_path": str(report_path),
            "verdict": report.verdict,
        }

    def finalize(self, reviewer_id: str, *, status: str = "finished") -> dict[str, Any]:
        self.os.update_reviewer(reviewer_id, status=status)
        return {"reviewer_id": reviewer_id, "status": status}
