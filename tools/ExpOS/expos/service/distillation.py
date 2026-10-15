"""Event-backed coordination for expensive data distillation jobs.

Workers own the local distillation recipe.  ExpOS only deduplicates the
expensive batch action, records the main-agent decision, and exposes compact
progress.  Nothing here launches a teacher request or a GPU job.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

from .data_pool import DataPoolService
from .experiment_os import ExperimentOS
from .io_utils import sha256_of_file


Decision = Literal["start", "reduce_size", "reject"]


class DistillationService:
    """Coordinate worker proposals without adding persistent state keys."""

    def __init__(self, os_: ExperimentOS, data_pool: DataPoolService):
        self.os = os_
        self.data_pool = data_pool

    def request(
        self,
        *,
        worker_id: str,
        source_data_id: str,
        teacher: str,
        prompt_path: str,
        target_rows: int,
        control_data_id: str | None = None,
        purpose: str = "",
        smoke_passed: int = 0,
        smoke_attempted: int = 0,
        estimated_seconds: int | None = None,
    ) -> dict[str, Any]:
        if target_rows <= 0:
            raise ValueError("target_rows must be positive")
        if smoke_attempted <= 0 or smoke_passed != smoke_attempted:
            raise ValueError("a successful smoke test is required before batch distillation")

        prompt = Path(prompt_path).expanduser().resolve()
        if not prompt.is_file():
            raise FileNotFoundError(f"prompt_path does not exist: {prompt}")
        source = self.data_pool.inspect_data(source_data_id)["record"]
        source_sha = str(source.get("sha256") or "")
        if not source_sha:
            artifact = Path(str(source.get("artifact_path") or ""))
            if not artifact.is_file():
                raise ValueError(f"source data {source_data_id} has no hashable artifact")
            source_sha = sha256_of_file(artifact)

        prompt_sha = sha256_of_file(prompt)
        fingerprint = hashlib.sha256(
            f"{source_sha}\0{teacher.strip()}\0{prompt_sha}".encode("utf-8")
        ).hexdigest()

        events = self.os.events.read_all()
        completed = self._latest(events, "distillation_completed", fingerprint)
        if completed:
            return {
                "decision": "reuse",
                "fingerprint": fingerprint,
                "artifact_id": completed.get("data_id"),
            }

        decided = self._latest(events, "distillation_decided", fingerprint)
        requested = self._latest(events, "distillation_requested", fingerprint)
        if decided:
            payload = decided.get("payload") or {}
            decision = str(payload.get("decision") or "")
            if decision in {"start", "reduce_size"}:
                claimed = self._latest(events, "distillation_claimed", fingerprint)
                if claimed and claimed.get("worker_id") != worker_id:
                    return {
                        "decision": "wait",
                        "fingerprint": fingerprint,
                        "owner_worker_id": claimed.get("worker_id"),
                        "reason": "an identical approved distillation is already running",
                    }
                if claimed is None:
                    self.os.events.emit(
                        "distillation_claimed",
                        worker_id=worker_id,
                        payload={"fingerprint": fingerprint},
                    )
                return {
                    "decision": decision,
                    "fingerprint": fingerprint,
                    "target_rows": payload.get("target_rows", target_rows),
                    "reason": payload.get("reason", ""),
                }
            if decision == "reject":
                return {
                    "decision": decision,
                    "fingerprint": fingerprint,
                    "target_rows": payload.get("target_rows", target_rows),
                    "reason": payload.get("reason", ""),
                }
        if requested:
            return {
                "decision": "wait",
                "fingerprint": fingerprint,
                "owner_worker_id": requested.get("worker_id"),
                "reason": "an identical distillation request is awaiting main-agent decision",
            }

        self.os.events.emit(
            "distillation_requested",
            worker_id=worker_id,
            payload={
                "fingerprint": fingerprint,
                "source_data_id": source_data_id,
                "teacher": teacher,
                "prompt_sha256": prompt_sha,
                "target_rows": target_rows,
                "control_data_id": control_data_id,
                "purpose": purpose,
                "smoke": {"passed": smoke_passed, "attempted": smoke_attempted},
                "estimated_seconds": estimated_seconds,
            },
        )
        return {
            "decision": "pending",
            "fingerprint": fingerprint,
            "instruction": (
                "Do not start the batch distillation. Finalize this worker so the "
                "controller can reactivate main-agent to decide the request."
            ),
        }

    def decide(
        self,
        *,
        fingerprint: str,
        decision: Decision,
        target_rows: int | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        events = self.os.events.read_all()
        request = self._latest(events, "distillation_requested", fingerprint)
        if not request:
            raise KeyError(f"unknown distillation fingerprint: {fingerprint}")
        requested_rows = int((request.get("payload") or {}).get("target_rows") or 0)
        if decision == "reduce_size":
            if target_rows is None or target_rows <= 0 or target_rows >= requested_rows:
                raise ValueError("reduce_size requires 0 < target_rows < requested target_rows")
        elif target_rows is not None:
            raise ValueError("target_rows is only valid for reduce_size")
        effective_rows = target_rows if decision == "reduce_size" else requested_rows
        self.os.events.emit(
            "distillation_decided",
            worker_id=request.get("worker_id"),
            payload={
                "fingerprint": fingerprint,
                "decision": decision,
                "target_rows": effective_rows,
                "reason": reason,
            },
        )
        return {
            "fingerprint": fingerprint,
            "decision": decision,
            "target_rows": effective_rows,
            "worker_id": request.get("worker_id"),
        }

    def progress(
        self,
        *,
        worker_id: str,
        fingerprint: str,
        completed: int,
        total: int,
        failed: int = 0,
    ) -> dict[str, Any]:
        if total <= 0 or completed < 0 or completed > total:
            raise ValueError("progress requires 0 <= completed <= total and total > 0")
        if failed < 0 or failed > completed:
            raise ValueError("failed must satisfy 0 <= failed <= completed")
        decision = self._latest(self.os.events.read_all(), "distillation_decided", fingerprint)
        if not decision or (decision.get("payload") or {}).get("decision") not in {
            "start",
            "reduce_size",
        }:
            raise ValueError("batch distillation has not been approved")
        self.os.events.emit(
            "artifact_progress",
            worker_id=worker_id,
            payload={
                "fingerprint": fingerprint,
                "stage": "distill",
                "completed": completed,
                "total": total,
                "failed": failed,
            },
        )
        return {
            "fingerprint": fingerprint,
            "completed": completed,
            "total": total,
            "failed": failed,
        }

    def complete(self, *, worker_id: str, fingerprint: str, data_id: str) -> dict[str, Any]:
        record = self.data_pool.inspect_data(data_id)["record"]
        sha = str(record.get("sha256") or "")
        if not sha:
            raise ValueError(f"registered data {data_id} has no sha256")
        self.os.events.emit(
            "distillation_completed",
            worker_id=worker_id,
            data_id=data_id,
            payload={"fingerprint": fingerprint, "sha256": sha},
        )
        return {"fingerprint": fingerprint, "data_id": data_id, "sha256": sha}

    @staticmethod
    def _latest(
        events: list[dict[str, Any]], event_type: str, fingerprint: str
    ) -> dict[str, Any] | None:
        for event in reversed(events):
            if event.get("type") != event_type:
                continue
            if (event.get("payload") or {}).get("fingerprint") == fingerprint:
                return event
        return None
