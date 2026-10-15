"""Structured event stream for ExperimentOS.

每一次 CLI 工具调用、每一次状态迁移都会以 :class:`OSEvent` 的 JSON 行形式追加
到 ``state/events.jsonl``。这是审计系统与 Main Agent 上下文重建的唯一事实源。
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from .contracts import OSEvent, dump_model
from .io_utils import append_jsonl, ensure_dir, iter_jsonl


class EventStream:
    """Append-only structured log persisted to ``state/events.jsonl``.

    All writes are serialized with a lock so concurrent CLI invocations don't
    interleave lines. Reads are cheap linear scans — good enough at the current
    scale (~thousands of events per run).
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.path = self.root / "state" / "events.jsonl"
        ensure_dir(self.path.parent)
        self._lock = threading.Lock()

    # ── Write ────────────────────────────────────────────────────────────────

    def emit(
        self,
        event_type: str,
        *,
        worker_id: str | None = None,
        experiment_id: str | None = None,
        reviewer_id: str | None = None,
        data_id: str | None = None,
        tool: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append a fully-typed event line and return the serialized form."""
        event = OSEvent(
            type=event_type,
            worker_id=worker_id,
            experiment_id=experiment_id,
            reviewer_id=reviewer_id,
            data_id=data_id,
            tool=tool,
            payload=payload or {},
        )
        record = dump_model(event)
        with self._lock:
            append_jsonl(self.path, record)
        return record

    # ── Read ─────────────────────────────────────────────────────────────────

    def read_all(self) -> list[dict[str, Any]]:
        """Return every event as a list of dicts (chronological order)."""
        return list(iter_jsonl(self.path))

    def filter(
        self,
        *,
        worker_id: str | None = None,
        experiment_id: str | None = None,
        reviewer_id: str | None = None,
        types: list[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return the tail of matching events (most recent last)."""
        want_types = set(types) if types else None
        matched: list[dict[str, Any]] = []
        for event in iter_jsonl(self.path):
            if worker_id is not None and event.get("worker_id") != worker_id:
                continue
            if experiment_id is not None and event.get("experiment_id") != experiment_id:
                continue
            if reviewer_id is not None and event.get("reviewer_id") != reviewer_id:
                continue
            if want_types is not None and event.get("type") not in want_types:
                continue
            matched.append(event)
        if limit is not None and limit >= 0:
            matched = matched[-limit:]
        return matched
