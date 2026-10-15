"""Per-call LLM timing / token / retry JSONL log.

Append-only thin logging layer for the "每 agent 权威 LLM 时间线" hook
(traj_analysis README TODO T1 / T4 / T5).  Each LLM call — success or failure —
writes one line to ``trajectories/<agent>.llm_calls.jsonl`` with:

- ``ts_start`` / ``ts_end`` / ``latency_ms`` — authoritative request→response
  wall-clock, immune to evomaster.log Start/End misattribution under concurrent
  workers (T1 / T4).
- ``response_id`` — pairs each record back to the trajectory step's
  ``raw_response.id``.
- ``prompt_tokens`` / ``completion_tokens`` / ``reasoning_tokens`` — token
  usage; reasoning_tokens only when the provider reports them.
- ``finish_reason`` — stop reason (``stop`` / ``tool_calls`` / ``length`` ...).
- ``retries`` / ``attempts`` / ``error`` — retry & failure accounting (T5);
  a failed call is still logged with ``error`` set and the exception re-raised
  so the agent loop behaviour is unchanged.

Design invariants (traj_analysis README): append-only, thread-safe, no
interface change, backward compatible.  Writes are serialized per path with a
process-wide lock so concurrent worker threads never interleave lines.  The
consumer is the traj_analysis scanner — the file existing is optional (the
analysis marks N/A when absent).  A write failure must never break the agent
loop, so exceptions are swallowed here.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

# process-wide per-path locks so concurrent agent threads append atomically
_path_locks: dict[str, threading.Lock] = {}
_path_locks_guard = threading.Lock()


def _lock_for(path: str) -> threading.Lock:
    global _path_locks
    with _path_locks_guard:
        lock = _path_locks.get(path)
        if lock is None:
            lock = _path_locks[path] = threading.Lock()
        return lock


def record_llm_call(path: str, record: dict[str, Any]) -> None:
    """Append one JSON line to the per-agent ``llm_calls.jsonl``.

    Swallows all IO errors — logging must never break the agent loop.
    """
    try:
        lock = _lock_for(path)
        with lock:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
    except Exception:  # noqa: BLE001
        pass
