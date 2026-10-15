"""Checkpoint state for the BENCHMARK.md authoring pipeline.

Tracks which benchmark ids have a passing, failing, or pending BENCHMARK.md so the
wave-based agent pipeline can resume after an interruption.  Pure local IO.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
STATE_PATH = Path(__file__).resolve().parent / "md_state.json"


def _default_state() -> dict:
    return {"pass": [], "fail": {}, "in_progress": [], "updated": ""}


def load_state() -> dict:
    if STATE_PATH.is_file():
        try:
            state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = _default_state()
        for key in ("pass", "fail", "in_progress"):
            state.setdefault(key, [] if key != "fail" else {})
        return state
    return _default_state()


def save_state(state: dict) -> None:
    state["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")


def mark_pass(state: dict, benchmark_id: str) -> None:
    state["pass"].append(benchmark_id)
    state["pass"] = sorted(set(state["pass"]))
    state["fail"].pop(benchmark_id, None)
    state["in_progress"] = [i for i in state["in_progress"] if i != benchmark_id]
    save_state(state)


def mark_fail(state: dict, benchmark_id: str, reason: str) -> None:
    state["fail"][benchmark_id] = reason
    state["in_progress"] = [i for i in state["in_progress"] if i != benchmark_id]
    save_state(state)


def mark_start(state: dict, benchmark_id: str) -> None:
    state["in_progress"] = sorted(set(state["in_progress"]) | {benchmark_id})
    save_state(state)


def pending_ids(all_ids: list[str], state: dict) -> list[str]:
    done = set(state["pass"]) | set(state["fail"]) | set(state["in_progress"])
    return [benchmark_id for benchmark_id in all_ids if benchmark_id not in done]
