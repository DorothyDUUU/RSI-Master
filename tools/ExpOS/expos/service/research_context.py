"""Derived research overview and versioned Main-to-Worker handoffs.

No second experiment database: facts come from the graph and work orders.
Enabled per run through run_contract.research_handoff; old runs are unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
import uuid
from pathlib import Path

from .io_utils import read_json, write_json


def enabled(state):
    return bool((state.get("run_contract") or {}).get("research_handoff", False))


def _atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".research-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _cell(value, limit=180):
    return str(value or "—").replace("|", "\\|").replace("\n", " ")[:limit]


class ResearchContext:
    def __init__(self, os_):
        self.os = os_
        self.root = os_.root

    def evidence(self, state):
        """Only substantive outcomes trigger a submit refresh, not polling timestamps."""
        graph = state.get("graph", {})
        result = {}
        for eid, exp in graph.get("experiments", {}).items():
            if exp.get("status") in {"completed", "finished", "failed", "aborted"}:
                result[eid] = {"status": exp.get("status"), "score": exp.get("score")}
        for rid, rev in graph.get("reviewers", {}).items():
            if rev.get("status") in {"completed", "finished", "failed", "aborted"}:
                report = self.root / "reviews" / rid / "REVIEW_REPORT.md"
                result[rid] = {"status": rev.get("status"), "verdict": rev.get("verdict_summary"),
                               "report_hash": hashlib.sha256(report.read_bytes()).hexdigest() if report.exists() else None}
        return result

    def revision(self, state):
        return hashlib.sha256(json.dumps(self.evidence(state), sort_keys=True).encode()).hexdigest()[:16]

    def render(self, state, *, limit=None, references=()):
        graph = state.get("graph", {})
        workers = graph.get("workers", {})
        experiments = graph.get("experiments", {})
        rows = sorted(experiments.items(), key=lambda item: item[1].get("created_at", 0))
        if limit is not None and len(rows) > limit:
            best = (state.get("best") or {}).get("experiment_id")
            preferred = set(references) | {best}
            preferred.update(eid for eid, e in rows if e.get("status") in {"pending", "queued", "running"})
            ranked = sorted(rows, key=lambda item: (item[0] in preferred, item[1].get("created_at", 0)), reverse=True)
            chosen = {eid for eid, _ in ranked[:limit]}
            rows = [(eid, e) for eid, e in rows if eid in chosen]
        text = ["# Research overview (generated facts)",
                f"Benchmark: {state.get('benchmark')}; evidence revision: {self.revision(state)}.",
                "| Direction / Worker | Experiment / reference | Purpose / change | Status / score |",
                "|---|---|---|---|"]
        for eid, exp in rows:
            order = read_json(self.root / "experiments" / eid / "work_order.json", default={}) or {}
            wid = exp.get("worker_id", "")
            purpose = order.get("hypothesis") or order.get("objective") or "Not recorded"
            change = order.get("changed_variable")
            if change:
                purpose += "; change: " + change
            score = exp.get("score")
            text.append(f"| {_cell(workers.get(wid, {}).get('objective'), 100)} / {wid} | {eid} / {_cell(exp.get('parent_experiment_id'))} | {_cell(purpose)} | {_cell(exp.get('status'))} / {score if score is not None else '—'} |")
        if not rows:
            text.append("No experiments registered yet.")
        if len(rows) < len(experiments):
            text.append(f"Showing {len(rows)}/{len(experiments)} experiments. Full table: experiment_os/RESEARCH_OVERVIEW.md.")
        preparing = [(wid, w) for wid, w in workers.items() if not w.get("experiment_ids")]
        if preparing:
            text.append("\nDirections without experiments:")
            for wid, w in preparing[-12:]:
                text.append(f"- {wid} ({w.get('status')}): {_cell(w.get('objective'), 160)}")
        text.append("\nScores are recorded values under this run's protocol, not proof of causality or evaluation validity.")
        return "\n".join(text) + "\n"

    def refresh_files(self, state):
        _atomic_text(self.root / "RESEARCH_OVERVIEW.md", self.render(state))
        judgment = state.get("research_judgment") or {}
        _atomic_text(self.root / "RESEARCH_STATE.md", self.judgment_text(state, judgment))

    def judgment_text(self, state, judgment=None):
        judgment = judgment if judgment is not None else state.get("research_judgment") or {}
        stale = judgment.get("evidence_revision") != self.revision(state)
        return (f"# Main research judgment (interpretation, not verified fact)\n"
                f"Version: {judgment.get('version', 0)}; updated_at: {judgment.get('updated_at', 'never')}; "
                f"evidence_revision: {judgment.get('evidence_revision', 'none')}.\n"
                + ("New outcomes may not yet be incorporated.\n" if stale else "")
                + judgment.get("text", "Main has not recorded a research judgment yet.") + "\n")

    def view(self, objective=""):
        with self.os._lock:
            state = self.os.load_state()
            self.refresh_files(state)
            refs = re.findall(r"exp_\d+", objective)
            return {"evidence_revision": self.revision(state),
                    "context": self.render(state, limit=24, references=refs) + "\n" + self.judgment_text(state)}

    def update(self, text, evidence_revision):
        if not text.strip() or len(text) > 6000:
            raise ValueError("Provide 1–6000 characters of research judgment.")
        with self.os._lock:
            state = self.os.load_state()
            if evidence_revision != self.revision(state):
                return {"status": "refresh_required", **self.view()}
            old = state.get("research_judgment") or {}
            judgment = {"version": old.get("version", 0) + 1, "updated_at": time.time(),
                        "evidence_revision": evidence_revision, "text": text}
            state["research_judgment"] = judgment
            self.os.save_state(state)
            write_json(self.os.state_dir / "research_history" / f"{judgment['version']:06d}.json", judgment)
            self.os.events.emit("research_judgment_updated", payload=judgment)
            return {"status": "updated", **judgment}

    def handoff(self, worker_id, objective=""):
        with self.os._lock:
            state = self.os.load_state()
            if not enabled(state):
                return ""
            if worker_id not in state.get("graph", {}).get("workers", {}):
                raise KeyError(worker_id)
            text = self.view(objective)["context"]
            text += ("\n# Your assignment\n" + objective + "\n"
                     "Use experiment_inspect with relevant experiment IDs before designing a follow-up. "
                     "Do not treat Main's hypotheses as established facts. Report findings in your own HISTORY.md; "
                     "Main maintains RESEARCH_STATE.md.\n")
            snapshot = {"time": time.time(), "context": text, "evidence": self.evidence(state),
                        "judgment_version": (state.get("research_judgment") or {}).get("version", 0)}
            directory = self.os.worker_dir / worker_id / "research_handoffs"
            write_json(directory / f"{time.time_ns()}-{uuid.uuid4().hex[:8]}.json", snapshot)
            write_json(directory / "last_seen.json", snapshot)
            return text

    def before_submit(self, worker_id):
        """Return new evidence once, without allocating an experiment or GPU task."""
        with self.os._lock:
            state = self.os.load_state()
            if not enabled(state):
                return None
            path = self.os.worker_dir / worker_id / "research_handoffs" / "last_seen.json"
            seen = read_json(path, default={}) or {}
            current = self.evidence(state)
            version = (state.get("research_judgment") or {}).get("version", 0)
            changed = sorted(key for key in set(current) | set(seen.get("evidence", {}))
                             if current.get(key) != seen.get("evidence", {}).get(key))
            if seen and not changed and version == seen.get("judgment_version"):
                return None
            objective = state.get("graph", {}).get("workers", {}).get(worker_id, {}).get("objective", "")
            context = self.handoff(worker_id, objective)
            return {"status": "deferred", "reason": "research_context_updated", "changed_ids": changed,
                    "context": context, "hint": "No experiment or GPU task allocated. Read the new evidence, inspect relevant IDs, then reconsider or repeat your submission."}

    def inspect(self, experiment_id):
        state = self.os.load_state()
        exp = state.get("graph", {}).get("experiments", {}).get(experiment_id)
        if exp is None:
            raise KeyError(f"Unknown experiment: {experiment_id}")
        directory = self.os.experiments_dir / experiment_id
        order = read_json(directory / "work_order.json", default={}) or {}
        return {"experiment": exp, "work_order": order,
                "result": read_json(directory / "result.json", default={}),
                "config_path": str(directory / "config.json"),
                "training_config_path": order.get("training_config_path"),
                "review_paths": [str(self.os.reviews_dir / rid / "REVIEW_REPORT.md") for rid in exp.get("reviewer_ids", [])]}
