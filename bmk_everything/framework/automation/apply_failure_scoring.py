#!/usr/bin/env python3
"""Reclassify empty-content samples as wrong answers (model failures).

Directive: a truncated sample that produced no answer (``finish_reason=length``,
empty ``content``) is a **model failure** and must be scored as wrong — it enters
the accuracy denominator as 0 credit, it is not silently dropped from scoring.

- ``empty_content`` samples: ``correct=None -> False`` + ``scored_as_failure=True``
  (``response`` stays empty and ``incomplete_reason`` stays ``empty_content`` so
  the trajectory remains truthful about *why* there is no answer).
- ``request_error`` samples (transport / infra faults, not model failures): stay
  incomplete (``correct=None``) and are excluded from the denominator.

``reward.json`` is recomputed so the accuracy denominator = attempted − request_error
and the numerator is unchanged; empty-content failures now pull the score down.

Idempotent: rows already carrying ``scored_as_failure=True`` are not modified.

Usage::

    framework/.venv/bin/python framework/automation/apply_failure_scoring.py [--dry-run]

Rewrites ``data/output/<id>/<trial>/samples.jsonl`` and ``reward.json`` in place
for every scored benchmark (197 auto + 3 legacy).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
AUTO_TRIAL = "qwen36-20260808"
LEGACY_TRIAL = "qwen36-legacy-20260808"
LEGACY = {"gsm8k", "gpqa", "aime2025"}


def trial_for(bench_id: str) -> str:
    return LEGACY_TRIAL if bench_id in LEGACY else AUTO_TRIAL


def jsonl(path: Path) -> list[dict]:
    # Split on raw '\n' only — splitlines() treats \x85 (NEL) as a boundary, which
    # would fragment rows whose reasoning content contains that character.
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def read_scorers() -> dict[str, str]:
    scorers = {}
    for spec_path in sorted((REPO / "format").glob("*/spec.json")):
        try:
            spec = json.loads(spec_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        scorers[spec.get("benchmark-id")] = spec.get("scorer")
    for _id, scorer in (("gsm8k", "exact_match"), ("gpqa", "multiple_choice"),
                        ("aime2025", "numeric")):
        scorers.setdefault(_id, scorer)
    return scorers


def recompute_reward(bench_id: str, trial: str, rows: list[dict],
                     scorer: str, reward_path: Path) -> None:
    completed = sum(1 for r in rows if r.get("complete"))
    correct = sum(1 for r in rows if r.get("correct") is True)
    failures = sum(1 for r in rows if r.get("scored_as_failure"))
    incomplete = sum(1 for r in rows if r.get("correct") is None)
    attempted = len(rows)
    scored = attempted - incomplete
    rule = "empty_content scored as wrong (model failure); request_error excluded"

    prev = json.loads(reward_path.read_text()) if reward_path.exists() else {}
    meta = dict(prev.get("meta_data") or {})

    if bench_id in LEGACY:
        accuracy = correct / scored if scored else None
        reward = accuracy if incomplete == 0 else None
        meta.update({"attempted": attempted, "completed": completed,
                     "incomplete": incomplete, "correct": correct,
                     "failures": failures, "accuracy": accuracy,
                     "scoring_rule": rule})
    else:
        scoring_supported = scorer != "unscored"
        observed = correct / scored if scored and scoring_supported else None
        reward = observed if incomplete == 0 and scoring_supported else None
        meta.update({"attempted": attempted, "completed": completed,
                     "incomplete": incomplete, "correct": correct,
                     "failures": failures, "scored": scored,
                     "observed_reward": observed, "scoring_rule": rule})

    reward_path.write_text(json.dumps(
        {"reward": reward, "meta_data": meta}, ensure_ascii=False, indent=2) + "\n")
    print(f"[reward] {bench_id}: correct={correct}/{scored}  "
          f"failures={failures}  incomplete={incomplete}  reward={reward}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change without writing anything")
    args = parser.parse_args()

    scorers = read_scorers()
    total_failures = 0
    total_request_error = 0
    touched = 0
    for bench_id, scorer in sorted(scorers.items()):
        if scorer == "unscored":
            continue  # trajectory-only; no scalar reward to recompute
        trial = trial_for(bench_id)
        samples_path = REPO / f"data/output/{bench_id}/{trial}/samples.jsonl"
        if not samples_path.exists():
            continue
        rows = jsonl(samples_path)
        failures = request_error = changed = 0
        for row in rows:
            if str(row.get("response") or "").strip():
                continue
            reason = row.get("incomplete_reason")
            if reason == "empty_content":
                if row.get("correct") is None and not row.get("scored_as_failure"):
                    row["correct"] = False
                    row["scored_as_failure"] = True
                    changed += 1
                if row.get("scored_as_failure"):
                    failures += 1
            elif reason == "request_error":
                request_error += 1
        total_failures += failures
        total_request_error += request_error
        if not failures:
            continue
        touched += 1
        if args.dry_run:
            print(f"[dry-run] {bench_id}: {failures} empties -> wrong, "
                  f"{request_error} request_error stay incomplete")
            continue
        if changed:
            with samples_path.open("w", encoding="utf-8") as sink:
                for row in rows:
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
        recompute_reward(bench_id, trial, rows, scorer,
                         REPO / f"data/output/{bench_id}/{trial}/reward.json")

    print(f"\nTOTAL: empty_content reclassified as failures = {total_failures}  "
          f"(benches touched = {touched})   request_error kept incomplete = {total_request_error}",
          flush=True)


if __name__ == "__main__":
    main()
