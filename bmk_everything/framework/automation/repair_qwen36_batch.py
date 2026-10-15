#!/usr/bin/env python3
"""Repair empty-content samples in a batch by re-running only those indices at higher max_tokens.

vllm reasoning models sometimes spend the entire ``max_tokens`` budget thinking and never emit a
final answer (``finish_reason=length``, empty ``content``).  This tool re-runs exactly those
indices through ``framework/run_eval.py --sample-indices`` into a separate repair trial, then merges
the recovered answers back into the original ``samples.jsonl`` and recomputes ``reward.json``.

Usage (repair override must raise ``max_tokens``/``timeout``)::

    framework/.venv/bin/python framework/automation/repair_qwen36_batch.py \
      --trial qwen36-20260808 --repair-trial qwen36-20260808-repair \
      --repair-override data/config-overrides/qwen36-repair.yaml \
      --openai-api-base https://.../v1 --openai-api-token dummy-noauth \
      --model Qwen3.6-35B-A3B --max-failures 5
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PYTHON = REPO / "framework/.venv/bin/python"
RUN_EVAL = REPO / "framework/run_eval.py"
SCORING_SUPPORTED = {"exact_match", "multiple_choice", "numeric", "boolean", "label",
                     "json", "squad"}


def jsonl(path: Path) -> list[dict]:
    # Split on raw '\n' only — splitlines() treats \x85 (NEL) and other Unicode
    # line-separators as boundaries, which fragments rows whose reasoning content
    # contains those characters (seen in bigbench-dyck-languages).
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def find_empty(bench_id: str, trial: str) -> list[int]:
    samples_path = REPO / f"data/output/{bench_id}/{trial}/samples.jsonl"
    if not samples_path.exists():
        return []
    return [int(row["index"]) for row in jsonl(samples_path)
            if not str(row.get("response") or "").strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", required=True)
    parser.add_argument("--repair-trial", required=True)
    parser.add_argument("--repair-override", required=True)
    parser.add_argument("--openai-api-base", required=True)
    parser.add_argument("--openai-api-token", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-failures", type=int, default=10)
    parser.add_argument("--ids", type=str, default=None,
                        help="comma-separated whitelist of benchmark ids to repair (default: all)")
    parser.add_argument("--plan-only", action="store_true",
                        help="print the repair plan without running any model")
    args = parser.parse_args()

    id_filter = {item.strip() for item in args.ids.split(",")} if args.ids else None
    jobs = []
    for spec_path in sorted((REPO / "format").glob("*/spec.json")):
        try:
            spec = json.loads(spec_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        bench_id = spec.get("benchmark-id")
        if not bench_id:
            continue
        if id_filter is not None and bench_id not in id_filter:
            continue
        indices = find_empty(bench_id, args.trial)
        if indices:
            jobs.append({"benchmark_id": bench_id, "scorer": spec.get("scorer"),
                         "indices": indices})
    total = sum(len(job["indices"]) for job in jobs)
    print(json.dumps({"benchmarks": len(jobs), "calls": total,
                      "repair_trial": args.repair_trial}, ensure_ascii=False), flush=True)
    if args.plan_only:
        for job in jobs:
            print(f"  {job['benchmark_id']}: {len(job['indices'])} indices")
        return

    failures = 0
    repaired_any = False
    for job in jobs:
        bench_id = job["benchmark_id"]
        indices = ",".join(str(i) for i in job["indices"])
        cmd = [str(PYTHON), str(RUN_EVAL), bench_id,
               "--openai-api-base", args.openai_api_base,
               "--openai-api-token", args.openai_api_token,
               "--model", args.model,
               "--sample-indices", indices,
               "--config-path", args.repair_override,
               "--trial-name", args.repair_trial]
        print(f"[repair] {bench_id} -> {job['indices']}", flush=True)
        process = subprocess.run(cmd, cwd=REPO)
        if process.returncode:
            failures += 1
            print(f"[repair] {bench_id} exit={process.returncode} (partial)", flush=True)
            if failures > args.max_failures:
                raise SystemExit("too many failures, aborting")
            # Fall through: merge whatever non-empty repairs were produced even if some
            # indices remain empty (the next sweep can retry those at a higher budget).

        # Merge repaired rows back into the original trial.
        repair_samples = REPO / f"data/output/{bench_id}/{args.repair_trial}/samples.jsonl"
        target_samples = REPO / f"data/output/{bench_id}/{args.trial}/samples.jsonl"
        if not repair_samples.exists():
            print(f"[repair] {bench_id}: no repair output, skipping merge", flush=True)
            continue
        rows = jsonl(target_samples)
        by_index = {int(row["index"]): row for row in jsonl(repair_samples)}
        merged = []
        for row in rows:
            replacement = by_index.get(int(row["index"]))
            if replacement and str(replacement.get("response") or "").strip():
                row = dict(row)
                for key in ("response", "reasoning_content", "finish_reason", "usage",
                            "complete", "incomplete_reason", "provider", "correct"):
                    if key in replacement:
                        row[key] = replacement[key]
                merged.append(row)
            else:
                merged.append(row)
        target_samples.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in merged) + "\n")

        # Recompute reward.json from the merged trajectory.
        scorer = job["scorer"]
        completed = sum(1 for r in merged if r.get("complete"))
        correct = sum(1 for r in merged if r.get("correct") is True)
        incomplete = len(merged) - completed
        scoring_supported = scorer != "unscored"
        observed = correct / completed if completed and scoring_supported else None
        reward = observed if incomplete == 0 and scoring_supported else None
        reward_path = REPO / f"data/output/{bench_id}/{args.trial}/reward.json"
        if reward_path.exists():
            prev = json.loads(reward_path.read_text())
            meta = prev.get("meta_data") or {}
            meta.update({"attempted": len(merged), "completed": completed,
                         "incomplete": incomplete, "correct": correct,
                         "observed_reward": observed, "repaired": True})
            reward_path.write_text(json.dumps(
                {"reward": reward, "meta_data": meta}, ensure_ascii=False, indent=2) + "\n")
        repaired_any = True
        print(f"[repair] {bench_id}: merged {len(job['indices'])} -> incomplete now {incomplete}", flush=True)

    print(f"[repair] done: {len(jobs)} benchmarks, {failures} failures", flush=True)


if __name__ == "__main__":
    main()
