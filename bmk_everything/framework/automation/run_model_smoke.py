#!/usr/bin/env python3
"""Execute the explicitly approved, resumable 700-call model smoke plan."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approved-call-count", type=int)
    args = parser.parse_args()
    format_dir = args.format_dir.resolve()
    repo_dir = format_dir.parent
    ids = [item["benchmark-id"] for item in read_json(format_dir / "benchmark.json")]
    by_adapter: dict[str, list[str]] = defaultdict(list)
    for benchmark_id in ids:
        manifest = read_json(format_dir / benchmark_id / "config.json")
        by_adapter[manifest.get("adapter", "legacy_runner")].append(benchmark_id)

    jobs = [
        {"key": f"base:{benchmark_id}", "benchmark_id": benchmark_id, "limit": 3, "phase": "smoke_3"}
        for benchmark_id in ids
    ]
    for adapter, benchmark_ids in sorted(by_adapter.items()):
        jobs.append({"key": f"family:{adapter}", "benchmark_id": benchmark_ids[0],
                     "limit": 20, "phase": "representative_20", "adapter": adapter})
    planned_calls = sum(job["limit"] for job in jobs)
    plan = {"jobs": len(jobs), "planned_calls": planned_calls,
            "benchmarks": len(ids), "adapter_families": sorted(by_adapter)}
    print(json.dumps(plan, ensure_ascii=False), flush=True)
    if not args.execute:
        return
    if args.approved_call_count != planned_calls:
        raise SystemExit(f"approved call count {args.approved_call_count} != planned {planned_calls}")
    for name in ("KIMI_MOONSHOT_API_KEY", "KIMI_KSYUN_API_KEY"):
        if not os.environ.get(name):
            raise SystemExit(f"missing {name}")
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "state.jsonl"
    override_path = args.output / "smoke_override.yaml"
    override_path.write_text("custom:\n  max_workers: 3\n  max_tokens: 512\n", encoding="utf-8")
    completed = {}
    if state_path.exists():
        for line in state_path.read_text().splitlines():
            item = json.loads(line)
            if item.get("ok"):
                completed[item["key"]] = item
    pending = [job for job in jobs if job["key"] not in completed]
    print(json.dumps({"resumed": len(completed), "pending_jobs": len(pending)}, ensure_ascii=False), flush=True)

    def execute(job: dict) -> dict:
        safe_key = job["key"].replace(":", "-")
        trial = f"kimi27-{safe_key}"
        command = [str(repo_dir / "framework/.venv/bin/python"), str(repo_dir / "framework/run_eval.py"),
                   job["benchmark_id"], "--model", "kimi-k2.7-code",
                   "--trial-name", trial, "--limit", str(job["limit"]),
                   "--config-path", str(override_path)]
        started = time.time()
        process = subprocess.run(command, cwd=repo_dir, capture_output=True, text=True,
                                 timeout=1800, env=dict(os.environ))
        output_dir = repo_dir / "data/output" / job["benchmark_id"] / trial
        reward_path = output_dir / "reward.json"
        reward = read_json(reward_path) if reward_path.exists() else None
        completed_samples = ((reward or {}).get("meta_data") or {}).get("completed")
        ok = process.returncode == 0 and reward is not None and completed_samples == job["limit"]
        return {**job, "ok": ok, "returncode": process.returncode,
                "completed_samples": completed_samples, "reward": (reward or {}).get("reward"),
                "elapsed_s": round(time.time() - started, 3), "output_dir": str(output_dir),
                "stdout_tail": process.stdout[-2000:], "stderr_tail": process.stderr[-4000:]}

    failures = 0
    with state_path.open("a", encoding="utf-8") as state, \
            concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(execute, job): job for job in pending}
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {**job, "ok": False, "error": repr(exc)}
            state.write(json.dumps(result, ensure_ascii=False) + "\n")
            state.flush()
            failures += int(not result["ok"])
            print(f"[{'OK' if result['ok'] else 'FAIL'}] {job['key']} "
                  f"completed={result.get('completed_samples')} reward={result.get('reward')}", flush=True)
            # Stop scheduling is not available for already submitted futures, but a
            # systemic early failure threshold prevents continuing the full batch.
            if failures >= 5 and len(completed) + len(pending) > 20:
                for other in futures:
                    other.cancel()
                raise SystemExit("stopped after 5 failed jobs; inspect state.jsonl before resume")
    rows = [json.loads(line) for line in state_path.read_text().splitlines()]
    latest = {row["key"]: row for row in rows}
    summary = {"planned_calls": planned_calls, "jobs": len(jobs),
               "passed_jobs": sum(item.get("ok", False) for item in latest.values()),
               "failed_jobs": sum(not item.get("ok", False) for item in latest.values()),
               "completed_samples": sum(item.get("completed_samples") or 0 for item in latest.values())}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
