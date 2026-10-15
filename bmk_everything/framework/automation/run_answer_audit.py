#!/usr/bin/env python3
"""Run the explicitly approved 10-answer-per-benchmark Kimi audit batch."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import time
from pathlib import Path


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def read_state(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().split("\n") if line]


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
    jobs = [{"key": benchmark_id, "benchmark_id": benchmark_id, "limit": 10} for benchmark_id in ids]
    planned_calls = sum(job["limit"] for job in jobs)
    print(json.dumps({"benchmarks": len(ids), "jobs": len(jobs),
                      "samples_per_benchmark": 10, "planned_calls": planned_calls}), flush=True)
    if not args.execute:
        return
    if args.approved_call_count != planned_calls:
        raise SystemExit(f"approved call count {args.approved_call_count} != planned {planned_calls}")
    for name in ("KIMI_MOONSHOT_API_KEY", "KIMI_KSYUN_API_KEY"):
        if not os.environ.get(name):
            raise SystemExit(f"missing {name}")

    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "state.jsonl"
    override_path = args.output / "audit_override.yaml"
    if not override_path.exists():
        override_path.write_text(
            "custom:\n  max_workers: 3\n  max_tokens: 8192\n",
            encoding="utf-8",
        )
    rows = read_state(state_path)
    latest = {row["key"]: row for row in rows}
    completed = {key: row for key, row in latest.items() if row.get("ok")}
    attempts = {key: sum(row["key"] == key for row in rows) for key in ids}
    pending = [job for job in jobs if job["key"] not in completed]
    print(json.dumps({"resumed": len(completed), "pending_jobs": len(pending)}), flush=True)

    def execute(job: dict) -> dict:
        attempt = attempts[job["key"]] + 1
        trial = f"kimi27-audit10-a{attempt}"
        command = [
            str(repo_dir / "framework/.venv/bin/python"), str(repo_dir / "framework/run_eval.py"),
            job["benchmark_id"], "--model", "kimi-k2.7-code",
            "--trial-name", trial, "--limit", str(job["limit"]),
            "--config-path", str(override_path),
        ]
        started = time.time()
        process = subprocess.run(
            command, cwd=repo_dir, capture_output=True, text=True,
            timeout=3600, env=dict(os.environ),
        )
        output_dir = repo_dir / "data/output" / job["benchmark_id"] / trial
        reward_path = output_dir / "reward.json"
        reward = read_json(reward_path) if reward_path.exists() else None
        completed_samples = ((reward or {}).get("meta_data") or {}).get("completed")
        ok = process.returncode == 0 and reward is not None and completed_samples == job["limit"]
        return {
            **job, "attempt": attempt, "ok": ok, "returncode": process.returncode,
            "completed_samples": completed_samples, "reward": (reward or {}).get("reward"),
            "elapsed_s": round(time.time() - started, 3), "output_dir": str(output_dir),
            "stdout_tail": process.stdout[-2000:], "stderr_tail": process.stderr[-4000:],
        }

    failures = 0
    iterator = iter(pending)
    with state_path.open("a", encoding="utf-8") as state, \
            concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures: dict[concurrent.futures.Future, dict] = {}

        def submit_next() -> bool:
            try:
                job = next(iterator)
            except StopIteration:
                return False
            futures[pool.submit(execute, job)] = job
            return True

        for _ in range(min(args.workers, len(pending))):
            submit_next()
        stop = False
        while futures:
            done, _ = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                job = futures.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = {**job, "attempt": attempts[job["key"]] + 1,
                              "ok": False, "error": repr(exc)}
                state.write(json.dumps(result, ensure_ascii=False) + "\n")
                state.flush()
                failures += int(not result["ok"])
                print(f"[{'OK' if result['ok'] else 'FAIL'}] {job['key']} "
                      f"completed={result.get('completed_samples')} reward={result.get('reward')}",
                      flush=True)
                if failures >= 5:
                    stop = True
                if not stop:
                    submit_next()
            # Once the threshold is reached, stop submitting new jobs but keep
            # collecting the at-most-workers jobs that were already running.
    if failures >= 5:
        raise SystemExit("stopped after 5 failed jobs; inspect state.jsonl before resume")

    latest = {row["key"]: row for row in read_state(state_path)}
    summary = {
        "planned_calls": planned_calls,
        "jobs": len(jobs),
        "passed_jobs": sum(bool(item.get("ok")) for item in latest.values()),
        "failed_jobs": sum(not item.get("ok", False) for item in latest.values()),
        "completed_samples": sum(item.get("completed_samples") or 0 for item in latest.values()),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary), flush=True)
    if summary["failed_jobs"]:
        raise SystemExit(f"{summary['failed_jobs']} jobs remain failed; resume to retry")


if __name__ == "__main__":
    main()
