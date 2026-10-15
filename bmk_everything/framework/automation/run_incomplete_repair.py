#!/usr/bin/env python3
"""Execute an explicitly approved, index-targeted incomplete-answer repair plan."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import time
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    # JSON strings may legally contain control characters such as U+0085.  Only
    # the physical LF delimiter separates JSONL records; str.splitlines() would
    # incorrectly split inside such a string.
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


def validate_plan(plan: dict, approved: int) -> list[dict]:
    jobs = plan.get("jobs") or []
    calls = sum(len(job.get("indices") or []) for job in jobs)
    if plan.get("execute") is not False:
        raise ValueError("repair plan must be the reviewed execute=false planning artifact")
    if plan.get("planned_calls") != calls or approved != calls:
        raise ValueError(f"approved/planned/expanded calls differ: {approved}/{plan.get('planned_calls')}/{calls}")
    if len({job["benchmark_id"] for job in jobs}) != len(jobs):
        raise ValueError("repair plan contains duplicate benchmark jobs")
    for job in jobs:
        indices = job.get("indices") or []
        if len(indices) != len(set(indices)) or any(not isinstance(index, int) or index < 0 for index in indices):
            raise ValueError(f"invalid indices for {job.get('benchmark_id')}")
        source = {int(row["index"]): row for row in jsonl(Path(job["source_output_dir"]) / "samples.jsonl")}
        invalid = [index for index in indices if index not in source or str(source[index].get("response") or "").strip()]
        if invalid:
            raise ValueError(f"source rows are missing or no longer empty for {job['benchmark_id']}: {invalid}")
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--trial-name", default="kimi27-targeted-repair")
    parser.add_argument("--max-failures", type=int, default=5)
    parser.add_argument("--max-new-jobs", type=int,
                        help="Run only the first N pending jobs, for a compatibility canary")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approved-call-count", type=int, required=True)
    args = parser.parse_args()

    plan = read_json(args.plan.resolve())
    jobs = validate_plan(plan, args.approved_call_count)
    print(json.dumps({"benchmarks": len(jobs), "planned_calls": plan["planned_calls"],
                      "max_tokens": plan["max_tokens"], "execute": args.execute}), flush=True)
    if not args.execute:
        return
    for name in ("KIMI_MOONSHOT_API_KEY", "KIMI_KSYUN_API_KEY"):
        if not os.environ.get(name):
            raise SystemExit(f"missing {name}")

    format_dir = args.format_dir.resolve()
    repo_dir = format_dir.parent
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "state.jsonl"
    override_path = args.output / "repair_override.yaml"
    if not override_path.exists():
        override_path.write_text(
            f"custom:\n  max_workers: 10\n  max_tokens: {int(plan['max_tokens'])}\n  timeout: 600\n",
            encoding="utf-8",
        )
    previous = jsonl(state_path)
    attempted_keys = {row["key"] for row in previous}
    attempted_calls = sum(int(row.get("calls") or 0) for row in previous)
    pending = [job for job in jobs if job["benchmark_id"] not in attempted_keys]
    if args.max_new_jobs is not None:
        if args.max_new_jobs < 1:
            raise SystemExit("--max-new-jobs must be positive")
        pending = pending[:args.max_new_jobs]
    pending_calls = sum(len(job["indices"]) for job in pending)
    if attempted_calls + pending_calls > args.approved_call_count:
        raise SystemExit("state plus pending work exceeds approved call count")
    print(json.dumps({"previous_jobs": len(attempted_keys), "pending_jobs": len(pending),
                      "pending_calls": pending_calls}), flush=True)

    def execute(job: dict) -> dict:
        benchmark_id = job["benchmark_id"]
        trial = args.trial_name
        command = [
            str(repo_dir / "framework/.venv/bin/python"), str(repo_dir / "framework/run_eval.py"),
            benchmark_id, "--model", "kimi-k2.7-code", "--trial-name", trial,
            "--config-path", str(override_path),
            "--sample-indices", ",".join(str(index) for index in job["indices"]),
        ]
        started = time.time()
        child_env = dict(os.environ)
        child_env["KIMI_PROVIDER_RETRIES"] = "1"
        child_env.pop("KIMI_THINKING_TYPE", None)
        child_env["KIMI_FINAL_ONLY"] = "1"
        process = subprocess.run(command, cwd=repo_dir, capture_output=True, text=True,
                                 timeout=3600, env=child_env)
        output_dir = repo_dir / "data/output" / benchmark_id / trial
        reward_path = output_dir / "reward.json"
        reward = read_json(reward_path) if reward_path.exists() else None
        samples_path = output_dir / "samples.jsonl"
        samples = jsonl(samples_path) if samples_path.exists() else []
        actual_indices = sorted(int(row["index"]) for row in samples)
        expected_indices = sorted(job["indices"])
        completed = ((reward or {}).get("meta_data") or {}).get("completed")
        ok = (process.returncode == 0 and completed == len(expected_indices)
              and actual_indices == expected_indices
              and all(str(row.get("response") or "").strip() for row in samples))
        return {
            "key": benchmark_id, "calls": len(expected_indices), "indices": expected_indices,
            "ok": ok, "returncode": process.returncode, "completed_samples": completed,
            "reward": (reward or {}).get("reward"), "elapsed_s": round(time.time() - started, 3),
            "output_dir": str(output_dir), "source_output_dir": job["source_output_dir"],
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
            done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                job = futures.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"key": job["benchmark_id"], "calls": len(job["indices"]),
                              "indices": job["indices"], "ok": False, "error": repr(exc)}
                state.write(json.dumps(result, ensure_ascii=False) + "\n")
                state.flush()
                failures += int(not result["ok"])
                print(f"[{'OK' if result['ok'] else 'FAIL'}] {result['key']} "
                      f"calls={result['calls']} completed={result.get('completed_samples')}", flush=True)
                if failures >= args.max_failures:
                    stop = True
                if not stop:
                    submit_next()
    rows = jsonl(state_path)
    summary = {
        "approved_calls": args.approved_call_count,
        "attempted_calls": sum(int(row.get("calls") or 0) for row in rows),
        "attempted_jobs": len(rows),
        "passed_jobs": sum(bool(row.get("ok")) for row in rows),
        "failed_jobs": sum(not row.get("ok", False) for row in rows),
        "completed_samples": sum(int(row.get("completed_samples") or 0) for row in rows),
        "remaining_unattempted_calls": args.approved_call_count - sum(int(row.get("calls") or 0) for row in rows),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                                encoding="utf-8")
    print(json.dumps(summary), flush=True)
    if failures:
        raise SystemExit("repair completed with failed jobs; review the resulting empty rows")


if __name__ == "__main__":
    main()
