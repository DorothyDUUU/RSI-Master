#!/usr/bin/env python3
"""Run many benchmarks in parallel against one model endpoint under a global concurrency cap.

Each benchmark runs through ``framework/run_eval.py``; per-process parallelism comes from the
benchmark's ``custom.max_workers`` (see --workers-per-bench), and the scheduler keeps at most
``max_concurrency // workers_per_bench`` subprocesses alive so the aggregate request rate never
exceeds ``max_concurrency``.  Trajectories (``samples.jsonl``) and scores (``reward.json``) land
under ``data/output/<benchmark-id>/<trial>/`` for every benchmark in the manifest.

State is checkpointed to ``--state`` so an interrupted batch can be resumed with the same command.

Usage example::

    framework/.venv/bin/python framework/automation/batch_run.py \
      --limit 500 --max-concurrency 150 --workers-per-bench 10 \
      --trial qwen36-20260808 \
      --openai-api-base https://.../v1 --openai-api-token dummy-noauth \
      --model Qwen3.6-35B-A3B --config-override data/config-overrides/qwen36.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
FORMAT_DIR = REPO_DIR / "format"
PYTHON = REPO_DIR / "framework/.venv/bin/python"
REAL_SAFE = __import__("re").compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

_GLOBAL_LIMIT = 500  # default rows per benchmark, overridden from --limit in main()


def load_manifest() -> list[dict]:
    """Discover runnable benchmarks (auto + legacy) with local row counts."""
    benches = []
    for spec_path in sorted(FORMAT_DIR.glob("*/spec.json")):
        try:
            spec = json.loads(spec_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        bench_dir = spec_path.parent
        config_path = bench_dir / "config.json"
        if not config_path.exists():
            continue
        manifest = json.loads(config_path.read_text())
        if manifest.get("integration-status", "catalogued") not in {"runnable", "verified"}:
            continue
        rows = (spec.get("dataset") or {}).get("local_rows") or 0
        split = (spec.get("dataset") or {}).get("split") or "default"
        benches.append({
            "id": manifest.get("benchmark-id", bench_dir.name),
            "rows": int(rows),
            "split": split,
            "scorer": spec.get("scorer", ""),
        })
    return benches


def resolve_limit(rows: int, cap: int) -> int:
    """Rows to run per benchmark: full dataset when small, ``cap`` when larger.

    Always returns a concrete number because the thin run_benchmark.py defaults to
    ``--limit 3``; omitting the flag silently truncates small datasets to 3 rows.
    """
    return min(rows, cap)


def load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    tmp.replace(path)


def run_benchmark(bench: dict, args) -> subprocess.Popen:
    trial = args.trial
    limit = resolve_limit(bench["rows"], args.limit)
    # NOTE: --split is intentionally NOT passed.  config.json declares `["default"]`
    # and run_eval validates the flag against that list; generic_runner reads the
    # real local split from spec.dataset.split itself.
    command = [
        str(PYTHON), str(REPO_DIR / "framework/run_eval.py"), bench["id"],
        "--openai-api-base", args.openai_api_base,
        "--openai-api-token", args.openai_api_token,
        "--model", args.model,
        "--trial-name", trial,
    ]
    if args.config_override:
        command += ["--config-path", args.config_override]
    command += ["--limit", str(limit)]
    log_path = REPO_DIR / "data/output/qwen36-batch/<TRIAL>/logs/<ID>.log"
    log_path = Path(str(log_path).replace("<TRIAL>", trial).replace("<ID>", bench["id"]))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w")
    return subprocess.Popen(command, cwd=REPO_DIR, stdout=log_file, stderr=subprocess.STDOUT), log_file


def parse_result(bench: dict, returncode: int, output_dir: Path) -> dict:
    entry = {
        "benchmark": bench["id"],
        "rows": bench["rows"],
        "limit": resolve_limit(bench["rows"], _GLOBAL_LIMIT),
        "returncode": returncode,
        "ok": returncode == 0,
    }
    reward_path = output_dir / "reward.json"
    if reward_path.exists():
        try:
            reward = json.loads(reward_path.read_text())
            entry["reward"] = reward.get("reward")
            meta = reward.get("meta_data") or {}
            entry["completed"] = meta.get("completed")
            entry["incomplete"] = meta.get("incomplete")
            entry["scoring_supported"] = meta.get("scoring_supported", True)
        except (OSError, json.JSONDecodeError):
            entry["reward_error"] = True
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=500, help="rows per benchmark; smaller datasets run in full")
    parser.add_argument("--max-concurrency", type=int, default=150)
    parser.add_argument("--workers-per-bench", type=int, default=10)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--state", default="data/output/qwen36-batch/<TRIAL>/state.json")
    parser.add_argument("--config-override")
    parser.add_argument("--openai-api-base", required=True)
    parser.add_argument("--openai-api-token", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout-hours", type=float, default=6.0)
    parser.add_argument("--benchmarks", help="comma-separated subset (default: all runnable)")
    args = parser.parse_args()

    if not REAL_SAFE.fullmatch(args.trial):
        raise SystemExit(f"error: unsafe trial name {args.trial!r}")
    global _GLOBAL_LIMIT
    _GLOBAL_LIMIT = args.limit
    args.state = str(args.state).replace("<TRIAL>", args.trial)
    state_path = REPO_DIR / args.state

    benches = load_manifest()
    if args.benchmarks:
        wanted = set(args.benchmarks.split(","))
        benches = [b for b in benches if b["id"] in wanted]
    if not benches:
        raise SystemExit("no benchmarks to run")

    state = load_state(state_path)
    state.setdefault("trial", args.trial)
    state.setdefault("model", args.model)
    state.setdefault("base", args.openai_api_base)
    state.setdefault("limit", args.limit)
    if "pending" not in state:
        state["pending"] = [b["id"] for b in benches]
    if "done" not in state:
        state["done"] = {}
    if "errors" not in state:
        state["errors"] = {}
    if "running" not in state:
        state["running"] = {}
    pending = [pid for pid in state["pending"] if pid not in state["done"] and pid not in state["errors"]]
    state["pending"] = pending

    by_id = {b["id"]: b for b in benches}
    active: dict[str, tuple[subprocess.Popen, object, float]] = {}  # id -> (proc, log, launched_at)
    slot_count = max(1, args.max_concurrency // args.workers_per_bench)
    print(f"[batch_run] {len(by_id)} benchmarks, {len(pending)} pending, "
          f"{len(active)} active, cap={args.max_concurrency} ({slot_count}x{args.workers_per_bench})",
          flush=True)

    def launch(bench_id: str) -> None:
        bench = by_id[bench_id]
        proc, log = run_benchmark(bench, args)
        state.setdefault("running", {})[bench_id] = {"launched": time.time()}
        active[bench_id] = (proc, log, time.time())
        save_state(state_path, state)
        print(f"[batch_run] start {bench_id} rows={bench['rows']} "
              f"limit={resolve_limit(bench['rows'], args.limit)} (active={len(active)})", flush=True)

    try:
        while True:
            # Reap finished processes.
            for bench_id, (proc, log, launched) in list(active.items()):
                if proc.poll() is None:
                    continue
                log.close()
                bench = by_id[bench_id]
                output_dir = REPO_DIR / f"data/output/{bench_id}/{args.trial}"
                entry = parse_result(bench, proc.returncode, output_dir)
                entry["elapsed_s"] = round(time.time() - launched, 1)
                if proc.returncode == 0:
                    state["done"][bench_id] = entry
                    status = f"reward={entry.get('reward')} completed={entry.get('completed')} incomplete={entry.get('incomplete')}"
                else:
                    state["errors"][bench_id] = entry
                    status = f"exit={proc.returncode}"
                del active[bench_id]
                save_state(state_path, state)
                print(f"[batch_run] done {bench_id}: {status} ({entry['elapsed_s']}s, "
                      f"active={len(active)}, done={len(state['done'])}, "
                      f"err={len(state['errors'])})", flush=True)

            # Kill stragglers.
            for bench_id, (proc, log, launched) in list(active.items()):
                if time.time() - launched > args.timeout_hours * 3600:
                    proc.kill()
                    log.close()
                    state["errors"][bench_id] = {"benchmark": bench_id, "reason": "timeout"}
                    del active[bench_id]
                    print(f"[batch_run] TIMEOUT {bench_id}", flush=True)

            # Launch until slots are full.
            while len(active) < slot_count and state["pending"]:
                bench_id = state["pending"].pop(0)
                if bench_id in state["done"] or bench_id in state["errors"]:
                    continue
                launch(bench_id)

            if not active and not state["pending"]:
                break
            time.sleep(5)
    except KeyboardInterrupt:
        for bench_id, (proc, log, _) in list(active.items()):
            proc.terminate()
        save_state(state_path, state)
        print("\n[batch_run] interrupted; state saved to", state_path, flush=True)
        sys.exit(130)

    # Final summary.
    summary = REPO_DIR / f"data/output/qwen36-batch/{args.trial}/summary.jsonl"
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w") as out:
        for bench_id in by_id:
            if bench_id in state["done"]:
                row = state["done"][bench_id]
            elif bench_id in state["errors"]:
                row = state["errors"][bench_id]
            else:
                continue
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    save_state(state_path, state)
    print(f"[batch_run] complete: done={len(state['done'])} errors={len(state['errors'])} "
          f"total={len(by_id)} summary={summary}", flush=True)


if __name__ == "__main__":
    main()
