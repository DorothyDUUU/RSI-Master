#!/usr/bin/env python3
"""Plan, but never execute, a targeted repair of empty answer records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().split("\n") if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-calls", type=int)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--reasoning-batch", type=Path, action="append", default=[])
    parser.add_argument("--only-without-reasoning", action="store_true")
    args = parser.parse_args()
    latest = {}
    for row in jsonl(args.batch_dir / "state.jsonl"):
        latest[row["key"]] = row
    reasoning_keys = set()
    for batch in args.reasoning_batch:
        for state in jsonl(batch / "state.jsonl"):
            samples_path = Path(state["output_dir"]) / "samples.jsonl"
            if not samples_path.exists():
                continue
            for sample in jsonl(samples_path):
                if str(sample.get("reasoning_content") or "").strip():
                    reasoning_keys.add((state["key"], int(sample["index"])))
    jobs = []
    for benchmark_id, state in sorted(latest.items()):
        samples = jsonl(Path(state["output_dir"]) / "samples.jsonl")
        indices = [int(row["index"]) for row in samples
                   if not str(row.get("response") or "").strip()]
        if args.only_without_reasoning:
            indices = [index for index in indices
                       if (benchmark_id, index) not in reasoning_keys]
        if indices:
            jobs.append({
                "benchmark_id": benchmark_id,
                "indices": indices,
                "calls": len(indices),
                "source_output_dir": state["output_dir"],
            })
    calls = sum(job["calls"] for job in jobs)
    if args.expected_calls is not None and calls != args.expected_calls:
        raise SystemExit(f"planned calls {calls} != expected {args.expected_calls}")
    plan = {
        "source_batch": str(args.batch_dir.resolve()),
        "benchmarks": len(jobs),
        "planned_calls": calls,
        "max_tokens": args.max_tokens,
        "execute": False,
        "note": "This file is a repair plan only; a separate explicit approval is required to call Kimi.",
        "jobs": jobs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: plan[key] for key in ("benchmarks", "planned_calls", "max_tokens")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
