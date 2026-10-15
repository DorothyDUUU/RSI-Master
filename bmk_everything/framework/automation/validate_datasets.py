#!/usr/bin/env python3
"""Stream one row from every generated integration and verify its field mapping."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
from pathlib import Path


def check(directory: Path, python: Path, timeout: int) -> dict:
    process = subprocess.run(
        [str(python), "run_benchmark.py", "--config", "config/config.yaml",
         "--dry-run", "--validate-data"],
        cwd=directory, capture_output=True, text=True, timeout=timeout,
    )
    return {
        "benchmark-id": directory.name,
        "ok": process.returncode == 0,
        "stdout": process.stdout[-1000:],
        "stderr": process.stderr[-2000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--retry-from", type=Path,
                        help="Only retry failed benchmark ids from an earlier result JSON")
    args = parser.parse_args()
    format_dir = args.format_dir.resolve()
    python = format_dir.parent / "framework/.venv/bin/python"
    directories = sorted(marker.parent for marker in format_dir.glob("*/.automated-integration"))
    if args.retry_from:
        previous = json.loads(args.retry_from.read_text())
        failed_ids = {item["benchmark-id"] for item in previous if not item["ok"]}
        directories = [directory for directory in directories if directory.name in failed_ids]
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(check, directory, python, args.timeout): directory for directory in directories}
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                result = {"benchmark-id": futures[future].name, "ok": False, "stderr": repr(exc)}
            results.append(result)
            print(f"[{'OK' if result['ok'] else 'FAIL'}] {result['benchmark-id']}", flush=True)
    results.sort(key=lambda item: item["benchmark-id"])
    output = args.output or args.format_dir.parent / "framework/automation/dataset_validation.json"
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    failed = [item for item in results if not item["ok"]]
    print(json.dumps({"checked": len(results), "passed": len(results) - len(failed),
                      "failed": len(failed), "output": str(output)}, ensure_ascii=False))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
