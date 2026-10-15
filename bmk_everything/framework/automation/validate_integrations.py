#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format-dir", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=200)
    args = parser.parse_args()
    registry = json.loads((args.format_dir / "benchmark.json").read_text())
    ids = [item["benchmark-id"] for item in registry]
    errors = []
    domains = Counter()
    if len(ids) != args.expected or len(set(ids)) != args.expected:
        errors.append(f"registry count/uniqueness: {len(ids)}/{len(set(ids))}")
    for benchmark_id in ids:
        directory = args.format_dir / benchmark_id
        generated = (directory / ".automated-integration").exists()
        required_files = ("config.json", "run_benchmark.py", "BENCHMARK.md")
        if generated:
            required_files += ("spec.json", "config/config.yaml")
        for required in required_files:
            if not (directory / required).exists():
                errors.append(f"{benchmark_id}: missing {required}")
        manifest = json.loads((directory / "config.json").read_text())
        domains[manifest.get("domain", "legacy")] += 1
        if manifest.get("integration-status", "runnable") != "runnable":
            errors.append(f"{benchmark_id}: status/resource gate failed")
        if generated:
            spec = json.loads((directory / "spec.json").read_text())
            if spec["resource-class"] != "light":
                errors.append(f"{benchmark_id}: resource gate failed")
            process = subprocess.run([sys.executable, "run_benchmark.py", "--config", "config/config.yaml", "--dry-run"],
                                     cwd=directory, capture_output=True, text=True)
            if process.returncode:
                errors.append(f"{benchmark_id}: dry-run failed: {process.stderr[-500:]}")
    if any(domains[name] < 25 for name in ("agentic_tool_calling", "code_swe", "domain_knowledge", "math_reasoning")):
        errors.append(f"domain minimum failed: {dict(domains)}")
    print(json.dumps({"count": len(ids), "domains": domains, "errors": errors[:20]}, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
