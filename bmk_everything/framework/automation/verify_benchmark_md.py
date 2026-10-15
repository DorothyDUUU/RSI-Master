#!/usr/bin/env python3
"""Validate every BENCHMARK.md against the five-section contract.

A BENCHMARK.md is acceptable when it:
  - exists and is substantially non-trivial (>= 1500 bytes)
  - has the five `## 1.` .. `## 5.` sections
  - contains no unresolved `<...>` runtime placeholders
  - names the benchmark and its scorer / eval-method
  - covers every declared split
  - when the scorer is `unscored`, names the required-metric judge

Exit code 1 lists every failing benchmark id.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
SECTIONS = [r"^##\s*1[\.、:：]", r"^##\s*2[\.、:：]", r"^##\s*3[\.、:：]",
            r"^##\s*4[\.、:：]", r"^##\s*5[\.、:：]"]
PLACEHOLDER = re.compile(r"<[A-Z_]+>")


def check_benchmark(benchmark_id: str, bench_dir: Path) -> dict:
    failures = []
    md_path = bench_dir / "BENCHMARK.md"
    if not md_path.is_file():
        return {"benchmark-id": benchmark_id, "ok": False, "failures": ["missing BENCHMARK.md"]}
    text = md_path.read_text(encoding="utf-8")
    if len(text.encode("utf-8")) < 1500:
        failures.append("too short (<1500 bytes)")
    for pattern in SECTIONS:
        if not re.search(pattern, text, re.MULTILINE):
            failures.append(f"missing section matching {pattern!r}")
    placeholders = sorted(set(PLACEHOLDER.findall(text)))
    if placeholders:
        failures.append(f"unresolved placeholders: {placeholders}")
    manifest = {}
    config_path = bench_dir / "config.json"
    if config_path.is_file():
        manifest = json.loads(config_path.read_text(encoding="utf-8"))
    name = manifest.get("benchmark-name") or benchmark_id
    if name not in text and benchmark_id not in text:
        failures.append("does not name the benchmark")
    scorer = manifest.get("eval-method")
    if scorer and scorer not in text:
        failures.append(f"does not mention scorer/eval-method: {scorer!r}")
    for split in manifest.get("split") or ["default"]:
        if str(split) not in text:
            failures.append(f"does not cover declared split: {split!r}")
    required = manifest.get("required-metric")
    if scorer == "unscored" and required and required not in text:
        failures.append(f"unscored benchmark missing required-metric: {required!r}")
    return {"benchmark-id": benchmark_id, "ok": not failures, "failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format-dir", type=Path, default=REPO_DIR / "format")
    parser.add_argument("--ids", help="Comma-separated benchmark ids; default: all")
    args = parser.parse_args()
    format_dir = args.format_dir.resolve()
    selected = {i.strip() for i in args.ids.split(",")} if args.ids else None
    ids = []
    for config_path in sorted(format_dir.glob("*/config.json")):
        benchmark_id = config_path.parent.name
        if benchmark_id == "benchmark_example":
            continue
        if selected and benchmark_id not in selected:
            continue
        ids.append(benchmark_id)
    results = [check_benchmark(benchmark_id, format_dir / benchmark_id) for benchmark_id in ids]
    failed = [result for result in results if not result["ok"]]
    print(json.dumps({"checked": len(results), "passed": len(results) - len(failed),
                      "failed": len(failed)}, ensure_ascii=False))
    for result in failed:
        print(f"FAIL {result['benchmark-id']}: {'; '.join(result['failures'])}", file=__import__("sys").stderr)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
