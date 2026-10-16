#!/usr/bin/env python3
from __future__ import annotations
import argparse
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR.parent.parent / "framework"))
from core.generic_runner import dry_run, run

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-data", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        dry_run(__import__("json").loads((BENCH_DIR / "spec.json").read_text()))
        if args.validate_data:
            run(BENCH_DIR, args.config, 1, validate_data=True)
        print(f"[dry-run] {BENCH_DIR.name}: spec OK")
        return
    run(BENCH_DIR, args.config, args.limit)

if __name__ == "__main__":
    main()
