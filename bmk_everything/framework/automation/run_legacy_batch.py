#!/usr/bin/env python3
"""Run the 4 hand-written legacy benchmarks (gsm8k / gpqa / aime2025 / aime2026) against one endpoint.

These lack spec.json so ``batch_run.py`` skips them; they are run here through the same
``framework/run_eval.py`` contract.  Each benchmark uses its own runner/.venv, and the runs are
sequential (per-benchmark ThreadPool inside each runner) so the aggregate request rate is bounded
by a single benchmark's ``max_workers`` — safe to launch alongside or after ``batch_run.py``.

Usage::

    framework/.venv/bin/python framework/automation/run_legacy_batch.py \
      --trial qwen36-legacy-20260808 \
      --openai-api-base https://.../v1 --openai-api-token dummy-noauth \
      --model Qwen3.6-35B-A3B --config-override data/config-overrides/qwen36.yaml
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PYTHON = REPO / "framework/.venv/bin/python"
RUN_EVAL = REPO / "framework/run_eval.py"
LEGACY = ["gsm8k", "gpqa", "aime2025", "aime2026"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", required=True)
    parser.add_argument("--openai-api-base", required=True)
    parser.add_argument("--openai-api-token", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config-override", default=None)
    args = parser.parse_args()

    for bench_id in LEGACY:
        cmd = [str(PYTHON), str(RUN_EVAL), bench_id,
               "--openai-api-base", args.openai_api_base,
               "--openai-api-token", args.openai_api_token,
               "--model", args.model,
               "--trial-name", args.trial]
        if args.config_override:
            cmd += ["--config-path", args.config_override]
        print(f"[legacy] start {bench_id}", flush=True)
        process = subprocess.run(cmd, cwd=REPO)
        if process.returncode:
            print(f"[legacy] {bench_id} exit={process.returncode}", flush=True)
        else:
            print(f"[legacy] {bench_id} ok", flush=True)


if __name__ == "__main__":
    main()
