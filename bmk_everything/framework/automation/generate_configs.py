#!/usr/bin/env python3
"""Generate detailed gsm8k-style config/config.yaml for the generated benchmarks.

The runtime placeholders (<MODEL_NAME>, <SPLIT_NAME>, ...) are substituted by
`framework/run_eval.py` (or the benchmark runner's own env fallbacks) so the same
committed config works across models and providers.  `custom.dataset_path`
mirrors the local data tree and is informational for the generic runner, which
serves rows from `spec.json`'s dataset segment.

Idempotent and non-destructive: only generated integrations (those carrying the
`.automated-integration` marker) are written; the three hand-written legacy
configs are never touched.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]


CONFIG_TEMPLATE = """general:
  anthropic_api_base: <ANTHROPIC_API_BASE>
  anthropic_api_token: <ANTHROPIC_API_TOKEN>
  openai_api_base: <OPENAI_API_BASE>
  openai_api_token: <OPENAI_API_TOKEN>
  model: <MODEL_NAME>
  split: <SPLIT_NAME>
  output_dir: data/output/<BENCHMARK_ID>/<TRIAL_NAME>
custom:
  dataset_path: data/datasets/{benchmark_id}
  dataset_revision: {revision}
  max_tokens: 4096
  max_workers: 8
  timeout: 180
  temperature: 0.0
"""


def spec_revision(spec: dict) -> str:
    dataset = spec.get("dataset") or {}
    hf = dataset.get("hf") or {}
    return str(hf.get("revision") or dataset.get("revision") or "main")


def generate_one(benchmark_id: str, spec: dict, config_path: Path) -> bool:
    text = CONFIG_TEMPLATE.format(
        benchmark_id=benchmark_id,
        revision=spec_revision(spec),
    )
    if config_path.exists() and config_path.read_text(encoding="utf-8") == text:
        return False
    tmp = config_path.parent / f".tmp-{os.getpid()}"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(config_path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format-dir", type=Path, default=REPO_DIR / "format")
    parser.add_argument("--ids", help="Comma-separated benchmark ids; default: all generated")
    parser.add_argument("--dry-run", action="store_true", help="Report what would change")
    args = parser.parse_args()
    format_dir = args.format_dir.resolve()
    selected = {i.strip() for i in args.ids.split(",")} if args.ids else None

    markers = sorted(format_dir.glob("*/.automated-integration"))
    if selected:
        markers = [marker for marker in markers if marker.parent.name in selected]
    changed = []
    for marker in markers:
        benchmark_id = marker.parent.name
        spec = json.loads((marker.parent / "spec.json").read_text(encoding="utf-8"))
        expected = CONFIG_TEMPLATE.format(benchmark_id=benchmark_id,
                                          revision=spec_revision(spec))
        config_path = marker.parent / "config" / "config.yaml"
        if config_path.exists() and config_path.read_text(encoding="utf-8") == expected:
            continue
        changed.append(benchmark_id)
        if not args.dry_run:
            generate_one(benchmark_id, spec, config_path)
    print(json.dumps({"checked": len(markers), "changed": changed,
                      "dry_run": args.dry_run}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
