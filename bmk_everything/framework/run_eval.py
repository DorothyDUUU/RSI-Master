#!/usr/bin/env python3
"""Unified, status-aware benchmark evaluation entrypoint."""
from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import subprocess
import sys
from pathlib import Path

import yaml

# This entrypoint lives in framework/ and is intended to be invoked from the
# repo root (framework's parent), e.g.:
#   framework/.venv/bin/python framework/run_eval.py <benchmark-id> --limit 500
FRAMEWORK_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = Path(os.environ.get("BENCHMARK_ROOT") or FRAMEWORK_DIR.parent).expanduser().resolve()
FORMAT_DIR = BENCHMARK_ROOT / "format"
sys.path.insert(0, str(FRAMEWORK_DIR))

from core.adapters import get_adapter
from core.catalog import discover, find
from core.config import deep_merge, redact


DEFAULT_API_BASE = "https://api.moonshot.cn/v1"
DEFAULT_ANTHROPIC_API_BASE = "https://api.anthropic.com/v1"
DEFAULT_API_TOKEN = ""
DEFAULT_MODEL_NAME = "kimi-k2.7-code"
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def resolve_param(flag_value: str | None, env_name: str, default: str) -> str:
    return flag_value or os.environ.get(env_name) or default


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        return yaml.safe_load(source) or {}


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def list_benchmarks(status: str | None, domain: str | None, eval_method: str | None) -> None:
    rows = discover(FORMAT_DIR)
    if status:
        rows = [row for row in rows if row.get("integration-status") == status]
    if domain:
        rows = [row for row in rows if row.get("domain") == domain]
    if eval_method:
        rows = [row for row in rows if row.get("eval-method") == eval_method]
    print("available benchmarks in format/:\n")
    for row in rows:
        print(
            f"  {row['benchmark-id']:<42}"
            f"  {row.get('integration-status', ''):<11}"
            f"  {row.get('domain', '-'):<22}"
            f"  splits={row.get('split') or ['default']}"
        )
    print(f"\ntotal: {len(rows)}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Unified BmkEverything evaluation entrypoint")
    result.add_argument("benchmark_id", nargs="?")
    result.add_argument("--benchmark-id", dest="benchmark_id_flag")
    result.add_argument("--model")
    result.add_argument("--split")
    result.add_argument("--trial-name")
    result.add_argument("--config-path")
    result.add_argument("--limit", type=int)
    result.add_argument("--sample-indices",
                        help="Comma-separated dataset indices; intended for approved targeted repair")
    result.add_argument("--openai-api-base")
    result.add_argument("--openai-api-token")
    result.add_argument("--anthropic-api-base")
    result.add_argument("--anthropic-api-token")
    result.add_argument("--list", action="store_true")
    result.add_argument("--status", choices=["catalogued", "runnable", "verified"])
    result.add_argument("--domain")
    result.add_argument("--eval-method")
    result.add_argument("--dry-run", action="store_true", help="Resolve only; do not write or call a model")
    result.add_argument("--validate-data", action="store_true",
                        help="With --dry-run, stream one row to validate the pinned dataset schema")
    return result


def main() -> None:
    args = parser().parse_args()
    if not FORMAT_DIR.is_dir():
        raise SystemExit(
            f"error: benchmark definitions missing: {FORMAT_DIR}. "
            "This release includes framework only; supply a complete benchmark tree "
            "via BENCHMARK_ROOT. See bmk_everything/SETUP.md."
        )
    if args.list:
        list_benchmarks(args.status, args.domain, args.eval_method)
        return

    benchmark_id = args.benchmark_id_flag or args.benchmark_id or os.environ.get("BENCHMARK_ID")
    if not benchmark_id:
        parser().error("missing benchmark-id")
    if not SAFE_NAME.fullmatch(benchmark_id):
        raise SystemExit(f"error: unsafe benchmark-id {benchmark_id!r}")

    try:
        manifest = find(FORMAT_DIR, benchmark_id)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    bench_dir = manifest["_dir"]
    status = manifest["integration-status"]
    adapter_name = manifest.get("adapter", "legacy_runner" if status in {"runnable", "verified"} else None)
    adapter = get_adapter(adapter_name)

    declared_splits = manifest.get("split") or ["default"]
    split = args.split or os.environ.get("SPLIT_NAME")
    if not split:
        split = "default" if "default" in declared_splits else declared_splits[0]
    if split not in declared_splits:
        raise SystemExit(f"error: split {split!r} not declared; available: {declared_splits}")

    config_path = bench_dir / "config" / "config.yaml"
    config = load_yaml(config_path) if config_path.exists() else {"general": {}, "custom": {}}
    if args.config_path:
        config = deep_merge(config, load_yaml(Path(args.config_path)))

    model = resolve_param(args.model, "MODEL_NAME", DEFAULT_MODEL_NAME)
    openai_base = resolve_param(args.openai_api_base, "OPENAI_API_BASE", DEFAULT_API_BASE)
    openai_token = resolve_param(args.openai_api_token, "OPENAI_API_TOKEN", DEFAULT_API_TOKEN)
    anthropic_base = resolve_param(args.anthropic_api_base, "ANTHROPIC_API_BASE", DEFAULT_ANTHROPIC_API_BASE)
    anthropic_token = resolve_param(args.anthropic_api_token, "ANTHROPIC_API_TOKEN", DEFAULT_API_TOKEN)
    trial_name = args.trial_name or os.environ.get("TRIAL_NAME") or f"default_{secrets.token_hex(4)}"
    if not SAFE_NAME.fullmatch(trial_name):
        raise SystemExit(f"error: unsafe trial-name {trial_name!r}")

    config.setdefault("general", {}).update({
        "model": model,
        "split": split,
        "openai_api_base": openai_base,
        "openai_api_token": openai_token,
        "anthropic_api_base": anthropic_base,
        "anthropic_api_token": anthropic_token,
    })
    template = config["general"].get("output_dir") or str(FORMAT_DIR.parent / "data/output/<BENCHMARK_ID>/<TRIAL_NAME>")
    output_dir = Path(template.replace("<BENCHMARK_ID>", benchmark_id).replace("<TRIAL_NAME>", trial_name))
    if not output_dir.is_absolute():
        output_dir = (FORMAT_DIR.parent / output_dir).resolve()
    config["general"]["output_dir"] = str(output_dir)

    dry_run = {
        "benchmark_id": benchmark_id,
        "status": status,
        "adapter": adapter_name,
        "model": model,
        "split": split,
        "trial_name": trial_name,
        "output_dir": str(output_dir),
        "config": redact(config),
    }
    if args.dry_run:
        print(json.dumps(dry_run, ensure_ascii=False, indent=2))
        runner = bench_dir / "run_benchmark.py"
        if runner.exists() and (bench_dir / "spec.json").exists():
            python = BENCHMARK_ROOT / "framework/.venv/bin/python"
            if not python.exists():
                python = Path(sys.executable)
            command = [str(python), str(runner), "--config", str(config_path), "--dry-run"]
            if args.validate_data:
                command.append("--validate-data")
            process = subprocess.run(command, cwd=bench_dir, env=dict(os.environ))
            if process.returncode:
                raise SystemExit(process.returncode)
        return

    if status not in {"runnable", "verified"}:
        reason = manifest.get("disabled-reason") or "online data and scoring verification is incomplete"
        raise SystemExit(f"error: benchmark {benchmark_id!r} is catalogued, not runnable: {reason}")
    if not adapter or not adapter.executable:
        raise SystemExit(f"error: adapter {adapter_name!r} is not executable yet")

    python = bench_dir / ".venv/bin/python"
    if not python.exists():
        python = BENCHMARK_ROOT / "framework/.venv/bin/python"
    runner = bench_dir / "run_benchmark.py"
    if not python.exists() or not runner.exists():
        raise SystemExit(f"error: runnable benchmark is missing {python if not python.exists() else runner}")

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = output_dir / "config.yaml"
    with resolved_config.open("w", encoding="utf-8") as destination:
        yaml.safe_dump(redact(config), destination, allow_unicode=True, sort_keys=False)

    command = [str(python), "run_benchmark.py", "--config", str(resolved_config)]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    child_env = dict(os.environ)
    child_env["BENCHMARK_ROOT"] = str(BENCHMARK_ROOT)
    if args.sample_indices:
        try:
            sample_indices = [int(item) for item in args.sample_indices.split(",")]
        except ValueError as exc:
            raise SystemExit("error: --sample-indices must contain integers") from exc
        if (any(index < 0 for index in sample_indices)
                or len(sample_indices) != len(set(sample_indices))):
            raise SystemExit("error: --sample-indices must be unique non-negative integers")
        child_env["SAMPLE_INDICES"] = ",".join(str(index) for index in sorted(sample_indices))
    child_env.update({
        "MODEL_NAME": model,
        "OPENAI_API_BASE": openai_base,
        "OPENAI_API_TOKEN": openai_token,
        "ANTHROPIC_API_BASE": anthropic_base,
        "ANTHROPIC_API_TOKEN": anthropic_token,
        "SPLIT_NAME": split,
        "TRIAL_NAME": trial_name,
    })
    print(f"[run_eval] benchmark={benchmark_id} model={model} split={split} trial={trial_name}")
    print(f"[run_eval] output_dir={output_dir}")
    process = subprocess.run(command, cwd=bench_dir, env=child_env)
    if process.returncode:
        raise SystemExit(process.returncode)

    reward_path = output_dir / "reward.json"
    if not reward_path.exists():
        raise SystemExit(f"error: runner succeeded but did not create {reward_path}")
    reward = read_json(reward_path)
    meta = reward.get("meta_data") or {}
    incomplete = int(meta.get("incomplete") or 0)
    scoring_supported = meta.get("scoring_supported", True)
    print(json.dumps({
        "benchmark": benchmark_id,
        "status": "incomplete" if incomplete else "ok",
        "reward": reward.get("reward"),
        "completed": meta.get("completed"),
        "incomplete": incomplete,
        "scoring_supported": scoring_supported,
        "output": str(output_dir),
    }, ensure_ascii=False, indent=2))
    if incomplete:
        raise SystemExit(f"error: {benchmark_id} returned {incomplete} incomplete samples")


if __name__ == "__main__":
    main()
