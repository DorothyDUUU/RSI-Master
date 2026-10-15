"""Run pinned PostTrainBench Inspect tasks against an existing OpenAI endpoint.

This adapter preserves BmkEverything's endpoint reuse and reward.json contract
while sourcing datasets, prompts, solvers, and scorers from the exact Inspect
Evals revision pinned by PostTrainBench.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import yaml


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_RUNTIME_ROOT = REPO_ROOT / "external/PostTrainBenchRuntime"
DEFAULT_PTB_ROOT = REPO_ROOT / "external/PostTrainBench"
RUNTIME_PYTHON = pathlib.Path(sys.executable)
EXPECTED_PTB_COMMIT = "3e3f1eee3551c243aeebe5f67ab5936d6c646c75"
EXPECTED_INSPECT_EVALS_COMMIT = "06001a83e6d7c709c2ede0570dce7f1031a0bad8"


def _placeholder(value: object) -> bool:
    return isinstance(value, str) and value.startswith("<") and value.endswith(">")


def _general(config: dict, key: str, env: str, default: object = None):
    value = (config.get("general") or {}).get(key)
    if value is None or _placeholder(value):
        return os.environ.get(env, default)
    return value


def _custom(config: dict, key: str, default: object = None):
    return (config.get("custom") or {}).get(key, default)


def _ensure_runtime() -> None:
    if os.environ.get("BMK_POSTTRAINBENCH_RUNTIME") == "1":
        import inspect_ai  # noqa: F401
        import inspect_evals  # noqa: F401
        return

    runtime_root = pathlib.Path(
        os.environ.get("POSTTRAINBENCH_RUNTIME_ROOT", str(DEFAULT_RUNTIME_ROOT))
    )
    inspect_ai_src = runtime_root / "runtime_sources/inspect_ai_vllm_stdout/src"
    inspect_evals_src = runtime_root / "runtime_sources/inspect_evals/src"
    packages = runtime_root / "site-packages"
    required = (RUNTIME_PYTHON, inspect_ai_src, inspect_evals_src, packages)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"PostTrainBench runtime is incomplete: {missing}")
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = ":".join(
        [str(inspect_ai_src), str(packages), str(inspect_evals_src)]
        + ([existing] if existing else [])
    )
    env["BMK_POSTTRAINBENCH_RUNTIME"] = "1"
    os.execve(str(RUNTIME_PYTHON), [str(RUNTIME_PYTHON), *sys.argv], env)


def _verify_pins() -> None:
    import subprocess

    ptb_root = pathlib.Path(os.environ.get("POSTTRAINBENCH_ROOT", str(DEFAULT_PTB_ROOT)))
    runtime_root = pathlib.Path(
        os.environ.get("POSTTRAINBENCH_RUNTIME_ROOT", str(DEFAULT_RUNTIME_ROOT))
    )
    pins = (
        (ptb_root, EXPECTED_PTB_COMMIT),
        (runtime_root / "runtime_sources/inspect_evals", EXPECTED_INSPECT_EVALS_COMMIT),
    )
    for repo, expected in pins:
        actual = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != expected:
            raise RuntimeError(f"unexpected revision for {repo}: {actual} != {expected}")


def _task(benchmark_id: str):
    if benchmark_id == "bfcl":
        from inspect_evals.bfcl import bfcl

        return bfcl()
    if benchmark_id == "humaneval":
        from inspect_evals.humaneval import humaneval

        return humaneval(sandbox="local")
    raise ValueError(f"unsupported Inspect benchmark: {benchmark_id}")


def main(benchmark_id: str) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", default=None)
    args = parser.parse_args()

    _ensure_runtime()
    _verify_pins()

    from inspect_ai import eval as inspect_eval
    from inspect_ai.util._display import init_display_type

    config = yaml.safe_load(pathlib.Path(args.config).read_text()) or {}
    model = str(_general(config, "model", "MODEL_NAME", ""))
    base_url = str(_general(config, "openai_api_base", "OPENAI_API_BASE", ""))
    api_key = str(_general(config, "openai_api_token", "OPENAI_API_TOKEN", "sk-123"))
    output_dir = pathlib.Path(str(_general(config, "output_dir", "OUTPUT_DIR", "output")))
    max_tokens = int(_custom(config, "max_tokens", 4000))
    max_connections = int(
        _custom(config, "max_workers", _custom(config, "max_connections", 1))
    )
    if not model or not base_url:
        raise SystemExit("model and openai_api_base are required")
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["OPENAI_API_KEY"] = api_key
    init_display_type("plain")
    kwargs = {}
    if args.limit is not None and args.limit >= 0:
        kwargs["limit"] = args.limit
    logs = inspect_eval(
        _task(benchmark_id),
        model=f"openai/{model}",
        model_base_url=base_url,
        model_args={"api_key": api_key, "responses_api": False},
        score_display=False,
        log_realtime=False,
        log_format="json",
        timeout=18_000_000,
        attempt_timeout=18_000_000,
        max_tokens=max_tokens,
        max_connections=max_connections,
        **kwargs,
    )
    if len(logs) != 1 or logs[0].results is None:
        raise RuntimeError(f"Inspect returned no result: {logs}")
    score = logs[0].results.scores[0]
    metrics = {name: metric.value for name, metric in score.metrics.items()}
    accuracy = float(metrics["accuracy"])
    total = len(logs[0].samples or [])
    reward = {
        "reward": accuracy,
        "meta_data": {
            "benchmark_id": benchmark_id,
            "protocol": "PostTrainBench official Inspect task via existing endpoint",
            "posttrainbench_commit": EXPECTED_PTB_COMMIT,
            "inspect_evals_commit": EXPECTED_INSPECT_EVALS_COMMIT,
            "model": model,
            "attempted": total,
            "completed": total,
            "incomplete": 0,
            "accuracy": accuracy,
            "metrics": metrics,
        },
    }
    (output_dir / "reward.json").write_text(
        json.dumps(reward, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


__all__ = ["main"]
