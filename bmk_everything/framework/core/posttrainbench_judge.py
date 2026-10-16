"""Run PostTrainBench judge tasks against RSI-Master's existing vLLM server."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import sys
from urllib.parse import urlparse

import yaml

from .arena_generation import install_arena_runtime
from .posttrainbench_inspect import (
    DEFAULT_PTB_ROOT,
    EXPECTED_PTB_COMMIT,
    _ensure_runtime,
    _verify_pins,
)


def _placeholder(value: object) -> bool:
    return isinstance(value, str) and value.startswith("<") and value.endswith(">")


def _general(config: dict, key: str, env: str, default: object = None):
    value = (config.get("general") or {}).get(key)
    if value is None or _placeholder(value):
        return os.environ.get(env, default)
    return value


def _custom(config: dict, key: str, default: object = None):
    return (config.get("custom") or {}).get(key, default)


def _load_official_module(benchmark_id: str):
    ptb_root = pathlib.Path(os.environ.get("POSTTRAINBENCH_ROOT", str(DEFAULT_PTB_ROOT)))
    task_dir = ptb_root / "src/eval/tasks" / benchmark_id
    evaluate = task_dir / "evaluate.py"
    if not evaluate.exists():
        raise FileNotFoundError(evaluate)
    # Both official judge tasks use the top-level package name
    # ``evaluation_code``. Clear a previously imported sibling task so a
    # multi-benchmark process cannot accidentally reuse the wrong package.
    for name in list(sys.modules):
        if name == "evaluation_code" or name.startswith("evaluation_code."):
            del sys.modules[name]
    sys.path.insert(0, str(task_dir))
    spec = importlib.util.spec_from_file_location(f"ptb_{benchmark_id}_evaluate", evaluate)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {evaluate}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, task_dir


def _normalize_none_message_content(response):
    """Make a null OpenAI message content score as an empty answer.

    The pinned Arena evaluator calls ``.strip()`` unconditionally. A reasoning
    parser can legally return ``content: null``; one such row must not abort the
    complete benchmark.
    """
    if not getattr(response, "ok", False):
        return response
    try:
        payload = response.json()
        message = (payload.get("choices") or [{}])[0].get("message") or {}
    except (AttributeError, IndexError, TypeError, ValueError):
        return response
    content = message.get("content")
    reasoning = message.get("reasoning_content")
    if content is None:
        message["content"] = ""
    elif content == "":
        if reasoning:
            if "</think>" in reasoning:
                message["content"] = reasoning.split("</think>", maxsplit=1)[1].strip()
            elif reasoning.startswith("<think>"):
                message["content"] = reasoning[len("<think>"):].strip()
            else:
                message["content"] = reasoning.strip()
        else:
            message["content"] = ""
    if content is None or content == "":
        response._content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return response


def _install_arena_response_guard(module):
    original_post = module.requests.Session.post

    def guarded_post(session, *args, **kwargs):
        return _normalize_none_message_content(
            original_post(session, *args, **kwargs)
        )

    module.requests.Session.post = guarded_post
    return original_post


def _apply_arena_judge_config(module, config: dict) -> None:
    """Apply repository config without modifying the pinned official evaluator."""
    judge_model = str(_custom(config, "judge_model", module.JUDGE_MODEL)).strip()
    if not judge_model:
        raise ValueError("custom.judge_model must not be empty")
    module.JUDGE_MODEL = judge_model

    reasoning_effort = _custom(config, "reasoning_effort", module.REASONING_EFFORT)
    if isinstance(reasoning_effort, str) and reasoning_effort.strip().lower() in {
        "",
        "none",
        "null",
    }:
        reasoning_effort = None
    module.REASONING_EFFORT = reasoning_effort

    if "OPENAI_BASE_URL" not in os.environ or not os.environ["OPENAI_BASE_URL"]:
        for candidate in [
            _custom(config, "judge_api_base"),
            os.environ.get("LLMROUTER_BASE_URL"),
            os.environ.get("OPENAI_API_BASE"),
        ]:
            if candidate and str(candidate).strip() and not _placeholder(candidate):
                cand_str = str(candidate).strip()
                if "127.0.0.1" not in cand_str and "localhost" not in cand_str:
                    os.environ["OPENAI_BASE_URL"] = cand_str
                    break
    if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
        for candidate in [
            _custom(config, "judge_api_token"),
            os.environ.get("LLMROUTER_API_KEY"),
        ]:
            if candidate and str(candidate).strip() and not _placeholder(candidate):
                os.environ["OPENAI_API_KEY"] = str(candidate).strip()
                break


def main(benchmark_id: str) -> None:
    _ensure_runtime()
    from dotenv import load_dotenv

    # Optional env file for judge credentials; unset means "rely on the process
    # environment" — no checkout path is hardcoded.
    _env_override = os.environ.get("POSTTRAINBENCH_ENV", "").strip()
    judge_env = pathlib.Path(_env_override) if _env_override else None
    if judge_env is not None and judge_env.is_file():
        load_dotenv(judge_env, override=False)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", default=None)
    args = parser.parse_args()
    config = yaml.safe_load(pathlib.Path(args.config).read_text()) or {}
    model = str(_general(config, "model", "MODEL_NAME", ""))
    base_url = str(_general(config, "openai_api_base", "OPENAI_API_BASE", ""))
    output_dir = pathlib.Path(str(_general(config, "output_dir", "OUTPUT_DIR", "output"))).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(base_url)
    if parsed.hostname not in {"127.0.0.1", "localhost", "0.0.0.0"} or not parsed.port:
        raise RuntimeError(
            f"official {benchmark_id} endpoint adapter requires a local endpoint, got {base_url}"
        )

    _verify_pins()
    module, task_dir = _load_official_module(benchmark_id)
    if benchmark_id == "arenahardwriting":
        _apply_arena_judge_config(module, config)
    port = parsed.port

    class ExistingEndpoint:
        def __init__(self, _args, _model_path):
            self.port = port

        def start(self):
            return self.port

        def stop(self, force: bool = False):
            return None

    module.VLLMServer = ExistingEndpoint
    metrics_path = output_dir / "official_metrics.json"
    limit = args.limit if args.limit is not None else -1
    max_tokens = int(_custom(config, "max_tokens", 16384))
    judge_workers = int(_custom(config, "judge_workers", 64))
    templates = pathlib.Path(os.environ.get("POSTTRAINBENCH_ROOT", str(DEFAULT_PTB_ROOT))) / "src/eval/templates"
    old_argv = sys.argv
    old_cwd = pathlib.Path.cwd()
    original_session_post = None
    restore_arena = None
    counts = None
    try:
        if benchmark_id == "arenahardwriting":
            original_session_post = _install_arena_response_guard(module)
            counts, restore_arena = install_arena_runtime(module, config, output_dir)
        os.chdir(task_dir)
        sys.argv = [
            str(task_dir / "evaluate.py"),
            "--model-path", model,
            "--max-new-tokens", str(max_tokens),
            "--limit", str(limit),
            "--judge-workers", str(judge_workers),
            "--templates-dir", str(templates),
            "--json-output-file", str(metrics_path),
        ]
        module.main()
    finally:
        if restore_arena is not None:
            restore_arena()
        if original_session_post is not None:
            module.requests.Session.post = original_session_post
        sys.argv = old_argv
        os.chdir(old_cwd)

    metrics = json.loads(metrics_path.read_text())
    accuracy = float(metrics["accuracy"])
    completed = int(metrics.get("n_examples") or (limit if limit >= 0 else 0))
    if counts is None:
        counts = {"attempted": completed, "completed": completed, "incomplete": 0}
    incomplete = counts["incomplete"]
    reward = {
        "reward": None if incomplete else accuracy,
        "meta_data": {
            "benchmark_id": benchmark_id,
            "protocol": "PostTrainBench official judge task via existing endpoint",
            "posttrainbench_commit": EXPECTED_PTB_COMMIT,
            "model": model,
            **counts,
            "accuracy": accuracy,
            "metrics": metrics,
        },
    }
    (output_dir / "reward.json").write_text(
        json.dumps(reward, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if incomplete:
        raise SystemExit(1)


__all__ = ["main"]
