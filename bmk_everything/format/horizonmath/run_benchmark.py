#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "framework"))
from core.subset import select_subset, write_subset_manifest

BENCHMARK_ID = "horizonmath"
EXPECTED_COMMIT = "a34c47278697fda59d566ba11b2087da1bcf4454"
MODE_FOR_SPLIT = {
    "numeric": "ground_truth_computable",
    "benchmark": "benchmark_best_known",
    "construction": "new_construction",
}


def placeholder(value):
    return isinstance(value, str) and value.startswith("<") and value.endswith(">")


def general(config, key, env, default=None):
    value = (config.get("general") or {}).get(key)
    if value is None or placeholder(value):
        return os.environ.get(env, default)
    return value


def custom(config, key, default=None):
    return (config.get("custom") or {}).get(key, default)


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def resolve_max_output_tokens(config: dict) -> int:
    """Resolve the generated-token budget without silently ignoring aliases.

    Transformers calls this value ``max_new_tokens`` while the OpenAI chat API
    calls it ``max_tokens``.  HorizonMath accepts either config spelling but
    sends the resolved value using the OpenAI field name.
    """
    options = config.get("custom") or {}
    max_tokens = options.get("max_tokens")
    max_new_tokens = options.get("max_new_tokens")
    if max_tokens is not None and max_new_tokens is not None:
        if int(max_tokens) != int(max_new_tokens):
            raise ValueError(
                "custom.max_tokens and custom.max_new_tokens disagree: "
                f"{max_tokens} != {max_new_tokens}"
            )
    value = max_new_tokens if max_new_tokens is not None else max_tokens
    value = 8192 if value is None else int(value)
    if value <= 0:
        raise ValueError(f"max output tokens must be positive, got {value}")
    return value


def compliance_environment(
    config: dict,
    fallback_token: str = "",
) -> dict[str, str]:
    """Build the environment for the pinned HorizonMath compliance judge.

    Candidate generation uses the experiment's OpenAI-compatible endpoint.
    Compliance review is a separate request and should use the stable internal
    judge configured for BmkEverything, without hard-coding credentials.
    """
    env = os.environ.copy()
    provider = str(custom(config, "compliance_provider", "")).strip()
    model = str(custom(config, "compliance_model", "")).strip()
    api_base = str(custom(config, "compliance_api_base", "")).strip()
    token_env = str(
        custom(config, "compliance_api_token_env", "OPENAI_API_KEY")
    ).strip()
    reasoning_effort = str(
        custom(config, "compliance_reasoning_effort", "high")
    ).strip()

    if provider:
        env["COMPLIANCE_PROVIDER"] = provider
    if model:
        env["COMPLIANCE_MODEL"] = model
    if reasoning_effort:
        env["COMPLIANCE_REASONING_EFFORT"] = reasoning_effort
    if api_base:
        env["OPENAI_BASE_URL"] = api_base
    token = env.get(token_env, "") if token_env else ""
    if not token:
        token = fallback_token
    if token:
        env["OPENAI_API_KEY"] = token
    return env


def load_official_system_messages(root: Path) -> dict[str, str]:
    """Evaluate only the two prompt-constant assignments from the pinned runner."""
    source = (root / "scripts/run_benchmark.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    selected = []
    wanted = {"_SYSTEM_MESSAGE_BASE", "SYSTEM_MESSAGES"}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in wanted for target in node.targets
        ):
            selected.append(node)
    namespace: dict[str, object] = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(root), "exec"), {}, namespace)
    messages = namespace.get("SYSTEM_MESSAGES")
    if not isinstance(messages, dict) or set(messages) != {
        "ground_truth_computable", "benchmark_best_known", "new_construction"
    }:
        raise RuntimeError("could not load official HorizonMath system messages")
    return messages


def _select_problems_with_manifest(problems: list[dict], split: str, limit: int | None, fingerprint: str = EXPECTED_COMMIT):
    indexed = list(enumerate(problems))
    if split == "calibration":
        indexed = [(i, p) for i, p in indexed if p.get("solvability") == 0]
    elif split in MODE_FOR_SPLIT:
        mode = MODE_FOR_SPLIT[split]
        indexed = [(i, p) for i, p in indexed if p.get("evaluation_mode") == mode]
    elif split != "default":
        raise ValueError(f"unknown split: {split}")
    selection = select_subset(
        problems,
        limit=limit,
        fingerprint=f"{fingerprint}:{split}",
        stable_key=lambda problem: problem["id"],
        stratum=lambda problem: problem.get("evaluation_mode", "unknown"),
        eligible_indices=[index for index, _ in indexed],
    )
    return selection.rows, selection.manifest


def select_problems(problems: list[dict], split: str, limit: int | None) -> list[tuple[int, dict]]:
    """Compatibility wrapper retained for callers and benchmark tests."""
    selected, _ = _select_problems_with_manifest(problems, split, limit)
    return selected


def call_model(task, base_url, model, token, max_tokens, temperature, timeout, retries,
               enable_thinking):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": task["system"]},
            {"role": "user", "content": task["problem"]["prompt"]},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    error = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(
                base_url.rstrip("/") + "/chat/completions",
                headers={"Authorization": f"Bearer {token}"},
                json=payload,
                timeout=timeout,
            )
            if not response.ok:
                raise RuntimeError(
                    f"HTTP {response.status_code}: {response.text[:1000]}"
                )
            body = response.json()
            message = body["choices"][0]["message"]
            content = message.get("content") or ""
            if not content.strip():
                raise RuntimeError("empty completion")
            return {
                "problem_id": task["problem"]["id"],
                "problem_index": task["index"],
                "title": task["problem"]["id"],
                "provider": "openai-compatible",
                "model": model,
                "response": content,
                "reasoning_content": message.get("reasoning_content") or "",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "usage": body.get("usage"),
                "finish_reason": body["choices"][0].get("finish_reason"),
            }, None
        except Exception as exc:
            error = exc
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
    return None, {
        "problem_id": task["problem"]["id"],
        "problem_index": task["index"],
        "mode": "runtime",
        "solvability": task["problem"].get("solvability"),
        "error_type": "runtime",
        "error_message": f"{type(error).__name__}: {error}",
    }


def _official_passed(eval_row: dict) -> bool:
    """Mirror ``evaluate_responses.py`` per-mode passed semantics.

    - numeric(ground_truth_computable): passed = success(20 位精确匹配)
    - construction(new_construction):  passed = valid
    - benchmark(benchmark_best_known): passed = valid 且 beats_baseline / no_baseline
      (matches_baseline / below_baseline 不算 passed —— 别用 valid 兜底, 否则会把
      「格式合法但没达到已知最优」的 benchmark 行错标成 correct=True)
    - runtime 等: False
    """
    mode = eval_row.get("mode")
    if mode is None:
        if "baseline_comparison" in eval_row:
            if not eval_row.get("valid"):
                return False
            return (eval_row.get("baseline_comparison") or {}).get("result") in (
                "beats_baseline", "no_baseline"
            )
        if "success" in eval_row:
            return bool(eval_row.get("success"))
        if "valid" in eval_row:
            return bool(eval_row.get("valid"))
    if mode == "numeric":
        return bool(eval_row.get("success"))
    if mode == "construction":
        return bool(eval_row.get("valid"))
    if mode == "benchmark":
        if not eval_row.get("valid"):
            return False
        bc = eval_row.get("baseline_comparison") or {}
        return bc.get("result") in ("beats_baseline", "no_baseline")
    return False


def _model_failure_reason(generation: dict, evaluation: dict) -> str | None:
    if not generation:
        return "missing_generation"
    if str(generation.get("finish_reason") or "").lower() == "length":
        return "length_truncated"
    error_type = str(evaluation.get("error_type") or "").lower()
    if error_type in {"extraction", "execution", "runtime"}:
        return f"{error_type}_failure"
    if evaluation.get("valid") is False:
        return "official_invalid"
    if evaluation.get("mode") == "benchmark" and not _official_passed(evaluation):
        return "baseline_not_beaten"
    return None


def _failure_origin(generation: dict, evaluation: dict, failure_reason: str | None) -> str | None:
    if failure_reason is None:
        return None
    if not generation:
        message = str(evaluation.get("error_message") or "").lower()
        if message and "empty completion" not in message:
            return "infrastructure"
    return "model"


def merge_sample_rows(
    responses: list[dict], evaluations: list[dict], prompts: list[dict] | None = None
) -> list[dict]:
    """Combine generation traces and official HorizonMath judgements.

    MetaMaster treats ``samples.jsonl`` as the durable per-sample trace.  The
    official evaluator only writes judgement fields to ``evaluation.jsonl``,
    so preserve those fields while adding the original model output and the
    common fields consumed by MetaMaster's reviewers.
    """
    response_by_key = {
        (row.get("problem_id"), row.get("problem_index")): row for row in responses
    }
    prompt_by_key = {
        (row.get("problem_id"), row.get("problem_index")): row for row in (prompts or [])
    }
    samples = []
    for evaluation in evaluations:
        key = (evaluation.get("problem_id"), evaluation.get("problem_index"))
        generation = response_by_key.get(key, {})
        prompt = prompt_by_key.get(key, {})
        row = dict(evaluation)
        failure_reason = _model_failure_reason(generation, evaluation)
        failure_origin = _failure_origin(generation, evaluation, failure_reason)
        correct = _official_passed(evaluation)
        row.update({
            "question": prompt.get("prompt"),
            "system_prompt": prompt.get("system"),
            "response": generation.get("response", ""),
            "reasoning_content": generation.get("reasoning_content", ""),
            "finish_reason": generation.get("finish_reason"),
            "usage": generation.get("usage"),
            "provider": generation.get("provider"),
            "model": generation.get("model"),
            "timestamp": generation.get("timestamp"),
            "correct": correct,
            "score": 1 if correct else 0,
            "complete": bool(generation),
            "model_output_valid": failure_reason is None,
            "failure_origin": failure_origin,
            "failure_reason": failure_reason,
            "incomplete_reason": None if generation else "missing_generation",
            "predicted": evaluation.get("actual_value"),
            "target": evaluation.get("expected_value"),
        })
        samples.append(row)
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-data", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    root = Path(os.environ.get("HORIZONMATH_ROOT") or custom(config, "horizonmath_root"))
    if not root.is_absolute():
        root = REPO_ROOT / root
    expected = str(custom(config, "official_commit", EXPECTED_COMMIT))
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if actual != expected:
        raise RuntimeError(f"HorizonMath commit mismatch: expected {expected}, got {actual}")
    problems = json.loads((root / "data/problems_full.json").read_text(encoding="utf-8"))
    split = args.split or str(general(config, "split", "SPLIT_NAME", "default"))
    selected, subset_manifest = _select_problems_with_manifest(problems, split, args.limit, actual)
    if args.dry_run:
        result = {"benchmark": BENCHMARK_ID, "official_commit": actual, "split": split, "count": len(selected)}
        if args.validate_data and selected:
            result["first"] = {k: selected[0][1].get(k) for k in ("id", "evaluation_mode", "solvability")}
        print(json.dumps(result, indent=2))
        return

    model = str(general(config, "model", "MODEL_NAME", ""))
    base_url = str(general(config, "openai_api_base", "OPENAI_API_BASE", ""))
    token = str(general(config, "openai_api_token", "OPENAI_API_TOKEN", "sk-123"))
    output_dir = Path(str(general(config, "output_dir", "OUTPUT_DIR", "output")))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_subset_manifest(output_dir, subset_manifest)
    max_output_tokens = resolve_max_output_tokens(config)
    system_messages = load_official_system_messages(root)
    tasks = [
        {"index": index, "problem": problem, "system": system_messages[problem["evaluation_mode"]]}
        for index, problem in selected
    ]
    run_config = {
        "run_id": os.environ.get("TRIAL_NAME", output_dir.name),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "provider": "openai-compatible",
        "model": model,
        "problems_file": "data/problems_full.json",
        "output_dir": str(output_dir),
        "official_commit": actual,
        "split": split,
        "max_output_tokens": max_output_tokens,
    }
    (output_dir / "config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    with (output_dir / "prompts.jsonl").open("w", encoding="utf-8") as stream:
        for task in tasks:
            stream.write(json.dumps({
                "problem_id": task["problem"]["id"], "problem_index": task["index"],
                "system": task["system"], "prompt": task["problem"]["prompt"],
            }) + "\n")

    responses_path = output_dir / "responses.jsonl"
    errors_path = output_dir / "evaluation.jsonl"
    write_lock = threading.Lock()
    max_workers = int(custom(config, "max_workers", 4))
    kwargs = dict(
        base_url=base_url, model=model, token=token,
        max_tokens=max_output_tokens,
        temperature=float(custom(config, "temperature", 0.0)),
        timeout=int(custom(config, "timeout", 1800)),
        retries=int(custom(config, "empty_retries", 1)),
        enable_thinking=env_bool(
            "HORIZONMATH_ENABLE_THINKING",
            bool(custom(config, "enable_thinking", True)),
        ),
    )
    completed = 0
    with responses_path.open("w", encoding="utf-8") as responses, errors_path.open("w", encoding="utf-8") as errors:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(call_model, task, **kwargs): task for task in tasks}
            for future in as_completed(futures):
                response, error = future.result()
                with write_lock:
                    if response:
                        responses.write(json.dumps(response, ensure_ascii=False) + "\n")
                        responses.flush()
                    else:
                        errors.write(json.dumps(error, ensure_ascii=False) + "\n")
                        errors.flush()
                    completed += 1
                    print(f"generation {completed}/{len(tasks)}", flush=True)

    python = root / ".venv/bin/python"
    command = [str(python), "scripts/evaluate_responses.py", str(output_dir), "--force"]
    subprocess.run(
        command,
        cwd=root,
        check=True,
        env=compliance_environment(config, fallback_token=token),
    )
    summary = json.loads((output_dir / "summary.json").read_text())
    responses = [json.loads(line) for line in responses_path.read_text().splitlines() if line.strip()]
    evaluations = [json.loads(line) for line in errors_path.read_text().splitlines() if line.strip()]
    prompts = [
        json.loads(line)
        for line in (output_dir / "prompts.jsonl").read_text().splitlines()
        if line.strip()
    ]
    samples = merge_sample_rows(responses, evaluations, prompts)
    (output_dir / "samples.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in samples), encoding="utf-8"
    )
    total = int(summary["total_problems"])
    passed = int(summary["passed"])
    score = passed / total if total else 0.0
    # completed = 实际非空生成数, incomplete = 空/失败生成数 —— 不能硬编码 total,
    # 否则 113 次 empty completion 会被当成「完成且得 0 分」, 掩盖协议/基础设施故障.
    # 注意: evaluation.jsonl 会被官方 evaluator 重写成全部题的判定, 不能当 error 计数用;
    # 只有 responses.jsonl 的条数 = 非空生成数, 所以 incomplete = total - 非空生成数.
    n_responses = len(responses)
    n_errors = total - n_responses
    length_truncated = sum(row.get("failure_reason") == "length_truncated" for row in samples)
    model_invalid = sum(not row.get("model_output_valid", False) for row in samples)
    infra_failures = sum(row.get("failure_origin") == "infrastructure" for row in samples)
    reward = {
        "reward": score,
        "meta_data": {
            "benchmark_id": BENCHMARK_ID, "official_commit": actual,
            "attempted": total, "completed": n_responses, "incomplete": n_errors,
            "correct": passed, "accuracy": score,
            "model_invalid": model_invalid,
            "length_truncated": length_truncated,
            "infra_failures": infra_failures,
            "evaluation_status": "ok" if infra_failures == 0 else "degraded",
            "official_comparable": infra_failures == 0,
            "scoring_supported": True, "official_summary": summary,
        },
    }
    (output_dir / "reward.json").write_text(json.dumps(reward, indent=2) + "\n")
    print(json.dumps(reward, indent=2))


if __name__ == "__main__":
    main()
