#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests
import yaml

BENCHMARK_ID = "livecodebench"
EXPECTED_COMMIT = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
SPLIT_DATES = {
    "recent_2025": (datetime(2025, 1, 1), datetime(2025, 4, 30, 23, 59, 59)),
    "full_v6": (None, None),
}
SANDBOX_RESULT_PREFIX = "__LCB_RESULT__"


def placeholder(value):
    return isinstance(value, str) and value.startswith("<") and value.endswith(">")


def setting(config, section, key, env=None, default=None):
    value = (config.get(section) or {}).get(key)
    if value is None or placeholder(value):
        return os.environ.get(env, default) if env else default
    return value


def load_rows(path: Path, split: str) -> list[dict]:
    start, end = SPLIT_DATES[split]
    rows = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            date = datetime.fromisoformat(row["contest_date"])
            if start and date < start:
                continue
            if end and date > end:
                continue
            rows.append(row)
    return rows


def select_rows(rows: list[dict], limit: int | None) -> list[dict]:
    raw = os.environ.get("SAMPLE_INDICES")
    if raw:
        indices = [int(item) for item in raw.split(",")]
        if any(index >= len(rows) for index in indices):
            raise ValueError(f"sample index out of range for {len(rows)} rows")
        rows = [rows[index] for index in indices]
    return rows[:limit] if limit is not None else rows


def api_completion(base_url, token, model, messages, custom, session):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(custom["temperature"]),
        "max_tokens": int(custom["max_tokens"]),
        "n": int(custom["samples_per_problem"]),
    }
    if custom.get("enable_thinking") is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": bool(custom["enable_thinking"])}
    response = session.post(
        base_url.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {token}"},
        json=payload,
        timeout=float(custom["request_timeout"]),
    )
    response.raise_for_status()
    body = response.json()
    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError("completion response has no choices")
    outputs = []
    reasoning = []
    reasons = []
    for choice in choices:
        message = choice.get("message") or {}
        outputs.append(message.get("content") or "")
        reasoning.append(
            message.get("reasoning_content")
            or message.get("reasoning")
            or ""
        )
        reasons.append(choice.get("finish_reason"))
    return outputs, reasons, reasoning, body.get("usage") or {}


def sandbox_fusion_preflight(base_url: str, session=requests) -> None:
    response = session.get(base_url.rstrip("/") + "/v1/ping", timeout=10)
    response.raise_for_status()


def sandbox_evaluate_one(
    base_url: str,
    sample: dict,
    generation: str,
    testing_util_b64: str,
    timeout: int,
    session,
) -> tuple[list, dict]:
    in_outs = json.loads(sample["input_output"])
    run_timeout = (timeout + 1) * len(in_outs["inputs"]) + 5
    wrapper = (
        "import os\n"
        "os.environ['OPENBLAS_NUM_THREADS'] = '1'\n"
        "os.environ['OMP_NUM_THREADS'] = '1'\n"
        "os.environ['MKL_NUM_THREADS'] = '1'\n"
        "import json\n"
        "from testing_util import run_test\n"
        f"sample = {sample!r}\n"
        f"generation = {generation!r}\n"
        f"results, metadata = run_test(sample, test=generation, timeout={timeout})\n"
        "def normalize(value):\n"
        "    if isinstance(value, dict):\n"
        "        return {str(k): normalize(v) for k, v in value.items()}\n"
        "    if isinstance(value, (list, tuple)):\n"
        "        return [normalize(v) for v in value]\n"
        "    if hasattr(value, 'item'):\n"
        "        try:\n"
        "            return value.item()\n"
        "        except (TypeError, ValueError):\n"
        "            pass\n"
        "    return value\n"
        f"print({SANDBOX_RESULT_PREFIX!r} + json.dumps(normalize({{'results': results, 'metadata': metadata}}), default=str))\n"
    )
    response = session.post(
        base_url.rstrip("/") + "/run_code",
        json={
            "code": wrapper,
            "language": "python",
            "files": {"testing_util.py": testing_util_b64},
            "compile_timeout": timeout,
            "run_timeout": run_timeout,
            "memory_limit_MB": 4096,
        },
        timeout=run_timeout + 30,
    )
    response.raise_for_status()
    body = response.json()
    run_result = body.get("run_result") or {}
    if body.get("status") != "Success" or run_result.get("return_code") != 0:
        raise RuntimeError(
            f"Sandbox Fusion failed: status={body.get('status')!r}, "
            f"run_status={run_result.get('status')!r}, "
            f"return_code={run_result.get('return_code')!r}, "
            f"stderr={(run_result.get('stderr') or '')[-500:]!r}"
        )
    stdout = run_result.get("stdout") or ""
    marker = stdout.rfind(SANDBOX_RESULT_PREFIX)
    if marker < 0:
        raise RuntimeError("Sandbox Fusion response did not contain the LCB result marker")
    result_line = stdout[marker + len(SANDBOX_RESULT_PREFIX):].splitlines()[0]
    payload = json.loads(result_line)
    if not isinstance(payload.get("results"), list) or not isinstance(payload.get("metadata"), dict):
        raise RuntimeError("Sandbox Fusion returned a malformed LCB result")
    return payload["results"], payload["metadata"]


def sandbox_codegen_metrics(
    samples_list: list[dict],
    generations_list: list[list[str]],
    *,
    base_url: str,
    testing_util_b64: str,
    workers: int,
    timeout: int,
    compute_metrics,
):
    slots: dict[tuple[int, int], tuple[list, dict]] = {}
    jobs = []
    for problem_index, generations in enumerate(generations_list):
        for generation_index, generation in enumerate(generations):
            key = (problem_index, generation_index)
            if generation.strip():
                jobs.append((key, samples_list[problem_index], generation))
            else:
                slots[key] = ([-2], {
                    "error_code": -6,
                    "error_message": "NoCodeExtracted",
                    # The provider returned normally but the model did not
                    # produce executable code.  Official pass@1 counts this as
                    # a wrong answer; it is not a Sandbox infrastructure gap.
                    "execution_incomplete": False,
                    "model_failure_reason": "no_code_extracted",
                })

    local = threading.local()

    def evaluate(job):
        key, sample, generation = job
        if not hasattr(local, "session"):
            local.session = requests.Session()
        try:
            result = sandbox_evaluate_one(
                base_url, sample, generation, testing_util_b64, timeout, local.session
            )
        except Exception as exc:
            result = ([-5], {
                "error_code": -5,
                "error_message": "SandboxFusionError",
                "sandbox_error": f"{type(exc).__name__}: {exc}",
            })
        return key, result

    if jobs:
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(jobs)))) as pool:
            futures = [pool.submit(evaluate, job) for job in jobs]
            for future in as_completed(futures):
                key, result = future.result()
                slots[key] = result

    results = {}
    metadata = []
    for problem_index, generations in enumerate(generations_list):
        ordered = [slots[(problem_index, i)] for i in range(len(generations))]
        results[problem_index] = [item[0] for item in ordered]
        metadata.append([json.dumps(item[1], ensure_ascii=False) for item in ordered])
    return compute_metrics(results, k_list=[1]), results, metadata


def metadata_dict(value) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-data", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    bench_dir = Path(__file__).resolve().parent
    custom = dict(config.get("custom") or {})
    official_root = (bench_dir / custom["official_root"]).resolve()
    dataset_path = (bench_dir / custom["dataset_path"]).resolve()
    split = setting(config, "general", "split", "SPLIT_NAME", "recent_2025")
    if split not in SPLIT_DATES:
        raise SystemExit(f"unsupported split: {split}")
    if not official_root.exists() or not dataset_path.exists():
        raise SystemExit("missing pinned official checkout or local dataset")
    import subprocess
    commit = subprocess.check_output(["git", "-C", official_root, "rev-parse", "HEAD"], text=True).strip()
    if commit != EXPECTED_COMMIT:
        raise SystemExit(f"official commit mismatch: {commit}")
    sys.path.insert(0, str(official_root))
    # Upstream prompts/__init__.py eagerly imports Anthropic-only scenarios.
    # Code generation does not use that SDK, so provide only the two constants
    # those optional modules import instead of requiring an unrelated package.
    if "anthropic" not in sys.modules:
        anthropic_stub = types.ModuleType("anthropic")
        anthropic_stub.HUMAN_PROMPT = "\n\nHuman:"
        anthropic_stub.AI_PROMPT = "\n\nAssistant:"
        sys.modules["anthropic"] = anthropic_stub
    rows = load_rows(dataset_path, split)
    if args.dry_run:
        result = {"benchmark": BENCHMARK_ID, "split": split, "rows": len(rows), "official_commit": commit,
                  "dataset_release": custom["dataset_release"], "dataset_fingerprint": custom["dataset_fingerprint"]}
        if args.validate_data:
            result["first_question_id"] = rows[0]["question_id"]
        print(json.dumps(result, indent=2))
        return
    rows = select_rows(rows, args.limit)
    previous_cwd = Path.cwd()
    os.chdir(official_root)
    try:
        from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
        from lcb_runner.evaluation.pass_k_utils import compute_metrics_from_results
        from lcb_runner.lm_styles import LMStyle
        from lcb_runner.prompts.code_generation import format_prompt_generation
        from lcb_runner.utils.extraction_utils import extract_code
    finally:
        os.chdir(previous_cwd)

    sandbox_url = setting(config, "custom", "sandbox_fusion_url", "SANDBOX_FUSION_URL")
    if not sandbox_url:
        raise SystemExit("custom.sandbox_fusion_url or SANDBOX_FUSION_URL is required")
    sandbox_fusion_preflight(sandbox_url)
    testing_util_source = (
        official_root / "lcb_runner" / "evaluation" / "testing_util.py"
    ).read_bytes()
    testing_util_b64 = base64.b64encode(testing_util_source).decode("ascii")

    model = setting(config, "general", "model", "MODEL_NAME")
    base_url = setting(config, "general", "openai_api_base", "OPENAI_API_BASE")
    token = setting(config, "general", "openai_api_token", "OPENAI_API_TOKEN", "")
    output_dir = Path(setting(config, "general", "output_dir", "OUTPUT_DIR"))
    output_dir.mkdir(parents=True, exist_ok=True)
    problems = [CodeGenerationProblem(**row) for row in rows]
    lock = threading.Lock()
    local = threading.local()
    completed = 0
    started = time.time()

    def generate(index_problem):
        index, problem = index_problem
        if not hasattr(local, "session"):
            local.session = requests.Session()
        messages = format_prompt_generation(problem, LMStyle.OpenAIChat)
        outputs, reasons, reasoning, usage = api_completion(
            base_url, token, model, messages, custom, local.session
        )
        codes = [extract_code(output, LMStyle.OpenAIChat) for output in outputs]
        return index, messages, outputs, codes, reasons, reasoning, usage

    generated = [None] * len(problems)
    responses_path = output_dir / "responses.jsonl"
    with responses_path.open("w", encoding="utf-8") as sink, ThreadPoolExecutor(max_workers=int(custom["max_workers"])) as pool:
        futures = {
            pool.submit(generate, pair): pair[0]
            for pair in enumerate(problems)
        }
        for future in as_completed(futures):
            index = futures[future]
            request_error = None
            try:
                index, messages, outputs, codes, reasons, reasoning, usage = future.result()
            except Exception as exc:
                # Preserve the rest of the batch and its artifacts.  This
                # sample remains an infrastructure-incomplete attempt, so the
                # final reward is null and the runner exits non-zero.
                messages = format_prompt_generation(problems[index], LMStyle.OpenAIChat)
                count = max(1, int(custom["samples_per_problem"]))
                outputs = [""] * count
                codes = [""] * count
                reasons = [None] * count
                reasoning = [""] * count
                usage = {}
                request_error = f"{type(exc).__name__}: {exc}"
            generated[index] = codes
            record = {"index": index, "question_id": problems[index].question_id, "contest_date": problems[index].contest_date.isoformat(),
                      "prompt": messages, "outputs": outputs, "code_list": codes,
                      "finish_reasons": reasons, "reasoning_contents": reasoning,
                      "usage": usage, "request_error": request_error}
            with lock:
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                sink.flush()
                completed += 1
                print(f"generation {completed}/{len(problems)}", flush=True)

    samples = [problem.get_evaluation_sample() for problem in problems]
    metrics, results, metadata = sandbox_codegen_metrics(
        samples,
        generated,
        base_url=sandbox_url,
        testing_util_b64=testing_util_b64,
        workers=int(custom["execution_workers"]),
        timeout=int(custom["execution_timeout"]),
        compute_metrics=compute_metrics_from_results,
    )
    pass_at_1 = float(metrics["pass@1"])
    eval_rows = []
    for index, problem in enumerate(problems):
        passed = bool(results[index][0]) and all(value is True for value in results[index][0])
        eval_rows.append({"question_id": problem.question_id, "contest_date": problem.contest_date.isoformat(),
                          "difficulty": problem.difficulty.value, "passed": passed, "result": results[index][0],
                          "metadata": metadata[index][0]})
    (output_dir / "eval_all.json").write_text(json.dumps(eval_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    # 合并生成轨迹与官方判定 → samples.jsonl(与 aime/horizonmath 同形的逐样本轨迹).
    # responses.jsonl 是按完成顺序写的, 用 index 定位回题目; 每题取第一个被评分的
    # sample(pass@1 只评 samples[0]), 同时保留完整 outputs/code_list 供复查.
    resp_by_index = {row["index"]: row for row in (
        json.loads(line) for line in responses_path.read_text(encoding="utf-8").splitlines() if line.strip()
    )}
    samples = []
    for index, problem in enumerate(problems):
        gen = resp_by_index.get(index, {})
        messages = gen.get("prompt") or []
        outputs = gen.get("outputs") or []
        codes = gen.get("code_list") or []
        reasons = gen.get("finish_reasons") or []
        reasoning = gen.get("reasoning_contents") or []
        usage = gen.get("usage") or {}
        request_error = gen.get("request_error")
        result = results[index][0] if index < len(results) else None
        result_metadata = metadata_dict(metadata[index][0]) if index < len(metadata) else {}
        passed = bool(result) and all(value is True for value in result)
        prompt_text = "\n".join(
            m.get("content", "") for m in messages
            if isinstance(m, dict) and m.get("role") == "user"
        )
        response = outputs[0] if outputs else ""
        code = codes[0] if codes else ""
        if request_error:
            incomplete_reason = "request_error"
        elif result_metadata.get("sandbox_error"):
            incomplete_reason = "sandbox_error"
        else:
            incomplete_reason = None
        if incomplete_reason is not None or passed:
            model_failure_reason = None
        elif not response:
            model_failure_reason = "empty_generation"
        elif not code:
            model_failure_reason = "no_code_extracted"
        elif reasons and reasons[0] == "length":
            model_failure_reason = "length_truncated"
        else:
            model_failure_reason = "wrong_answer"
        samples.append({
            "index": index,
            "question_id": problem.question_id,
            "contest_date": problem.contest_date.isoformat(),
            "difficulty": problem.difficulty.value,
            "prompt": messages,
            "prompt_text": prompt_text,
            "response": response,
            "reasoning_content": reasoning[0] if reasoning else "",
            "code": code,
            "finish_reason": reasons[0] if reasons else None,
            "usage": usage,
            "outputs": outputs,
            "code_list": codes,
            "passed": passed,
            "result": result,
            "metadata": metadata[index][0] if index < len(metadata) else None,
            "correct": passed,
            "request_complete": request_error is None,
            "complete": incomplete_reason is None,
            "incomplete_reason": incomplete_reason,
            "model_failure_reason": model_failure_reason,
            "error": request_error,
        })
    (output_dir / "samples.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in samples), encoding="utf-8")
    completed = sum(sample["complete"] for sample in samples)
    incomplete = len(samples) - completed
    correct = sum(sample["passed"] for sample in samples)
    reward = {"reward": pass_at_1 if incomplete == 0 else None, "meta_data": {
              "accuracy": pass_at_1, "pass@1": pass_at_1, "observed_reward": pass_at_1,
              "attempted": len(problems), "completed": completed, "incomplete": incomplete,
              "correct": correct, "split": split, "dataset_release": custom["dataset_release"],
              "official_commit": commit, "model": model, "samples_per_problem": int(custom["samples_per_problem"]),
              "temperature": float(custom["temperature"]), "enable_thinking": bool(custom["enable_thinking"]),
              "execution_backend": "sandbox_fusion", "elapsed_seconds": time.time() - started}}
    (output_dir / "reward.json").write_text(json.dumps(reward, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(reward, ensure_ascii=False, indent=2))
    if incomplete:
        raise RuntimeError(
            f"LiveCodeBench evaluation incomplete: {incomplete}/{len(problems)} samples; "
            "artifacts were written but the run is not successful"
        )


if __name__ == "__main__":
    main()
