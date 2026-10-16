#!/usr/bin/env python3
"""Full HLE multimodal judge runner for BmkEverything."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
import yaml
from tqdm import tqdm


BENCHMARK_ID = "hle"
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_SHA256 = "4663364783399ffce168dbc8a56bc670e498f299fb64b631f322b8847f5d327a"
DATASET_ROWS = 2158
DATASET_REVISION = "bdytx5/hle:test"
DEFAULT_OPENAI_BASE = "https://api.moonshot.cn/v1"
DEFAULT_GRADER_MODEL = "deepseek-v4-flash"
DEFAULT_GRADER_BASE = (
    "https://your-judge.example.com/v1"
)

SPLIT_TO_CATEGORY = {
    "default": None,
    "math": "Math",
    "physics": "Physics",
    "bio-medicine": "Biology/Medicine",
    "cs-ai": "Computer Science/AI",
    "engineering": "Engineering",
    "chemistry": "Chemistry",
    "humanities-social-science": "Humanities/Social Science",
    "other": "Other",
}

JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not
based on the precise and unambiguous [correct_answer] below.

[question]: <<question>>

[response]: <<response>>

[correct_answer]: <<correct_answer>>

Your judgement must be in the following JSON format:
{
  "extracted_final_answer": "The final exact answer extracted from the response, or null",
  "reasoning": "Why the extracted final answer is correct or incorrect",
  "correct": "yes or no",
  "confidence": "An integer from 0 to 100"
}

Rules:
- Extract the final answer from the response, not an intermediate answer.
- Return null if there is no exact, final answer to extract.
- Mark correct as yes if the extracted answer matches the correct answer,
  including equivalent mathematical forms and a small margin of error for numerical problems.
- Mark correct as no if there is any meaningful inconsistency, ambiguity, or non-equivalence.
- Do not solve the problem from scratch or argue for an answer different from correct_answer.
- Focus only on whether the extracted final answer matches correct_answer.

Return only JSON."""


def is_placeholder(value: object) -> bool:
    return isinstance(value, str) and value.startswith("<") and value.endswith(">")


def resolve_general(config: dict, key: str, env: str, default=None):
    value = (config.get("general") or {}).get(key)
    if value is None or is_placeholder(value):
        return os.environ.get(env) or default
    return value


def resolve_custom(config: dict, key: str, default=None):
    if key == "grader_api_base" and os.environ.get("JUDGE_OPENAI_API_BASE"):
        return os.environ["JUDGE_OPENAI_API_BASE"]
    if key == "grader_model" and os.environ.get("JUDGE_MODEL"):
        return os.environ["JUDGE_MODEL"]
    return (config.get("custom") or {}).get(key, default)


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def resolve_output_dir(config: dict, trial_name: str) -> Path:
    template = resolve_general(config, "output_dir", "OUTPUT_DIR", "")
    if not template:
        template = "data/output/<BENCHMARK_ID>/<TRIAL_NAME>"
    value = (
        template.replace("<BENCHMARK_ID>", BENCHMARK_ID)
        .replace("<TRIAL_NAME>", trial_name)
    )
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def resolved_config(config: dict, model: str, split: str, output_dir: Path) -> dict:
    result = copy.deepcopy(config)
    result.setdefault("general", {}).update(
        {
            "model": model,
            "split": split,
            "openai_api_base": resolve_general(
                config, "openai_api_base", "OPENAI_API_BASE", DEFAULT_OPENAI_BASE
            ),
            "openai_api_token": "<OPENAI_API_TOKEN>",
            "anthropic_api_base": resolve_general(
                config,
                "anthropic_api_base",
                "ANTHROPIC_API_BASE",
                "https://api.anthropic.com/v1",
            ),
            "anthropic_api_token": "<ANTHROPIC_API_TOKEN>",
            "output_dir": str(output_dir),
        }
    )
    return result


def build_prompt(row: dict):
    question = str(row["question"]).strip()
    if row.get("answer_type") == "multipleChoice":
        instruction = "Return only the single correct option letter as 'ANSWER: X'."
    else:
        instruction = "Return only the final answer, prefixed with 'ANSWER:'."
    image = str(row.get("image") or "").strip()
    if not image:
        return f"{question}\n\n{instruction}"
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{question}\n\n{instruction}"},
                {"type": "image_url", "image_url": {"url": image}},
            ],
        }
    ]

def build_judge_prompt(row: dict, response: str) -> str:
    prompt = JUDGE_PROMPT
    prompt = prompt.replace("<<question>>", str(row["question"]).strip())
    prompt = prompt.replace("<<response>>", response.strip())
    prompt = prompt.replace("<<correct_answer>>", str(row["answer"]).strip())
    prompt = prompt.replace("<<answer_type>>", str(row.get("answer_type") or "unknown"))
    return prompt


def resolve_openai_token(config: dict) -> str:
    token = resolve_general(config, "openai_api_token", "OPENAI_API_TOKEN", "")
    if token:
        return token
    base = resolve_general(
        config, "openai_api_base", "OPENAI_API_BASE", DEFAULT_OPENAI_BASE
    ).rstrip("/")
    if base == DEFAULT_OPENAI_BASE.rstrip("/"):
        return os.environ.get("KIMI_MOONSHOT_API_KEY", "")
    if base == "https://kspmas.ksyun.com/v1":
        return os.environ.get("KIMI_KSYUN_API_KEY", "")
    return ""


def resolve_grader_token(config: dict) -> str:
    env_name = str(resolve_custom(config, "grader_api_token_env", "OPENAI_API_TOKEN"))
    return os.environ.get(env_name, "")


def is_context_overflow_response(response: requests.Response) -> bool:
    if response.status_code != 400:
        return False
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    error = payload.get("error") if isinstance(payload, dict) else None
    message = error.get("message", "") if isinstance(error, dict) else ""
    normalized = str(message).lower()
    return "maximum context length" in normalized and "output tokens" in normalized


def is_aborted(finish_reason: str | None) -> bool:
    return str(finish_reason or "").lower() == "abort"


def chat_completion(
    messages: list[dict],
    *,
    base_url: str,
    token: str,
    model: str,
    max_tokens: int,
    fallback_max_tokens: list[int] | None = None,
    temperature: float,
    timeout: int,
    retries: int = 3,
) -> dict:
    budgets = [max_tokens]
    for value in fallback_max_tokens or []:
        budget = int(value)
        if 0 < budget < max_tokens and budget not in budgets:
            budgets.append(budget)
    last_error: Exception | None = None
    total_attempts = 0
    requested_max_tokens = max_tokens
    for budget in budgets:
        requested_max_tokens = budget
        context_overflow = False
        for attempt in range(1, retries + 1):
            total_attempts += 1
            response = None
            try:
                response = requests.post(
                    f"{base_url.rstrip('/')}/chat/completions",
                    json={
                        "model": model,
                        "messages": messages,
                        "max_tokens": budget,
                        "temperature": temperature,
                    },
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    timeout=timeout,
                )
                response.raise_for_status()
                body = response.json()
                choices = body.get("choices") or []
                if not choices:
                    raise ValueError("response missing choices")
                message = choices[0].get("message") or {}
                content = message.get("content") or ""
                if isinstance(content, list):
                    content = "".join(
                        str(item.get("text") or "")
                        for item in content
                        if isinstance(item, dict)
                    )
                content = str(content).strip()
                reasoning = (
                    message.get("reasoning_content")
                    or message.get("reasoning")
                    or message.get("reasoning_text")
                    or ""
                )
                if not content and reasoning:
                    content = str(reasoning).strip()
                if not content:
                    raise ValueError("empty response content")
                finish_reason = choices[0].get("finish_reason")
                if is_aborted(finish_reason):
                    raise RuntimeError("generation aborted before completion")
                return {
                    "content": content,
                    "raw_content": content,
                    "reasoning_content": reasoning,
                    "finish_reason": finish_reason,
                    "usage": body.get("usage") or {},
                    "provider": "openai-compatible",
                    "complete": True,
                    "error": None,
                    "attempts": total_attempts,
                    "requested_max_tokens": budget,
                }
            except Exception as exc:
                last_error = exc
                if response is not None and is_context_overflow_response(response):
                    context_overflow = True
                    break
                if attempt < retries:
                    time.sleep(2**attempt)
        if not context_overflow:
            break
    return {
        "content": "",
        "raw_content": "",
        "reasoning_content": "",
        "finish_reason": None,
        "usage": {},
        "provider": "openai-compatible",
        "complete": False,
        "error": f"{type(last_error).__name__}: {last_error}" if last_error else "request failed",
        "attempts": total_attempts,
        "requested_max_tokens": requested_max_tokens,
    }


def is_truncated(finish_reason: str | None, usage: dict | None, max_tokens: int) -> bool:
    if str(finish_reason or "").lower() in {"length", "max_tokens", "max_output_tokens"}:
        return True
    if not usage:
        return False
    completion_tokens = usage.get("completion_tokens")
    return isinstance(completion_tokens, int) and completion_tokens >= max_tokens


def record_candidate_is_truncated(record: dict) -> bool:
    candidate = record.get("candidate") or {}
    return bool(candidate.get("truncated")) or is_truncated(
        candidate.get("finish_reason"),
        candidate.get("usage") or {},
        int(candidate.get("requested_max_tokens") or 2**31),
    )


def scored_outcome(candidate: dict, judge: dict, truncated: bool) -> tuple[bool, bool | None]:
    if truncated:
        return True, False
    complete = bool(candidate.get("complete") and judge.get("complete"))
    return complete, judge.get("correct") if complete else None


def summarize_records(records: list[dict]) -> dict:
    """Return a numeric lower-bound score without hiding evaluation failures.

    Model-originated invalid outputs are ordinary zero-score attempts.  Candidate
    transport and Judge failures also contribute zero to the lower-bound score,
    but are reported separately so the run is not presented as officially
    comparable.
    """
    attempted = len(records)
    completed = sum(1 for record in records if record.get("complete"))
    correct = sum(1 for record in records if record.get("correct") is True)
    candidate_failures = sum(
        1 for record in records if not record.get("candidate", {}).get("complete")
    )
    judge_failures = sum(
        1
        for record in records
        if record.get("candidate", {}).get("complete")
        and not record.get("candidate", {}).get("truncated")
        and not record.get("judge", {}).get("complete")
    )
    infra_failures = candidate_failures + judge_failures
    return {
        "attempted": attempted,
        "completed": completed,
        "incomplete": attempted - completed,
        "correct": correct,
        "accuracy": correct / attempted if attempted else 0.0,
        "observed_accuracy": correct / completed if completed else 0.0,
        "candidate_failures": candidate_failures,
        "judge_failures": judge_failures,
        "infra_failures": infra_failures,
        "evaluation_status": "ok" if infra_failures == 0 else "degraded",
        "official_comparable": infra_failures == 0,
    }


def load_candidate_checkpoint(
    path: Path,
    rows: list[dict],
    signature: dict,
) -> list[dict | None]:
    results: list[dict | None] = [None] * len(rows)
    if not path.exists():
        return results
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("signature") != signature:
                raise RuntimeError(
                    f"candidate checkpoint signature mismatch at line {line_number}"
                )
            index = record.get("index")
            if not isinstance(index, int) or index < 0 or index >= len(rows):
                raise RuntimeError(
                    f"candidate checkpoint index is invalid at line {line_number}"
                )
            if record.get("id") != rows[index].get("id"):
                raise RuntimeError(
                    f"candidate checkpoint row mismatch at index {index}"
                )
            result = record.get("result")
            if not isinstance(result, dict):
                raise RuntimeError(
                    f"candidate checkpoint result is invalid at line {line_number}"
                )
            results[index] = None if is_aborted(result.get("finish_reason")) else result
    return results


def candidate_from_record(record: dict) -> dict:
    candidate = record.get("candidate") or {}
    return {
        "index": record.get("index"),
        "id": record.get("id"),
        "category": record.get("category"),
        "answer_type": record.get("answer_type"),
        "prompt": record.get("prompt"),
        "answer": record.get("answer"),
        "response": candidate.get("content", ""),
        "reasoning_content": candidate.get("reasoning_content", ""),
        "finish_reason": candidate.get("finish_reason"),
        "usage": candidate.get("usage") or {},
        "complete": bool(candidate.get("complete")),
        "error": candidate.get("error"),
        "provider": candidate.get("provider") or "openai-compatible",
        "latency_s": candidate.get("latency_s", 0.0),
        "retries": candidate.get("retries", 0),
        "requested_max_tokens": candidate.get("requested_max_tokens"),
    }


def judge_from_record(record: dict) -> dict:
    judge = record.get("judge") or {}
    return {
        "index": record.get("index"),
        "complete": bool(judge.get("complete")),
        "correct": record.get("correct") if judge.get("complete") else None,
        "parsed": judge.get("parsed") or {},
        "response": judge.get("content", ""),
        "reasoning": judge.get("reasoning", ""),
        "confidence": judge.get("confidence"),
        "provider": judge.get("provider"),
        "error": judge.get("error"),
        "latency_s": judge.get("latency_s", 0.0),
        "retries": judge.get("retries", 0),
        "requested_max_tokens": judge.get("requested_max_tokens"),
    }


def load_resume_state(
    path: Path,
    rows: list[dict],
    preserve_truncated: bool = False,
) -> tuple[dict[int, dict], list[dict | None], list[dict | None]]:
    with path.open(encoding="utf-8") as source:
        previous_rows = [json.loads(line) for line in source if line.strip()]
    if len(previous_rows) != len(rows):
        raise RuntimeError("resume samples and dataset have different row counts")
    preserved_records: dict[int, dict] = {}
    candidate_results: list[dict | None] = [None] * len(rows)
    judge_results: list[dict | None] = [None] * len(rows)
    for index, (row, record) in enumerate(zip(rows, previous_rows)):
        if record.get("index") != index or record.get("id") != row.get("id"):
            raise RuntimeError(f"resume row mismatch at index {index}")
        candidate = candidate_from_record(record)
        aborted = is_aborted(candidate.get("finish_reason"))
        truncated = record_candidate_is_truncated(record)
        if (
            candidate.get("complete")
            and not aborted
            and (not truncated or preserve_truncated)
        ):
            candidate_results[index] = candidate
        judge = judge_from_record(record)
        if (
            record.get("complete")
            and candidate.get("complete")
            and not aborted
            and (
                (not truncated and judge.get("complete"))
                or (truncated and preserve_truncated)
            )
        ):
            preserved_records[index] = record
            judge_results[index] = judge
    return preserved_records, candidate_results, judge_results


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    if not cleaned:
        return {}
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}
    return {}


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "correct", "1"}:
            return True
        if lowered in {"false", "no", "incorrect", "0"}:
            return False
    return None


def parse_confidence(value: Any) -> int:
    try:
        confidence = int(float(value))
    except Exception:
        confidence = 100
    return max(0, min(100, confidence))


def load_rows(dataset_path: Path, expected_sha: str, split: str, limit: int | None) -> list[dict]:
    category = SPLIT_TO_CATEGORY[split]
    data_file = dataset_path / "default.jsonl"
    data = data_file.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if sha != expected_sha:
        raise RuntimeError(f"dataset hash mismatch: expected {expected_sha}, got {sha}")
    rows = []
    text = data.decode("utf-8")
    decoder = json.JSONDecoder()
    idx = 0
    text_len = len(text)
    while idx < text_len:
        while idx < text_len and text[idx].isspace():
            idx += 1
        if idx >= text_len:
            break
        row, end = decoder.raw_decode(text, idx)
        idx = end
        if category is not None and row.get("category") != category:
            continue
        if "question" not in row or "answer" not in row:
            raise RuntimeError(f"missing question or answer in record at byte offset {idx}")
        row = dict(row)
        row["prompt"] = build_prompt(row)
        rows.append(row)
        if limit is not None and limit > 0 and len(rows) >= limit:
            break
    return rows


def evaluate_candidate(row: dict, **kwargs) -> dict:
    started = time.perf_counter()
    try:
        prompt = row["prompt"]
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt
        completion = chat_completion(messages, **kwargs)
        return {
            "index": row["index"],
            "id": row.get("id"),
            "category": row.get("category"),
            "answer_type": row.get("answer_type"),
            "prompt": row["prompt"],
            "answer": row.get("answer"),
            "response": completion["content"],
            "reasoning_content": completion["reasoning_content"],
            "finish_reason": completion["finish_reason"],
            "usage": completion["usage"],
            "complete": bool(completion.get("complete")),
            "error": completion.get("error"),
            "provider": completion["provider"],
            "latency_s": round(time.perf_counter() - started, 6),
            "retries": completion["attempts"] - 1,
            "requested_max_tokens": completion.get("requested_max_tokens"),
        }
    except Exception as exc:
        return {
            "index": row["index"],
            "id": row.get("id"),
            "category": row.get("category"),
            "answer_type": row.get("answer_type"),
            "prompt": row.get("prompt", ""),
            "answer": row.get("answer"),
            "response": "",
            "reasoning_content": "",
            "finish_reason": None,
            "usage": {},
            "complete": False,
            "error": f"{type(exc).__name__}: {exc}",
            "provider": "openai-compatible",
            "latency_s": round(time.perf_counter() - started, 6),
            "retries": 0,
            "requested_max_tokens": None,
        }

def evaluate_judge(row: dict, candidate: dict, **kwargs) -> dict:
    started = time.perf_counter()
    if not candidate.get("complete"):
        return {
            "index": row["index"],
            "complete": False,
            "correct": None,
            "parsed": {},
            "response": "",
            "reasoning": "",
            "confidence": None,
            "provider": None,
            "error": "candidate_incomplete",
            "latency_s": round(time.perf_counter() - started, 6),
            "retries": 0,
        }
    prompt = build_judge_prompt(row, candidate["response"])
    max_attempts = max(1, int(kwargs.pop("retries", 1)))
    last_completion = None
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            judge_completion = chat_completion(
                [{"role": "user", "content": prompt}],
                retries=1,
                **kwargs,
            )
            last_completion = judge_completion
            if not judge_completion.get("complete"):
                raise RuntimeError(judge_completion.get("error") or "judge request failed")
            parsed = parse_json_object(judge_completion["content"])
            correct = parse_bool(parsed.get("correct"))
            if correct is None:
                raise ValueError("judge returned malformed JSON or missing correct field")
            confidence = parse_confidence(parsed.get("confidence", 100))
            return {
                "index": row["index"],
                "complete": True,
                "correct": correct,
                "parsed": parsed,
                "response": judge_completion["content"],
                "reasoning": str(parsed.get("reasoning") or ""),
                "confidence": confidence,
                "provider": judge_completion["provider"],
                "error": None,
                "latency_s": round(time.perf_counter() - started, 6),
                "retries": attempt - 1,
                "requested_max_tokens": judge_completion.get("requested_max_tokens"),
            }
        except Exception as exc:
            last_error = exc
            if attempt < max_attempts:
                time.sleep(2**attempt)
    return {
        "index": row["index"],
        "complete": False,
        "correct": None,
        "parsed": {},
        "response": (last_completion or {}).get("content", ""),
        "reasoning": "",
        "confidence": None,
        "provider": (last_completion or {}).get("provider"),
        "error": f"{type(last_error).__name__}: {last_error}" if last_error else "judge failed",
        "latency_s": round(time.perf_counter() - started, 6),
        "retries": max_attempts - 1,
        "requested_max_tokens": (last_completion or {}).get("requested_max_tokens"),
    }


def accuracy_by_key(rows: list[dict], key: str) -> dict[str, float]:
    grouped: dict[str, list[bool]] = {}
    for row in rows:
        value = row.get(key)
        if value is None or row.get("complete") is not True:
            continue
        grouped.setdefault(str(value), []).append(bool(row.get("correct")))
    return {
        group: (sum(scores) / len(scores) if scores else 0.0)
        for group, scores in sorted(grouped.items())
    }


def dry_run(config: dict, split: str) -> int:
    dataset_path = resolve_repo_path(
        str(resolve_custom(config, "dataset_path", "data/datasets/hle"))
    )
    expected_sha = str(resolve_custom(config, "dataset_sha256", DATASET_SHA256))
    rows = load_rows(dataset_path, expected_sha, split, limit=1)
    first_row = rows[0]
    _ = build_prompt(first_row)
    _ = build_judge_prompt(first_row, "sample response")
    print(
        json.dumps(
            {
                "benchmark-id": BENCHMARK_ID,
                "status": "ok",
                "split": split,
                "dataset": {
                    "path": str(dataset_path),
                    "rows": DATASET_ROWS,
                    "sha256": expected_sha,
                    "revision": str(
                        resolve_custom(config, "dataset_revision", DATASET_REVISION)
                    ),
                },
                "adapter": "legacy_runner",
                "scorer": "unscored",
                "required-metric": "accuracy",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full HLE through BmkEverything")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-data", action="store_true")
    parser.add_argument(
        "--resume-from",
        help="reuse successful stages from an earlier samples.jsonl",
    )
    parser.add_argument(
        "--preserve-truncated",
        action="store_true",
        help="preserve truncations already produced at the maximum compatible budget",
    )
    parser.add_argument(
        "--rejudge",
        action="store_true",
        help="rerun Judge on preserved non-truncated candidates without rerunning the candidate model",
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    split = str(args.split or resolve_general(config, "split", "SPLIT_NAME", "default"))
    if split not in SPLIT_TO_CATEGORY:
        raise SystemExit(
            f"error: unsupported split {split!r}; available: {sorted(SPLIT_TO_CATEGORY)}"
        )
    if args.dry_run or args.validate_data:
        raise SystemExit(dry_run(config, split))

    started = time.time()
    model = str(resolve_general(config, "model", "MODEL_NAME", "kimi-k2.7-code"))
    base_url = str(resolve_general(config, "openai_api_base", "OPENAI_API_BASE", ""))
    token = str(resolve_openai_token(config))
    output_dir = Path(
        str(
            resolve_general(
                config,
                "output_dir",
                "OUTPUT_DIR",
                "data/output/hle/default",
            )
        )
    )
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = resolve_repo_path(
        str(resolve_custom(config, "dataset_path", "data/datasets/hle"))
    )
    expected_sha = str(resolve_custom(config, "dataset_sha256", DATASET_SHA256))
    rows = load_rows(dataset_path, expected_sha, split, args.limit)
    for index, row in enumerate(rows):
        row["index"] = index

    candidate_kwargs = {
        "base_url": base_url,
        "token": token,
        "model": model,
        "max_tokens": int(
            resolve_custom(
                config, "candidate_max_tokens", resolve_custom(config, "max_tokens", 65536)
            )
        ),
        "fallback_max_tokens": resolve_custom(
            config, "candidate_fallback_max_tokens", []
        ),
        "temperature": float(resolve_custom(config, "temperature", 0.0)),
        "timeout": int(
            resolve_custom(config, "candidate_timeout", resolve_custom(config, "timeout", 300))
        ),
        "retries": int(resolve_custom(config, "candidate_retries", 3)),
    }
    judge_kwargs = {
        "base_url": str(resolve_custom(config, "grader_api_base", DEFAULT_GRADER_BASE)),
        "token": str(resolve_grader_token(config)),
        "model": str(resolve_custom(config, "grader_model", DEFAULT_GRADER_MODEL)),
        "max_tokens": int(resolve_custom(config, "grader_max_tokens", 32768)),
        "temperature": float(resolve_custom(config, "grader_temperature", 0.0)),
        "timeout": int(resolve_custom(config, "grader_timeout", 300)),
        "retries": int(resolve_custom(config, "grader_retries", 3)),
    }
    candidate_workers = int(
        resolve_custom(config, "candidate_workers", resolve_custom(config, "max_workers", 8))
    )
    judge_workers = int(
        resolve_custom(config, "grader_workers", resolve_custom(config, "max_workers", 8))
    )

    print(
        f"[hle] candidate={model} judge={judge_kwargs['model']} split={split} samples={len(rows)}"
    )

    if not isinstance(candidate_kwargs["fallback_max_tokens"], list):
        raise RuntimeError("custom.candidate_fallback_max_tokens must be a list")

    preserved_records: dict[int, dict] = {}
    resume_candidates: list[dict | None] = [None] * len(rows)
    judge_results: list[dict | None] = [None] * len(rows)
    if args.resume_from:
        preserved_records, resume_candidates, judge_results = load_resume_state(
            Path(args.resume_from),
            rows,
            preserve_truncated=args.preserve_truncated,
        )
    if args.rejudge:
        preserved_records = {
            index: record
            for index, record in preserved_records.items()
            if record_candidate_is_truncated(record)
        }
        judge_results = [None] * len(rows)

    checkpoint_path = output_dir / "candidates.jsonl"
    candidate_signature = {
        key: value for key, value in candidate_kwargs.items() if key != "token"
    }
    checkpoint_results = load_candidate_checkpoint(
        checkpoint_path,
        rows,
        candidate_signature,
    )
    candidate_results = [
        checkpoint_results[index] or resume_candidates[index]
        for index in range(len(rows))
    ]
    pending_indices = [
        index
        for index, result in enumerate(candidate_results)
        if result is None or not result.get("complete")
    ]
    preserved_candidates = len(rows) - len(pending_indices)
    if preserved_candidates:
        print(
            f"[hle] candidate checkpoint preserved={preserved_candidates} "
            f"pending={len(pending_indices)}"
        )
    with ThreadPoolExecutor(
        max_workers=min(candidate_workers, max(len(pending_indices), 1))
    ) as executor, checkpoint_path.open("a", encoding="utf-8") as checkpoint:
        futures = {
            executor.submit(evaluate_candidate, rows[index], **candidate_kwargs): index
            for index in pending_indices
        }
        with tqdm(
            total=len(rows),
            initial=preserved_candidates,
            desc="candidate",
            unit="sample",
            dynamic_ncols=True,
        ) as bar:
            for future in as_completed(futures):
                index = futures[future]
                candidate_results[index] = future.result()
                checkpoint.write(
                    json.dumps(
                        {
                            "signature": candidate_signature,
                            "index": index,
                            "id": rows[index].get("id"),
                            "result": candidate_results[index],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                checkpoint.flush()
                bar.update(1)

    candidate_truncated = [
        bool(
            candidate
            and candidate.get("complete")
            and is_truncated(
                candidate.get("finish_reason"),
                candidate.get("usage") or {},
                int(
                    candidate.get("requested_max_tokens")
                    or candidate_kwargs["max_tokens"]
                ),
            )
        )
        for candidate in candidate_results
    ]
    judge_indices = [
        index
        for index, candidate in enumerate(candidate_results)
        if candidate
        and candidate.get("complete")
        and not candidate_truncated[index]
        and judge_results[index] is None
    ]
    if judge_indices:
        with ThreadPoolExecutor(max_workers=min(judge_workers, max(len(judge_indices), 1))) as executor:
            futures = {
                executor.submit(evaluate_judge, rows[index], candidate_results[index], **judge_kwargs): index
                for index in judge_indices
            }
            with tqdm(total=len(judge_indices), desc="judge", unit="sample", dynamic_ncols=True) as bar:
                for future in as_completed(futures):
                    index = futures[future]
                    judge_results[index] = future.result()
                    bar.update(1)

    records = []
    for index, row in enumerate(rows):
        if index in preserved_records:
            records.append(preserved_records[index])
            continue
        candidate = candidate_results[index]
        judge = judge_results[index] if judge_results[index] is not None else {
            "index": index,
            "complete": False,
            "correct": None,
            "parsed": {},
            "response": "",
            "reasoning": "",
            "confidence": None,
            "provider": None,
            "error": "judge_not_run",
            "latency_s": 0.0,
            "retries": 0,
        }
        truncated = candidate_truncated[index]
        complete, correct = scored_outcome(candidate, judge, truncated)
        valid = bool(candidate.get("complete") and not truncated and judge.get("complete"))
        failure_origin = None
        failure_reason = None
        if truncated:
            failure_origin = "model"
            failure_reason = "length_truncated"
        elif not candidate.get("complete"):
            failure_origin = "infrastructure"
            failure_reason = "candidate_error"
        elif not judge.get("complete"):
            failure_origin = "infrastructure"
            failure_reason = "judge_error"
        error = (
            "CandidateTruncated: response reached the configured output limit"
            if truncated
            else candidate.get("error") or judge.get("error")
        )
        records.append(
            {
                "index": index,
                "id": row.get("id"),
                "category": row.get("category"),
                "answer_type": row.get("answer_type"),
                "prompt": row.get("prompt"),
                "answer": row.get("answer"),
                "candidate": {
                    "provider": candidate.get("provider"),
                    "content": candidate.get("response", ""),
                    "reasoning_content": candidate.get("reasoning_content", ""),
                    "finish_reason": candidate.get("finish_reason"),
                    "usage": candidate.get("usage") or {},
                    "complete": candidate.get("complete"),
                    "truncated": truncated,
                    "error": candidate.get("error"),
                    "latency_s": candidate.get("latency_s", 0.0),
                    "retries": candidate.get("retries", 0),
                    "requested_max_tokens": candidate.get("requested_max_tokens"),
                },
                "judge": {
                    "provider": judge.get("provider"),
                    "content": judge.get("response", ""),
                    "parsed": judge.get("parsed") or {},
                    "reasoning": judge.get("reasoning", ""),
                    "confidence": judge.get("confidence"),
                    "complete": judge.get("complete"),
                    "error": judge.get("error"),
                    "latency_s": judge.get("latency_s", 0.0),
                    "retries": judge.get("retries", 0),
                    "requested_max_tokens": judge.get("requested_max_tokens"),
                },
                "correct": correct,
                "score": 1 if correct is True else 0,
                "complete": complete,
                "valid": valid,
                "failure_origin": failure_origin,
                "failure_reason": failure_reason,
                "incomplete_reason": None if complete else (
                    "candidate_error" if not candidate.get("complete") else "judge_error"
                ),
                "error": error,
            }
        )

    summary = summarize_records(records)
    completed = summary["completed"]
    incomplete = summary["incomplete"]
    correct = summary["correct"]
    accuracy = summary["accuracy"]
    reward = accuracy

    with (output_dir / "samples.jsonl").open("w", encoding="utf-8") as sink:
        for record in records:
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")

    answer_type_counts = {
        str(key): sum(1 for row in rows if row.get("answer_type") == key)
        for key in sorted({row.get("answer_type") for row in rows if row.get("answer_type") is not None})
    }
    category_counts = {
        str(key): sum(1 for row in rows if row.get("category") == key)
        for key in sorted({row.get("category") for row in rows if row.get("category") is not None})
    }

    meta_data = {
        "benchmark": BENCHMARK_ID,
        "benchmark_id": BENCHMARK_ID,
        "model": model,
        "candidate_provider": "openai-compatible",
        "judge_model": judge_kwargs["model"],
        "judge_api_base": judge_kwargs["base_url"],
        "judge_protocol": "paper_hle_equivalence",
        "metric": "accuracy",
        "split": split,
        "total": len(rows),
        "attempted": len(rows),
        "completed": completed,
        "incomplete": incomplete,
        "candidate_failures": summary["candidate_failures"],
        "judge_failures": summary["judge_failures"],
        "infra_failures": summary["infra_failures"],
        "evaluation_status": summary["evaluation_status"],
        "official_comparable": summary["official_comparable"],
        "correct": correct,
        "accuracy": accuracy,
        "observed_reward": summary["observed_accuracy"],
        "truncated_total": sum(
            1 for record in records if record["candidate"].get("truncated")
        ),
        "answer_type_counts": answer_type_counts,
        "category_counts": category_counts,
        "by_answer_type": accuracy_by_key(records, "answer_type"),
        "by_category": accuracy_by_key(records, "category"),
        "dataset": {
            "path": "data/datasets/hle/default.jsonl",
            "rows": DATASET_ROWS,
            "selected_rows": len(rows),
            "revision": str(resolve_custom(config, "dataset_revision", DATASET_REVISION)),
            "sha256": expected_sha,
        },
        "elapsed_s": round(time.time() - started, 3),
        "scoring_supported": True,
    }
    (output_dir / "reward.json").write_text(
        json.dumps({"reward": reward, "meta_data": meta_data}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(
            resolved_config(config, model, split, output_dir),
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "benchmark": BENCHMARK_ID,
                "status": summary["evaluation_status"],
                "reward": reward,
                "accuracy": accuracy,
                "completed": completed,
                "incomplete": incomplete,
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
