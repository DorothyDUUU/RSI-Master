#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import yaml


BENCHMARK_ID = "cmphysbench"
EXPECTED_COMMIT = "b2cd8571279450f0861759f47d98e9fc577aa993"
EXPECTED_DATA_SHA256 = "84104d21c93769d8f7af152874d31439ce2c7ce427349edcd8c811970d7ed526"
SYSTEM_PROMPT = (
    "You are a condensed matter physics expert. Please read the following question and provide a step-by-step solution "
    "using only the given symbols. Do not introduce any new symbols that are not provided in the problem statement. "
    "Please place your reasoning inside <think>...</think> tags. "
    "Your final answer must be presented as a readable LaTeX formula, enclosed in a \\boxed{} environment."
)


def _placeholder(value):
    return isinstance(value, str) and value.startswith("<") and value.endswith(">")


def _setting(config, section, key, env=None, default=None):
    value = (config.get(section) or {}).get(key)
    if value is None or _placeholder(value):
        return os.environ.get(env, default) if env else default
    return value


def _resolve(bench_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (bench_dir / path).resolve()


def _resolve_executable(bench_dir: Path, value: str) -> Path:
    """Keep the venv launcher path instead of dereferencing its python symlink."""
    path = Path(value)
    return path if path.is_absolute() else Path(os.path.abspath(bench_dir / path))


def _verify_scorer_runtime(scorer_python: Path) -> None:
    if not scorer_python.is_file():
        raise SystemExit(f"CMPhysBench scorer Python is missing: {scorer_python}")
    check = subprocess.run(
        [str(scorer_python), "-c", "import numpy, sympy"],
        capture_output=True,
        text=True,
    )
    if check.returncode != 0:
        detail = (check.stderr or check.stdout).strip()
        raise SystemExit(
            f"CMPhysBench scorer environment is incomplete: {detail}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select(rows: list[dict], limit: int | None) -> list[dict]:
    raw = os.environ.get("SAMPLE_INDICES")
    if raw:
        indices = [int(item) for item in raw.split(",")]
        if any(index >= len(rows) for index in indices):
            raise ValueError(f"sample index out of range for {len(rows)} rows")
        rows = [rows[index] for index in indices]
    if limit is not None and limit >= 0:
        rows = rows[:limit]
    return rows


def _question(row: dict) -> str:
    question = (row.get("context") or "") + (row.get("question") or "")
    return question + "\nHere are the relevant symbols:\n" + (row.get("symbol") or "")


def _separate_reasoning(content: str, reasoning: str) -> tuple[str, str]:
    """Repair Qwen reasoning-parser fallbacks without treating thoughts as answers."""
    content = content or ""
    reasoning = reasoning or ""
    if "</think>" in content:
        leaked_reasoning, content = content.rsplit("</think>", 1)
        leaked_reasoning = leaked_reasoning.lstrip()
        if leaked_reasoning.startswith("<think>"):
            leaked_reasoning = leaked_reasoning[len("<think>"):]
        reasoning = "\n".join(part for part in (reasoning.rstrip(), leaked_reasoning.strip()) if part)
    elif content.lstrip().startswith("<think>"):
        leaked_reasoning = content.lstrip()[len("<think>"):]
        reasoning = "\n".join(part for part in (reasoning.rstrip(), leaked_reasoning.strip()) if part)
        content = ""
    return content.strip(), reasoning.strip()


def _normalize_record(record: dict, native_context_tokens: int) -> dict:
    answer = (record.get("llm_answer") or [""])[0]
    answer, reasoning = _separate_reasoning(answer, record.get("reasoning_content") or "")
    record["llm_answer"] = [answer]
    record["reasoning_content"] = reasoning
    usage = record.get("usage") or {}
    if not isinstance(record.get("native_context_exhausted"), bool):
        record["native_context_exhausted"] = (
            record.get("finish_reason") == "length"
            and usage.get("total_tokens") == native_context_tokens
        )
    return record


def _call_model(row, *, base_url, token, model, custom, session):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _question(row)},
        ],
        "temperature": float(custom["temperature"]),
        "top_p": float(custom["top_p"]),
        "top_k": int(custom["top_k"]),
        "min_p": float(custom["min_p"]),
        "n": 1,
    }
    if custom.get("max_tokens") is not None:
        payload["max_tokens"] = int(custom["max_tokens"])
    if custom.get("enable_thinking") is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": bool(custom["enable_thinking"])}
    last_error = None
    for attempt in range(int(custom.get("request_retries", 2)) + 1):
        try:
            response = session.post(
                base_url.rstrip("/") + "/chat/completions",
                headers={"Authorization": f"Bearer {token}"},
                json=payload,
                timeout=float(custom["request_timeout"]),
            )
            response.raise_for_status()
            body = response.json()
            break
        except (requests.RequestException, ValueError, KeyError) as exc:
            last_error = exc
            if attempt >= int(custom.get("request_retries", 2)):
                raise
            time.sleep(min(2 ** attempt, 8))
    else:  # pragma: no cover - loop either breaks or raises
        raise RuntimeError(f"request failed: {last_error}")
    choice = body["choices"][0]
    message = choice["message"]
    content, reasoning = _separate_reasoning(
        message.get("content") or "",
        message.get("reasoning_content") or message.get("reasoning") or "",
    )
    native_context_tokens = int(custom["native_context_tokens"])
    expected_cap = choice.get("finish_reason") == "length" and (
        custom.get("max_tokens") is not None
        or (body.get("usage") or {}).get("total_tokens") == native_context_tokens
    )
    return {
        "index": row["id"],
        "question": (row.get("context") or "") + (row.get("question") or ""),
        "gt_answer": row.get("final_answer"),
        "llm_answer": [content],
        "reasoning_content": reasoning,
        "topic": row.get("topic"),
        "answer_type": row.get("answer_type"),
        "finish_reason": choice.get("finish_reason"),
        "usage": body.get("usage"),
        "native_context_exhausted": expected_cap,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-data", action="store_true")
    parser.add_argument("--score-existing", action="store_true")
    parser.add_argument(
        "--resume", action="store_true",
        help="reuse unique existing responses and request only missing benchmark IDs",
    )
    parser.add_argument(
        "--retry-length", action="store_true",
        help="with --resume, request existing finish_reason=length rows again",
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    bench_dir = Path(__file__).resolve().parent
    custom = dict(config.get("custom") or {})
    official_root = _resolve(bench_dir, str(custom["official_root"]))
    dataset_path = _resolve(bench_dir, str(custom["dataset_path"]))
    scorer_python = _resolve_executable(bench_dir, str(custom["scorer_python"]))
    scorer_script = bench_dir / "score_results.py"

    commit = subprocess.check_output(
        ["git", "-C", str(official_root), "rev-parse", "HEAD"], text=True
    ).strip()
    expected_commit = str(custom.get("official_commit", EXPECTED_COMMIT))
    if commit != expected_commit:
        raise SystemExit(f"official commit mismatch: expected {expected_commit}, got {commit}")
    if _sha256(dataset_path) != str(custom.get("dataset_sha256", EXPECTED_DATA_SHA256)):
        raise SystemExit("CMPhysBench dataset SHA256 mismatch")
    rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != 520:
        raise SystemExit(f"expected 520 CMPhysBench rows, got {len(rows) if isinstance(rows, list) else type(rows)}")
    rows = _select(rows, args.limit)
    if args.dry_run:
        result = {
            "benchmark": BENCHMARK_ID,
            "split": "default",
            "rows": len(rows),
            "official_commit": commit,
            "dataset_revision": custom.get("dataset_revision"),
            "dataset_sha256": _sha256(dataset_path),
        }
        if args.validate_data and rows:
            result["first"] = {key: rows[0].get(key) for key in ("id", "topic", "answer_type")}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    _verify_scorer_runtime(scorer_python)

    model = str(_setting(config, "general", "model", "MODEL_NAME", ""))
    base_url = str(_setting(config, "general", "openai_api_base", "OPENAI_API_BASE", ""))
    token = str(_setting(config, "general", "openai_api_token", "OPENAI_API_TOKEN", "sk-local"))
    output_dir = Path(str(_setting(config, "general", "output_dir", "OUTPUT_DIR", "output")))
    output_dir.mkdir(parents=True, exist_ok=True)
    local = threading.local()
    lock = threading.Lock()
    records: list[dict] = []
    errors: list[dict] = []
    started = time.time()

    def generate(row):
        if not hasattr(local, "session"):
            local.session = requests.Session()
        return _call_model(
            row, base_url=base_url, token=token, model=model, custom=custom, session=local.session
        )

    responses_path = output_dir / "responses.jsonl"
    errors_path = output_dir / "errors.jsonl"
    if args.score_existing:
        if not responses_path.is_file():
            raise SystemExit(f"missing existing responses: {responses_path}")
        records = [
            _normalize_record(json.loads(line), int(custom["native_context_tokens"]))
            for line in responses_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if errors_path.is_file():
            errors = [json.loads(line) for line in errors_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        responses_path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
    else:
        existing_by_id = {}
        if args.resume and responses_path.is_file():
            for line in responses_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = _normalize_record(json.loads(line), int(custom["native_context_tokens"]))
                if args.retry_length and record.get("finish_reason") == "length":
                    continue
                if record.get("index") is not None:
                    existing_by_id[record["index"]] = record
        records = list(existing_by_id.values())
        pending_rows = [row for row in rows if row.get("id") not in existing_by_id]
        print(
            f"resume existing={len(records)} pending={len(pending_rows)} total={len(rows)}",
            flush=True,
        )
        mode = "a" if args.resume and responses_path.is_file() else "w"
        with responses_path.open(mode, encoding="utf-8") as sink, ThreadPoolExecutor(
            max_workers=int(custom["max_workers"])
        ) as pool:
            futures = {pool.submit(generate, row): row for row in pending_rows}
            for done, future in enumerate(as_completed(futures), 1):
                row = futures[future]
                try:
                    record = future.result()
                    records.append(record)
                    with lock:
                        sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                        sink.flush()
                except Exception as exc:  # keep auditable partial output
                    errors.append({"index": row.get("id"), "error": f"{type(exc).__name__}: {exc}"})
                print(f"generation {done}/{len(pending_rows)}", flush=True)

        # Canonicalize and atomically rewrite one response per benchmark ID.
        records_by_id = {record.get("index"): record for record in records if record.get("index") is not None}
        ordered_records = [records_by_id[row["id"]] for row in rows if row["id"] in records_by_id]
        temp_responses = responses_path.with_suffix(".jsonl.tmp")
        temp_responses.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in ordered_records),
            encoding="utf-8",
        )
        os.replace(temp_responses, responses_path)
        records = ordered_records

    errors_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in errors), encoding="utf-8"
    )
    ids = [record.get("index") for record in records]
    finish_reasons = {}
    for record in records:
        reason = str(record.get("finish_reason"))
        finish_reasons[reason] = finish_reasons.get(reason, 0) + 1
    native_context_tokens = int(custom["native_context_tokens"])
    expected_cap_exhaustions = sum(
        bool(record.get("native_context_exhausted")) for record in records
    )
    unexpected_length_truncations = sum(
        record.get("finish_reason") == "length" and not record.get("native_context_exhausted")
        for record in records
    )
    validation = {
        "valid": False,
        "benchmark": BENCHMARK_ID,
        "attempted": len(rows),
        "completed": len(records),
        "unique": len(set(ids)),
        "errors": len(errors),
        "finish_reasons": finish_reasons,
        "length_truncations": finish_reasons.get("length", 0),
        "expected_cap_exhaustions": expected_cap_exhaustions,
        "unexpected_length_truncations": unexpected_length_truncations,
        "thinking_leaks": sum(
            ((record.get("llm_answer") or [""])[0]).lstrip().startswith("<think>")
            or "</think>" in ((record.get("llm_answer") or [""])[0])
            for record in records
        ),
        "reasoning_outputs": sum(bool(record.get("reasoning_content")) for record in records),
        "max_tokens_policy": "remaining_native_context" if custom.get("max_tokens") is None else "fixed",
        "max_tokens": custom.get("max_tokens"),
        "native_context_tokens": native_context_tokens,
        "temperature": float(custom["temperature"]),
        "top_p": float(custom["top_p"]),
        "top_k": int(custom["top_k"]),
        "min_p": float(custom["min_p"]),
    }
    # Model-output failures remain scoreable: empty/final-format errors score 0,
    # while length truncation and thinking leakage stay in validation telemetry.
    # Only an incomplete request set blocks the benchmark run.
    validation["violations"] = [
        "missing_or_failed_requests"
    ] if len(records) != len(rows) or bool(errors) else []
    validation["valid"] = not validation["violations"]
    (output_dir / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not validation["valid"]:
        raise SystemExit(f"CMPhysBench run rejected by validation gate: {validation['violations']}")
    score_command = [
        str(scorer_python), str(scorer_script),
        "--official-root", str(official_root),
        "--responses", str(responses_path),
        "--output-dir", str(output_dir),
        "--workers", str(custom.get("score_workers", 16)),
    ]
    subprocess.run(score_command, check=True)
    reward_path = output_dir / "reward.json"
    reward = json.loads(reward_path.read_text(encoding="utf-8"))
    meta = reward["meta_data"]
    meta.update({
        "attempted": len(rows),
        "completed": len(records),
        "incomplete": len(errors),
        "model": model,
        "official_commit": commit,
        "dataset_revision": custom.get("dataset_revision"),
        "temperature": float(custom["temperature"]),
        "top_p": float(custom["top_p"]),
        "top_k": int(custom["top_k"]),
        "min_p": float(custom["min_p"]),
        "max_tokens": custom.get("max_tokens"),
        "max_tokens_policy": validation["max_tokens_policy"],
        "validation": validation,
        "elapsed_seconds": time.time() - started,
    })
    meta["observed_reward"] = meta["accuracy"]
    reward_path.write_text(json.dumps(reward, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(reward, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
