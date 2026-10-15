#!/usr/bin/env python3
"""Recover empty responses without consulting gold targets.

The script first extracts only explicit final-answer markers from saved
reasoning.  Remaining rows with reasoning can be sent to a short finalizer.
Progress is stored per (benchmark_id, sample_index), so interruption never
causes completed samples to be repeated.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


ANSWER_LINE = re.compile(
    r"(?im)^\s*(?:therefore[, :]*)?(?:the\s+)?(?:final\s+)?answer\s*:\s*(\S[^\n]{0,200})\s*$"
)
ANSWER_IS_LINE = re.compile(
    r"(?im)^\s*(?:therefore[, :]*)?(?:the\s+)?(?:final\s+)?answer\s+"
    r"(?:is|would be)\s+(.{1,200})\s*$"
)
BOXED = re.compile(r"\\boxed\{([^{}\n]{1,200})\}")
NUMERIC = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def batch_rows(batch: Path) -> dict[tuple[str, int], dict]:
    result = {}
    for state in jsonl(batch / "state.jsonl"):
        samples = Path(state["output_dir"]) / "samples.jsonl"
        for row in jsonl(samples):
            result[(state["key"], int(row["index"]))] = row
    return result


def scorer_for(format_dir: Path, benchmark_id: str) -> str:
    spec = format_dir / benchmark_id / "spec.json"
    if spec.exists():
        return str(read_json(spec).get("scorer") or "exact_match")
    return {"aime2025": "numeric", "gsm8k": "numeric", "gpqa": "multiple_choice"}.get(
        benchmark_id, "exact_match"
    )


def reasoning_row(key: tuple[str, int], reasoning_sources: list[dict], source: dict) -> dict:
    for reasoning in reasoning_sources:
        candidate = reasoning.get(key)
        if candidate and str(candidate.get("reasoning_content") or "").strip():
            return candidate
    return source[key]


def clean_candidate(candidate: str, scorer: str) -> str | None:
    text = candidate.strip().strip('"\'`').rstrip(".。")
    if scorer == "multiple_choice":
        match = re.fullmatch(r"(?i)(?:option\s*)?[\[(]?([A-Z])[\])]?", text)
        return f"ANSWER: {match.group(1).upper()}" if match else None
    if scorer == "numeric":
        match = re.fullmatch(r"\$?([-+]?\d+(?:,\d{3})*(?:\.\d+)?)%?", text)
        return f"ANSWER: {match.group(1)}" if match else None
    if scorer == "boolean":
        return f"ANSWER: {text.lower()}" if text.lower() in {"yes", "no", "true", "false"} else None
    if scorer == "json":
        try:
            json.loads(text)
        except Exception:
            return None
    if not text or len(text) > 200:
        return None
    return f"ANSWER: {text}"


def strict_extract(reasoning: str, scorer: str) -> tuple[str, str] | None:
    """Return only explicit end-of-reasoning final markers; never use targets."""
    if not reasoning.strip():
        return None
    tail = reasoning[-2500:]
    candidates: list[tuple[int, str, str]] = []
    for method, regex in (("answer_marker", ANSWER_LINE), ("answer_is", ANSWER_IS_LINE)):
        for match in regex.finditer(tail):
            candidates.append((match.start(), method, match.group(1)))
    for match in BOXED.finditer(tail):
        candidates.append((match.start(), "boxed", match.group(1)))
    for _, method, candidate in sorted(candidates, reverse=True):
        cleaned = clean_candidate(candidate, scorer)
        if cleaned:
            return cleaned, method
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format-dir", type=Path, required=True)
    parser.add_argument("--source-batch", type=Path, required=True)
    parser.add_argument("--reasoning-batch", type=Path, required=True)
    parser.add_argument("--fallback-reasoning-batch", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-empty", type=int)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--primary-provider", choices=["moonshot", "ksyun"], default="moonshot")
    parser.add_argument("--thinking-type", choices=["enabled", "disabled"])
    parser.add_argument("--continue-partial", action="store_true",
                        help="Continue saved reasoning as a partial assistant message")
    parser.add_argument("--execute-finalizer", action="store_true")
    parser.add_argument("--approved-call-count", type=int)
    parser.add_argument("--max-new-calls", type=int)
    args = parser.parse_args()

    format_dir = args.format_dir.resolve()
    sys.path.insert(0, str(format_dir))
    sys.path.insert(0, str(format_dir.parent / "framework"))
    from core.provider import chat_detailed

    source = batch_rows(args.source_batch)
    reasoning_sources = [batch_rows(args.reasoning_batch)]
    if args.fallback_reasoning_batch:
        reasoning_sources.append(batch_rows(args.fallback_reasoning_batch))
    empty_keys = sorted(key for key, row in source.items() if not str(row.get("response") or "").strip())
    if args.expected_empty is not None and len(empty_keys) != args.expected_empty:
        raise SystemExit(f"expected {args.expected_empty} empty source rows, found {len(empty_keys)}")

    args.output.mkdir(parents=True, exist_ok=True)
    sample_state = args.output / "sample_state.jsonl"
    previous = jsonl(sample_state)
    latest = {(row["benchmark_id"], int(row["index"])): row for row in previous}

    extracted = []
    for key in empty_keys:
        if key in latest:
            continue
        benchmark_id, index = key
        row = reasoning_row(key, reasoning_sources, source)
        result = strict_extract(str(row.get("reasoning_content") or ""),
                                scorer_for(format_dir, benchmark_id))
        if result:
            response, detail = result
            record = {
                "benchmark_id": benchmark_id, "index": index, "response": response,
                "method": "reasoning_extraction", "method_detail": detail,
                "provider": None, "finish_reason": "recovered", "usage": {},
            }
            extracted.append(record)
            latest[key] = record
    if extracted:
        with sample_state.open("a", encoding="utf-8") as sink:
            for record in extracted:
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")

    pending = []
    for key in empty_keys:
        if key in latest:
            continue
        row = reasoning_row(key, reasoning_sources, source)
        saved_reasoning = str(row.get("reasoning_content") or "").strip()
        if saved_reasoning:
            pending.append((key, saved_reasoning))

    attempted_finalizer = sum(row.get("method") == "finalizer" for row in latest.values())
    total_finalizer_plan = attempted_finalizer + len(pending)
    print(json.dumps({
        "source_empty": len(empty_keys), "strict_extracted": sum(
            row.get("method") == "reasoning_extraction" for row in latest.values()),
        "without_reasoning": sum(not str(reasoning_row(key, reasoning_sources, source).get(
            "reasoning_content") or "").strip() for key in empty_keys),
        "finalizer_plan": total_finalizer_plan, "finalizer_attempted": attempted_finalizer,
        "finalizer_pending": len(pending), "execute_finalizer": args.execute_finalizer,
    }, ensure_ascii=False), flush=True)
    if not args.execute_finalizer:
        materialize(args.output, latest)
        return
    if args.approved_call_count != total_finalizer_plan:
        raise SystemExit(
            f"approved finalizer calls {args.approved_call_count} != plan {total_finalizer_plan}"
        )
    if args.max_new_calls is not None:
        if args.max_new_calls < 1:
            raise SystemExit("--max-new-calls must be positive")
        pending = pending[:args.max_new_calls]

    os.environ["KIMI_FINAL_ONLY"] = "1"
    os.environ["KIMI_PROVIDER_RETRIES"] = "1"
    os.environ["KIMI_PRIMARY_PROVIDER"] = args.primary_provider
    if args.thinking_type:
        os.environ["KIMI_THINKING_TYPE"] = args.thinking_type
    else:
        os.environ.pop("KIMI_THINKING_TYPE", None)

    def finalize(item: tuple[tuple[str, int], str]) -> dict:
        (benchmark_id, index), saved_reasoning = item
        tail = saved_reasoning[-24000:]
        prompt = (
            "The text below is an unfinished reasoning trace for one benchmark question. "
            "Do not restart or explain the reasoning. Infer the conclusion already supported by "
            "the trace and return exactly one short line in the form ANSWER: <answer>. "
            "Never leave the answer empty.\n\nUNFINISHED REASONING:\n" + tail
        )
        messages = [{"role": "user", "content": prompt}]
        if args.continue_partial:
            messages = [
                {"role": "user", "content": (
                    "Continue the saved reasoning below without restarting. Finish immediately "
                    "and output exactly one short line: ANSWER: <answer>."
                )},
                {"role": "assistant", "content": "", "reasoning_content": tail,
                 "partial": True},
            ]
        try:
            result = chat_detailed(messages,
                                   model="kimi-k2.7-code", max_tokens=args.max_tokens,
                                   timeout=600, retries=1)
            return {
                "benchmark_id": benchmark_id, "index": index, "response": result.content,
                "method": "finalizer", "provider": result.provider,
                "finish_reason": result.finish_reason, "usage": result.usage,
                "reasoning_content": result.reasoning_content,
                "complete": result.complete,
            }
        except Exception as exc:
            return {
                "benchmark_id": benchmark_id, "index": index, "response": "",
                "method": "finalizer", "provider": None, "finish_reason": None,
                "usage": {}, "reasoning_content": "", "complete": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    with sample_state.open("a", encoding="utf-8") as sink, \
            concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(finalize, item) for item in pending]
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            latest[(record["benchmark_id"], int(record["index"]))] = record
            print(f"[{'OK' if str(record.get('response') or '').strip() else 'EMPTY'}] "
                  f"{record['benchmark_id']}:{record['index']}", flush=True)

    materialize(args.output, latest)


def materialize(output: Path, latest: dict[tuple[str, int], dict]) -> None:
    by_benchmark: dict[str, list[dict]] = defaultdict(list)
    for (benchmark_id, index), record in latest.items():
        row = dict(record)
        row["index"] = index
        row.setdefault("reasoning_content", "")
        row.setdefault("complete", bool(str(row.get("response") or "").strip()))
        row.setdefault("incomplete_reason", None if row["complete"] else "empty_content")
        row.setdefault("targets", [])
        row.setdefault("correct", None)
        by_benchmark[benchmark_id].append(row)
    state_rows = []
    for benchmark_id, rows in sorted(by_benchmark.items()):
        rows.sort(key=lambda row: int(row["index"]))
        out_dir = output / "benchmarks" / benchmark_id
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "samples.jsonl").open("w", encoding="utf-8") as sink:
            for row in rows:
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
        state_rows.append({
            "key": benchmark_id,
            "calls": sum(row.get("method") == "finalizer" for row in rows),
            "indices": [int(row["index"]) for row in rows],
            "ok": all(bool(str(row.get("response") or "").strip()) for row in rows),
            "completed_samples": sum(bool(str(row.get("response") or "").strip()) for row in rows),
            "output_dir": str(out_dir.resolve()),
        })
    with (output / "state.jsonl").open("w", encoding="utf-8") as sink:
        for row in state_rows:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "sample_records": len(latest),
        "strict_extracted": sum(row.get("method") == "reasoning_extraction" for row in latest.values()),
        "finalizer_calls": sum(row.get("method") == "finalizer" for row in latest.values()),
        "nonempty": sum(bool(str(row.get("response") or "").strip()) for row in latest.values()),
        "empty": sum(not bool(str(row.get("response") or "").strip()) for row in latest.values()),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
