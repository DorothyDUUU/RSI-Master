#!/usr/bin/env python3
"""Audit a completed answer batch without making model calls."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


def read_json(path: Path):
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().split("\n") if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format-dir", type=Path, required=True)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--bigbench-metrics", type=Path)
    parser.add_argument("--include-incomplete", action="store_true",
                        help="Audit state rows even when the job is marked incomplete")
    args = parser.parse_args()
    format_dir = args.format_dir.resolve()
    sys.path.insert(0, str(format_dir))
    sys.path.insert(0, str(format_dir.parent / "framework"))
    from core.scoring import (boolean_match, exact_match, json_match, label_match,
                              multiple_choice_match, numeric_match, squad_match, unscored)

    scorers = {
        "exact_match": exact_match,
        "json": json_match,
        "multiple_choice": multiple_choice_match,
        "numeric": numeric_match,
        "boolean": boolean_match,
        "label": label_match,
        "squad": squad_match,
        "unscored": unscored,
    }
    successful = {}
    for row in jsonl(args.batch_dir / "state.jsonl"):
        if row.get("ok") or args.include_incomplete:
            successful[row["key"]] = row
    upstream = {}
    if args.bigbench_metrics:
        upstream = {row["benchmark_id"]: row for row in read_json(args.bigbench_metrics)}

    providers = Counter()
    groups = defaultdict(Counter)
    per_benchmark = []
    parse_errors = []
    for benchmark_id, state in successful.items():
        manifest = read_json(format_dir / benchmark_id / "config.json")
        adapter = manifest.get("adapter", "legacy_runner")
        spec_path = format_dir / benchmark_id / "spec.json"
        spec = read_json(spec_path) if spec_path.exists() else None
        scorer_name = spec["scorer"] if spec else "legacy"
        samples = []
        for line_no, line in enumerate(
                (Path(state["output_dir"]) / "samples.jsonl").read_text().split("\n"), 1):
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                parse_errors.append({"benchmark_id": benchmark_id, "line": line_no,
                                     "error": str(exc)})
        empty = sum(not str(row.get("response") or "").strip() for row in samples)
        correct = sum(bool(row.get("correct")) for row in samples)
        mismatch = 0
        if spec and scorer_name != "unscored":
            scorer = scorers[scorer_name]
            for row in samples:
                if not str(row.get("response") or "").strip():
                    continue
                recomputed = scorer(row.get("response", ""), row.get("targets") or [])
                mismatch += int(bool(recomputed) != bool(row.get("correct")))
        for row in samples:
            providers[row.get("provider", "legacy-no-provider-field")] += 1
        groups[adapter].update(tasks=1, samples=len(samples), empty=empty, correct=correct,
                               scorer_mismatches=mismatch)

        metric_status = None
        upstream_metric = None
        effective_metric = None
        if adapter == "bigbench" and benchmark_id in upstream:
            metric = upstream[benchmark_id]
            declared = metric.get("metrics") or []
            upstream_metric = metric.get("preferred_score") or (
                declared[0] if len(declared) == 1 else None
            )
            targets = [str(target) for row in samples for target in (row.get("targets") or [])]
            letters = sum(bool(re.fullmatch(r"[A-Z]", target)) for target in targets)
            effective_metric = (
                "multiple_choice_grade" if targets and letters == len(targets)
                else "exact_str_match" if letters == 0 else "mixed"
            )
            unsupported = {"bleu", "bleurt", "rougeLsum", "log_likelihood",
                           "sequence_f1", "macro_f1"}
            if upstream_metric in unsupported:
                metric_status = "unsupported_preferred"
            elif upstream_metric is None:
                metric_status = "upstream_unspecified"
            elif upstream_metric == effective_metric or (
                    effective_metric == "mixed" and upstream_metric in
                    {"multiple_choice_grade", "exact_str_match"}):
                metric_status = "aligned"
            else:
                metric_status = "nonpreferred_or_mismatch"

        per_benchmark.append({
            "benchmark_id": benchmark_id,
            "adapter": adapter,
            "scorer": scorer_name,
            "samples": len(samples),
            "nonempty_answers": len(samples) - empty,
            "empty_answers": empty,
            "correct": correct,
            "scorer_recheck_mismatches": mismatch,
            "upstream_metric": upstream_metric,
            "effective_metric": effective_metric,
            "metric_status": metric_status,
        })

    per_benchmark.sort(key=lambda row: row["benchmark_id"])
    metric_statuses = Counter(
        row["metric_status"] for row in per_benchmark if row["metric_status"]
    )
    summary = {
        "benchmarks": len(per_benchmark),
        "samples": sum(row["samples"] for row in per_benchmark),
        "nonempty_answers": sum(row["nonempty_answers"] for row in per_benchmark),
        "empty_answers": sum(row["empty_answers"] for row in per_benchmark),
        "deterministic_correct": sum(row["correct"] for row in per_benchmark),
        "scorer_recheck_mismatches": sum(
            row["scorer_recheck_mismatches"] for row in per_benchmark
        ),
        "tasks_with_empty_answers": sum(row["empty_answers"] > 0 for row in per_benchmark),
        "tasks_with_all_answers_empty": sum(
            row["empty_answers"] == row["samples"] for row in per_benchmark
        ),
        "providers": dict(providers),
        "groups": {key: dict(value) for key, value in sorted(groups.items())},
        "bigbench_metric_status": dict(metric_statuses),
        "parse_errors": parse_errors,
    }
    (args.batch_dir / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.batch_dir / "per_benchmark_audit.json").write_text(
        json.dumps(per_benchmark, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if parse_errors or summary["scorer_recheck_mismatches"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
