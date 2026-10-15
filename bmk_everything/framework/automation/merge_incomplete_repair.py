#!/usr/bin/env python3
"""Create a non-destructive 2,000-row view with approved repair rows substituted."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


def latest_ok(path: Path) -> dict[str, dict]:
    rows = {}
    for row in jsonl(path):
        if row.get("ok"):
            rows[row["key"]] = row
    return rows


def latest(path: Path) -> dict[str, dict]:
    rows = {}
    for row in jsonl(path):
        rows[row["key"]] = row
    return rows


def merge_sample(original: dict, replacement: dict) -> dict:
    """Overlay model output while retaining canonical scoring metadata."""
    update = dict(replacement)
    if original.get("targets") and not update.get("targets"):
        update.pop("targets", None)
    merged = dict(original)
    merged.update(update)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format-dir", type=Path, required=True)
    parser.add_argument("--source-batch", type=Path, required=True)
    parser.add_argument("--repair-batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-repair-calls", type=int)
    parser.add_argument("--metadata-batch", type=Path,
                        help="Canonical 2,000-row batch used to restore target metadata")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    format_dir = args.format_dir.resolve()
    sys.path.insert(0, str(format_dir))
    sys.path.insert(0, str(format_dir.parent / "framework"))
    from core.scoring import (boolean_match, exact_match, json_match, label_match,
                              multiple_choice_match, numeric_match, squad_match)
    from core.generic_runner import _load_rows, _targets
    from automation.spec_fixes_data import SPEC_OVERRIDES, apply_spec_fixes
    scorers = {"exact_match": exact_match, "json": json_match, "label": label_match,
               "multiple_choice": multiple_choice_match, "numeric": numeric_match,
               "boolean": boolean_match, "squad": squad_match}

    source = latest(args.source_batch / "state.jsonl")
    metadata_source = latest(args.metadata_batch / "state.jsonl") if args.metadata_batch else None
    repairs = latest(args.repair_batch / "state.jsonl")
    catalog = [row["benchmark-id"] for row in read_json(format_dir / "benchmark.json")]
    if set(source) != set(catalog):
        raise SystemExit(f"source batch coverage differs from catalog: {len(source)}/{len(catalog)}")
    repair_calls = sum(int(row["calls"]) for row in repairs.values())
    expected_calls = args.expected_repair_calls
    if expected_calls is None:
        if len(repairs) != 101 or repair_calls != 375:
            raise SystemExit("repair batch is not the approved 101-job/375-call result")
    elif repair_calls != expected_calls:
        raise SystemExit(f"repair batch calls {repair_calls} != expected {expected_calls}")
    if args.output.exists():
        if not args.replace:
            raise SystemExit(f"output exists: {args.output}; pass --replace to rebuild")
        resolved = args.output.resolve()
        if resolved.name != "kimi27-answer-audit10-repaired-20260807":
            raise SystemExit(f"refusing to replace unexpected path: {resolved}")
        shutil.rmtree(resolved)
    args.output.mkdir(parents=True)

    state_rows = []
    summary = {"benchmarks": 0, "samples": 0, "repaired_samples": 0,
               "nonempty_answers": 0, "empty_answers": 0,
               "score_trustworthy_benchmarks": 0, "unscored_benchmarks": 0,
               "historical_mixed_benchmarks": 0}
    for benchmark_id in catalog:
        source_dir = Path(source[benchmark_id]["output_dir"])
        original = {int(row["index"]): dict(row) for row in jsonl(source_dir / "samples.jsonl")}
        if metadata_source:
            metadata_dir = Path(metadata_source[benchmark_id]["output_dir"])
            canonical = {int(row["index"]): dict(row)
                         for row in jsonl(metadata_dir / "samples.jsonl")}
            if set(canonical) != set(original):
                raise SystemExit(f"{benchmark_id}: metadata/source indices differ")
            original = {index: merge_sample(canonical[index], original[index])
                        for index in original}
        if len(original) != 10:
            raise SystemExit(f"{benchmark_id}: source sample count is {len(original)}, expected 10")
        repair = repairs.get(benchmark_id)
        repaired_indices: set[int] = set()
        if repair:
            samples_path = Path(repair.get("output_dir") or "") / "samples.jsonl"
            replacements = jsonl(samples_path) if samples_path.exists() else []
            planned_indices = set(int(index) for index in repair["indices"])
            actual_indices = {int(row["index"]) for row in replacements}
            if not actual_indices.issubset(planned_indices):
                raise SystemExit(f"{benchmark_id}: repair output contains unplanned indices")
            replacements = [row for row in replacements
                            if str(row.get("response") or "").strip()]
            repaired_indices = {int(row["index"]) for row in replacements}
            for row in replacements:
                index = int(row["index"])
                original[index] = merge_sample(original[index], row)

        spec_path = format_dir / benchmark_id / "spec.json"
        spec = apply_spec_fixes(read_json(spec_path)) if spec_path.exists() else None
        scorer_name = spec.get("scorer") if spec else "legacy"
        scoring_supported = scorer_name != "unscored"
        retained_historical = set(original) - repaired_indices
        historical_mixed = bool(spec and benchmark_id in SPEC_OVERRIDES and retained_historical)
        score_trustworthy = scoring_supported and not historical_mixed
        scorer = scorers.get(scorer_name)
        if spec and scorer_name != "unscored":
            missing_target_indices = [index for index, row in original.items()
                                      if not row.get("targets")]
            if missing_target_indices:
                dataset_rows = _load_rows(spec, max(missing_target_indices) + 1)
                for index in missing_target_indices:
                    original[index]["targets"] = _targets(dataset_rows[index], spec)
        rows = []
        for index in sorted(original):
            row = dict(original[index])
            row["provenance"] = "repair" if index in repaired_indices else "original"
            if scorer_name == "unscored":
                row["correct"] = None
            elif spec and scorer and str(row.get("response") or "").strip():
                row["correct"] = bool(scorer(row["response"], row.get("targets") or []))
            elif not spec and str(row.get("response") or "").strip():
                if benchmark_id in {"aime2025", "gsm8k"}:
                    row["correct"] = bool(numeric_match(
                        row["response"], [row.get("target")]))
                elif benchmark_id == "gpqa":
                    row["correct"] = bool(multiple_choice_match(
                        row["response"], [row.get("correct_letter")]))
            elif not str(row.get("response") or "").strip():
                row["correct"] = None
            rows.append(row)

        attempted = len(rows)
        completed = sum(bool(str(row.get("response") or "").strip()) for row in rows)
        incomplete = attempted - completed
        correct = sum(row.get("correct") is True for row in rows)
        observed_reward = correct / completed if completed and scoring_supported else None
        reward_value = observed_reward if not incomplete and score_trustworthy else None
        out_dir = args.output / "benchmarks" / benchmark_id
        out_dir.mkdir(parents=True)
        with (out_dir / "samples.jsonl").open("w", encoding="utf-8") as sink:
            for row in rows:
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
        reward = {"reward": reward_value, "meta_data": {
            "benchmark": benchmark_id, "metric": scorer_name,
            "attempted": attempted, "completed": completed, "incomplete": incomplete,
            "correct": correct, "observed_reward": observed_reward,
            "scoring_supported": scoring_supported, "score_trustworthy": score_trustworthy,
            "historical_mixed": historical_mixed, "repaired_samples": len(repaired_indices),
        }}
        (out_dir / "reward.json").write_text(json.dumps(reward, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
        state_rows.append({"key": benchmark_id, "ok": completed == 10,
                           "completed_samples": completed, "output_dir": str(out_dir.resolve()),
                           "source_output_dir": str(source_dir),
                           "repair_output_dir": repair.get("output_dir") if repair else None})
        summary["benchmarks"] += 1
        summary["samples"] += attempted
        summary["repaired_samples"] += len(repaired_indices)
        summary["nonempty_answers"] += completed
        summary["empty_answers"] += incomplete
        summary["score_trustworthy_benchmarks"] += int(score_trustworthy)
        summary["unscored_benchmarks"] += int(not scoring_supported)
        summary["historical_mixed_benchmarks"] += int(historical_mixed)

    with (args.output / "state.jsonl").open("w", encoding="utf-8") as sink:
        for row in state_rows:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                                encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["benchmarks"] != 200 or summary["samples"] != 2000:
        raise SystemExit("merged batch failed coverage contract")


if __name__ == "__main__":
    main()
