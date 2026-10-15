#!/usr/bin/env python3
"""Batch-validate the local dataset tree against each benchmark's spec.

For every benchmark (generated + the three legacy integrations) this script
checks the local JSONL data is present, parseable, and maps onto the prompt /
target / choice transforms the generic runner will apply.  It is pure local IO
(no model calls, no network).

Outputs a JSON report plus a Markdown summary table into the registry docs dir:
    <repo>/data/benchmark-survey/registry/docs/DATASET_VALIDATION_<date>.{json,md}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_DIR / "framework"))

from core.generic_runner import _choices, _prompt, _targets  # noqa: E402


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_no}: {exc}") from exc
    return rows


def validate_generated(benchmark_id: str, spec: dict, data_dir: Path) -> dict:
    result = {"benchmark-id": benchmark_id, "type": "generated", "flags": []}
    source = spec.get("dataset") or {}
    if not str(source.get("path", "")).startswith("data/datasets"):
        result.update({"status": "not_local", "note": "spec still points at HF"})
        return result
    split = source.get("split", "default")
    jsonl_path = data_dir / benchmark_id / f"{split}.jsonl"
    if not jsonl_path.is_file():
        result.update({"status": "missing", "path": str(jsonl_path)})
        return result
    try:
        rows = jsonl_rows(jsonl_path)
    except Exception as exc:  # noqa: BLE001
        result.update({"status": "unparseable", "error": str(exc),
                       "path": str(jsonl_path)})
        return result
    declared = source.get("local_rows")
    if declared is not None and declared != len(rows):
        result["flags"].append(f"row_count_declared={declared}_actual={len(rows)}")
    scorer = spec.get("scorer")
    # Field-based choices mechanism (explicit options via fields).  A multiple_choice
    # scorer whose choices are NOT field-based is legitimate when the options are
    # embedded in the prompt text and the targets are single letters (e.g. gaokao).
    fields = spec.get("fields") or {}
    has_choices_mechanism = any(fields.get(key) for key in
                                ("choices", "choice_fields", "choices_merge", "choices_text"))
    empty_prompt = 0
    empty_targets = 0
    bad_choices = 0
    samples = []
    for row in rows[:5]:
        prompt = _prompt(row, spec)
        choices = _choices(row, spec)
        targets = _targets(row, spec)
        empty_prompt += int(not (prompt or "").strip())
        if scorer != "unscored":
            empty_targets += int(not targets)
        if scorer == "multiple_choice" and len(choices) < 2:
            # Only a real problem when the options were supposed to come from fields
            # (then a missing/empty choices field breaks the prompt) or when the
            # targets are not letters (nothing for a letter scorer to match).
            # gaokao-style rows (options inline in the question, targets are letters)
            # legitimately have no field-based choices.
            targets_are_letters = all(re.fullmatch(r"[A-Z]", str(t)) for t in targets)
            if has_choices_mechanism or not targets_are_letters:
                bad_choices += 1
        samples.append({
            "prompt_len": len(prompt or ""),
            "n_choices": len(choices),
            "n_targets": len(targets),
        })
    if empty_prompt:
        result["flags"].append(f"empty_prompt_in_sample={empty_prompt}")
    if empty_targets and scorer != "unscored":
        result["flags"].append(f"empty_targets_in_sample={empty_targets}")
    if bad_choices:
        result["flags"].append(f"mc_choices_lt2_in_sample={bad_choices}")
    result.update({
        "status": "ok" if not result["flags"] else "flagged",
        "rows": len(rows), "declared_rows": declared, "scorer": scorer,
        "adapter": spec.get("adapter"), "split": split, "samples": samples,
        "path": str(jsonl_path),
    })
    return result


def validate_legacy(benchmark_id: str, data_dir: Path) -> dict:
    """Validate a legacy integration's local data (symlinked to PostTrainBench).

    Data layout varies: aime2025 ships `test.jsonl`, gpqa ships `gpqa_main.jsonl`,
    gsm8k ships HuggingFace parquet under `main/`/`socratic/`.  So instead of
    hardcoding `test.jsonl`, discover actual files: prefer any `.jsonl` (parse it
    row-by-row); otherwise fall back to confirming parquet files exist and are
    non-empty (their own run_benchmark.py loads them, not the generic runner).
    """
    result = {"benchmark-id": benchmark_id, "type": "legacy", "flags": []}
    link = data_dir / benchmark_id
    if not link.is_symlink():
        result.update({"status": "not_linked", "note": "no local symlink"})
        return result
    jsonl_paths = sorted(link.glob("*.jsonl"))
    parquet_paths = sorted(link.glob("**/*.parquet"))
    if jsonl_paths:
        jsonl_path = jsonl_paths[0]
        try:
            rows = jsonl_rows(jsonl_path)
        except Exception as exc:  # noqa: BLE001
            result.update({"status": "unparseable", "error": str(exc),
                           "path": str(jsonl_path)})
            return result
        result.update({"status": "ok", "rows": len(rows), "path": str(jsonl_path),
                       "format": "jsonl", "other_jsonl": [str(p) for p in jsonl_paths[1:]]})
    elif parquet_paths:
        empty = [str(p) for p in parquet_paths if os.path.getsize(p) == 0]
        if empty:
            result.update({"status": "flagged", "path": empty[0],
                           "flags": ["empty_parquet"], "format": "parquet",
                           "parquet_files": [str(p) for p in parquet_paths]})
            return result
        result.update({"status": "ok", "rows": "parquet", "format": "parquet",
                       "path": str(parquet_paths[0]),
                       "parquet_files": [str(p) for p in parquet_paths]})
    else:
        result.update({"status": "missing", "note": "no .jsonl or .parquet under symlink"})
    return result


def validate_legacy_runner(benchmark_id: str, spec: dict, data_dir: Path) -> dict:
    """Validate a handwritten legacy_runner's local JSONL contract.

    The checks are generic: local path, declared row count, optional file hash,
    and required row/rubric fields supplied by the benchmark spec. Benchmark-
    specific semantic guards belong in the benchmark's own runner.
    """
    result = {"benchmark-id": benchmark_id, "type": "legacy_runner", "flags": []}
    source = spec.get("dataset") or {}
    local_path = source.get("path", "")
    if not str(local_path).startswith("data/datasets"):
        result.update({"status": "not_local", "note": "spec does not point to data/datasets"})
        return result

    split = source.get("split", "default")
    jsonl_path = data_dir / benchmark_id / f"{split}.jsonl"
    if not jsonl_path.is_file():
        result.update({"status": "missing", "path": str(jsonl_path)})
        return result

    try:
        rows = jsonl_rows(jsonl_path)
    except Exception as exc:  # noqa: BLE001
        result.update({"status": "unparseable", "error": str(exc),
                       "path": str(jsonl_path)})
        return result

    declared = source.get("local_rows")
    if declared is not None and declared != len(rows):
        result["flags"].append(
            f"row_count_declared={declared}_actual={len(rows)}"
        )

    expected_hash = source.get("sha256")
    if expected_hash:
        actual_hash = hashlib.sha256(jsonl_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            result["flags"].append(
                f"sha256_expected={expected_hash}_actual={actual_hash}"
            )

    validation = spec.get("validation") or {}
    required_row_fields = validation.get("required-row-fields") or []
    rubrics_field = validation.get("rubrics-field")
    required_rubric_fields = validation.get("required-rubric-fields") or []

    missing_rows = 0
    missing_rubrics = 0
    for row in rows:
        if any(field not in row for field in required_row_fields):
            missing_rows += 1
        if rubrics_field:
            rubric_rows = row.get(rubrics_field)
            if not isinstance(rubric_rows, list) or not rubric_rows:
                missing_rubrics += 1
                continue
            if any(
                any(rubric.get(field) is None for field in required_rubric_fields)
                for rubric in rubric_rows
            ):
                missing_rubrics += 1

    if missing_rows:
        result["flags"].append(f"missing_required_row_fields={missing_rows}")
    if missing_rubrics:
        result["flags"].append(f"missing_required_rubric_fields={missing_rubrics}")

    result.update({
        "status": "ok" if not result["flags"] else "flagged",
        "rows": len(rows),
        "declared_rows": declared,
        "split": split,
        "path": str(jsonl_path),
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format-dir", type=Path, default=REPO_DIR / "format")
    parser.add_argument("--data-dir", type=Path, default=REPO_DIR / "data/datasets")
    parser.add_argument("--docs-dir", type=Path,
                        default=REPO_DIR / "data/benchmark-survey/registry/docs")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="Only print what would run")
    args = parser.parse_args()
    format_dir = args.format_dir.resolve()
    data_dir = args.data_dir.resolve()
    docs_dir = args.docs_dir.resolve()
    docs_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for marker in sorted(format_dir.glob("*/.automated-integration")):
        benchmark_id = marker.parent.name
        spec = read_json(marker.parent / "spec.json")
        jobs.append((benchmark_id, spec))
    for benchmark_id in ("gsm8k", "gpqa", "aime2025"):
        jobs.append((benchmark_id, None))
    for spec_path in sorted(format_dir.glob("*/spec.json")):
        benchmark_id = spec_path.parent.name
        if (spec_path.parent / ".automated-integration").exists():
            continue
        spec = read_json(spec_path)
        if spec.get("adapter") == "legacy_runner":
            jobs.append((benchmark_id, spec))

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for benchmark_id, spec in jobs:
            if spec is None:
                futures[pool.submit(validate_legacy, benchmark_id, data_dir)] = benchmark_id
            elif spec.get("adapter") == "legacy_runner":
                futures[pool.submit(validate_legacy_runner, benchmark_id, spec, data_dir)] = benchmark_id
            else:
                futures[pool.submit(validate_generated, benchmark_id, spec, data_dir)] = benchmark_id
        for future in as_completed(futures):
            benchmark_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                result = {"benchmark-id": benchmark_id, "type": "error", "error": repr(exc),
                          "flags": []}
            results.append(result)
            print(f"[{result.get('status', 'error').upper():9}] {benchmark_id}", flush=True)

    results.sort(key=lambda item: item["benchmark-id"])
    date = time.strftime("%Y%m%d")
    json_path = docs_dir / f"DATASET_VALIDATION_{date}.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    md_path = docs_dir / f"DATASET_VALIDATION_{date}.md"
    lines = [f"# 本地数据集批量验证报告（{date}）", "",
             f"- 基准数量：{len(results)}",
             f"- 通过：{sum(1 for r in results if r['status'] == 'ok')}",
             f"- 标记：{sum(1 for r in results if r['status'] == 'flagged')}",
             f"- 失败/缺失/异常：{sum(1 for r in results if r['status'] not in {'ok', 'flagged'})}",
             "", "| benchmark-id | 类型 | 状态 | 行数 | 声明行数 | scorer | 标记 |", "|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(
            f"| {r['benchmark-id']} | {r.get('type', '-')} | {r.get('status', 'error')} "
            f"| {r.get('rows', '-')} | {r.get('declared_rows', '-')} "
            f"| {r.get('scorer', '-')} | {'; '.join(r.get('flags', []))} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ok = sum(1 for r in results if r["status"] == "ok")
    flagged = sum(1 for r in results if r["status"] == "flagged")
    failed = [r for r in results if r["status"] not in {"ok", "flagged"}]
    print(json.dumps({"checked": len(results), "ok": ok, "flagged": flagged,
                      "failed": len(failed), "failures": failed,
                      "json": str(json_path), "md": str(md_path)},
                     ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
