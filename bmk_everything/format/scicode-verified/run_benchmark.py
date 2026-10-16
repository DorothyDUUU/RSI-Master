#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "framework"))
from core.subset import select_subset, write_subset_manifest


BENCHMARK_ID = "scicode-verified"
EXPECTED_COMMIT = "ddab4a92f8d80a7113ab946628e994b52354d838"
EXPECTED_JSONL_MD5 = "5c604d8dbf52642bd94e13b92c8f52eb"
EXPECTED_H5_MD5 = "2b41a7df40ddc23ce651ec05b8ecb6f8"
SPLITS = {"with_background": "on", "without_background": "off"}
OFFICIAL_SKIPPED_STEPS = {("13", 5), ("62", 0), ("76", 2)}


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


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_ids(problems: list[dict], limit: int | None, fingerprint: str):
    selection = select_subset(
        problems,
        limit=limit,
        fingerprint=fingerprint,
        stable_key=lambda problem: problem["problem_id"],
    )
    return [str(problem["problem_id"]) for _, problem in selection.rows], selection.manifest


def _safe_tag(output_dir: Path) -> str:
    digest = hashlib.sha1(str(output_dir.resolve()).encode()).hexdigest()[:10]
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", output_dir.name).strip("-.") or "run"
    return f"bmk-{stem[:48]}-{digest}"


def _expected_scored_steps(problems: list[dict], selected_ids: list[str]) -> int:
    selected = set(selected_ids)
    return sum(
        1
        for problem in problems
        if str(problem["problem_id"]) in selected
        for index, _ in enumerate(problem.get("sub_steps") or [])
        if (str(problem["problem_id"]), index) not in OFFICIAL_SKIPPED_STEPS
    )


def _validation_findings(
    validation: dict,
    expected_env_labels: set[str],
    score_env_labels: list[set[str]],
) -> tuple[list[str], list[str]]:
    """Separate model-output diagnostics from evaluator-integrity failures."""
    warnings = [
        name
        for name, failed in (
            ("incomplete_problems", validation["completed_problems"] != validation["attempted_problems"]),
            ("generated_step_count_mismatch", validation["generated_steps"] != validation["expected_generated_steps"]),
            ("scored_step_count_mismatch", validation["scored_steps"] != validation["expected_scored_steps"]),
            ("unexpected_length_truncation", validation["unexpected_length_truncations"] != 0),
            ("thinking_leaked_into_final_content", validation["thinking_leaks"] != 0),
            ("reasoning_present_while_thinking_disabled", not validation["thinking"] and validation["reasoning_outputs"] != 0),
        )
        if failed
    ]
    fatal = [
        "grading_env_mismatch"
        for failed in (
            not expected_env_labels
            or any(labels != expected_env_labels for labels in score_env_labels),
        )
        if failed
    ]
    return warnings, fatal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-data", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    bench_dir = Path(__file__).resolve().parent
    custom = dict(config.get("custom") or {})
    official_root = _resolve(bench_dir, str(custom["official_root"]))
    python = _resolve_executable(bench_dir, str(custom["official_python"]))
    data = official_root / "scicode_verified/problems_test.jsonl"
    h5 = official_root / "scicode_verified/test_data_cleaned.h5"
    manifest_path = official_root / "scicode_verified/manifest.json"

    commit = subprocess.check_output(
        ["git", "-C", str(official_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != str(custom.get("official_commit", EXPECTED_COMMIT)):
        raise SystemExit(f"SciCode-Verified commit mismatch: {commit}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if _md5(data) != manifest.get("problems_test_jsonl_md5", EXPECTED_JSONL_MD5):
        raise SystemExit("SciCode-Verified JSONL manifest mismatch")
    if _md5(h5) != manifest.get("h5_md5", EXPECTED_H5_MD5):
        raise SystemExit("SciCode-Verified HDF5 manifest mismatch")
    problems = [json.loads(line) for line in data.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids, subset_manifest = _selected_ids(problems, args.limit, _md5(data))
    expected_scored_steps = _expected_scored_steps(problems, ids)
    split = str(_setting(config, "general", "split", "SPLIT_NAME", "with_background"))
    if split not in SPLITS:
        raise SystemExit(f"unsupported split {split!r}; choose {sorted(SPLITS)}")
    if args.dry_run:
        result = {
            "benchmark": BENCHMARK_ID,
            "split": split,
            "problems": len(ids),
            "scored_subproblems_full": 287,
            "dataset_version": manifest.get("version"),
            "official_commit": commit,
            "jsonl_md5": _md5(data),
            "h5_md5": _md5(h5),
        }
        if args.validate_data and ids:
            result["first_problem_id"] = ids[0]
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    model = str(_setting(config, "general", "model", "MODEL_NAME", ""))
    base_url = str(_setting(config, "general", "openai_api_base", "OPENAI_API_BASE", ""))
    token = str(_setting(config, "general", "openai_api_token", "OPENAI_API_TOKEN", "sk-local"))
    output_dir = Path(str(_setting(config, "general", "output_dir", "OUTPUT_DIR", "output")))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_subset_manifest(output_dir, subset_manifest)
    run_tag = _safe_tag(output_dir)
    command = [
        str(python), str(official_root / "eval_clean/run_deepseek_eval.py"),
        "--dataset", "cleaned",
        "--background", SPLITS[split],
        "--run", run_tag,
        "--model", model,
        "--provider", "local",
        "--workers", str(custom["max_workers"]),
        "--temperature", str(custom["temperature"]),
        "--top-p", str(custom["top_p"]),
        "--top-k", str(custom["top_k"]),
        "--min-p", str(custom["min_p"]),
    ]
    if custom.get("max_tokens") is not None:
        command += ["--max-tokens", str(custom["max_tokens"])]
    grading_envs = [str(entry) for entry in custom.get("grading_envs") or []]
    if grading_envs:
        command += ["--envs", *grading_envs]
    if custom.get("full_envs"):
        command += ["--full-envs"]
    if ids and len(ids) != len(problems):
        command += ["--only", ",".join(ids)]
    env = dict(os.environ)
    env.update({"LOCAL_API_URL": base_url, "LOCAL_API_KEY": token})
    process = subprocess.run(command, cwd=official_root, env=env)
    if process.returncode:
        raise SystemExit(process.returncode)

    run_root = official_root / "eval_clean/ds_runs" / run_tag
    candidates = list(run_root.glob(f"cleaned_*/{'with_bg' if split == 'with_background' else 'no_bg'}/results.json"))
    if len(candidates) != 1:
        raise SystemExit(f"expected one official results.json under {run_root}, found {candidates}")
    official_results = candidates[0]
    official_dir = official_results.parent
    copied = output_dir / "official_run"
    if copied.exists():
        shutil.rmtree(copied)
    shutil.copytree(official_dir, copied)
    payload = json.loads(official_results.read_text(encoding="utf-8"))
    summary = payload["summary"]
    result_rows = payload.get("results") or []
    attempted = len(ids)
    completed = int(summary.get("problems", len(result_rows)))
    incomplete = max(0, attempted - completed)
    raw_paths = list(official_dir.rglob("*.raw.json"))
    code_paths = list(official_dir.rglob("*.code.py"))
    score_paths = list(official_dir.rglob("*.score.json"))
    raw_rows = [json.loads(path.read_text(encoding="utf-8")) for path in raw_paths]
    score_rows = [json.loads(path.read_text(encoding="utf-8")) for path in score_paths]
    finish_reasons = {}
    for row in raw_rows:
        reason = str(row.get("finish_reason"))
        finish_reasons[reason] = finish_reasons.get(reason, 0) + 1
    expected_env_labels = {entry.split(":", 1)[0] for entry in grading_envs}
    score_env_labels = [set((row.get("per_env") or {}).keys()) for row in score_rows]
    configured_cap = custom.get("max_tokens")
    native_context_tokens = int(custom.get("native_context_tokens", 40960))
    length_rows = [row for row in raw_rows if row.get("finish_reason") == "length"]
    expected_cap_exhaustions = sum(
        (
            configured_cap is not None
            and int((row.get("usage") or {}).get("completion_tokens") or -1) == int(configured_cap)
        )
        or (
            configured_cap is None
            and int((row.get("usage") or {}).get("total_tokens") or -1) == native_context_tokens
        )
        for row in length_rows
    )
    validation = {
        "valid": False,
        "benchmark": BENCHMARK_ID,
        "split": split,
        "attempted_problems": attempted,
        "completed_problems": completed,
        "generated_steps": len(raw_rows),
        "expected_generated_steps": int(summary.get("steps_generated", -1)),
        "scored_steps": len(score_rows),
        "expected_scored_steps": int(summary.get("steps_total", -1)),
        "code_files": len(code_paths),
        "finish_reasons": finish_reasons,
        "length_truncations": finish_reasons.get("length", 0),
        "expected_cap_exhaustions": expected_cap_exhaustions,
        "unexpected_length_truncations": len(length_rows) - expected_cap_exhaustions,
        "thinking_leaks": sum("<think>" in (row.get("response") or "") for row in raw_rows),
        "reasoning_outputs": sum(bool(row.get("reasoning_content")) for row in raw_rows),
        "thinking": bool(custom.get("thinking", True)),
        "expected_grading_envs": sorted(expected_env_labels),
        "scores_with_all_envs": sum(labels == expected_env_labels for labels in score_env_labels),
        "max_tokens_policy": "remaining_native_context" if custom.get("max_tokens") is None else "fixed",
        "max_tokens": custom.get("max_tokens"),
        "native_context_tokens": native_context_tokens,
    }
    validation["warnings"], validation["fatal_violations"] = _validation_findings(
        validation, expected_env_labels, score_env_labels
    )
    validation["violations"] = validation["fatal_violations"]
    validation["valid"] = not validation["fatal_violations"]
    (output_dir / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not validation["valid"]:
        raise SystemExit(f"SciCode-Verified run rejected by validation gate: {validation['fatal_violations']}")
    with (output_dir / "samples.jsonl").open("w", encoding="utf-8") as sink:
        for row in result_rows:
            sample = dict(row)
            generated = int(sample.get("n_steps_generated") or 0)
            scored = int(sample.get("n_steps_scored") or 0)
            sample["attempted"] = True
            sample["valid"] = generated == scored
            sample["correct"] = bool(sample.get("problem_correct"))
            sample["score"] = 1 if sample["correct"] else 0
            sample["failure_origin"] = "model" if not sample["valid"] else None
            sample["failure_reason"] = "incomplete_generation" if not sample["valid"] else None
            sink.write(json.dumps(sample, ensure_ascii=False) + "\n")
        completed_ids = {str(row.get("problem_id")) for row in result_rows}
        for problem_id in ids:
            if problem_id not in completed_ids:
                sink.write(json.dumps({
                    "problem_id": problem_id,
                    "problem_correct": False,
                    "attempted": True,
                    "valid": False,
                    "correct": False,
                    "score": 0,
                    "zero_filled": True,
                    "failure_origin": "model",
                    "failure_reason": "incomplete_generation",
                }, ensure_ascii=False) + "\n")
    problems_correct = int(summary.get("problems_correct") or 0)
    steps_correct = int(summary.get("steps_correct") or 0)
    observed_problem_accuracy = float(summary.get("problem_accuracy") or 0.0)
    observed_step_accuracy = float(summary.get("step_accuracy") or 0.0)
    problem_accuracy = problems_correct / attempted if attempted else 0.0
    step_accuracy = steps_correct / expected_scored_steps if expected_scored_steps else 0.0
    reward = {
        "reward": problem_accuracy,
        "meta_data": {
            "accuracy": problem_accuracy,
            "observed_reward": observed_problem_accuracy,
            "problem_accuracy": problem_accuracy,
            "observed_problem_accuracy": observed_problem_accuracy,
            "step_accuracy": step_accuracy,
            "observed_step_accuracy": observed_step_accuracy,
            "correct": problems_correct,
            "steps_correct": steps_correct,
            "steps_total": expected_scored_steps,
            "observed_steps_total": int(summary.get("steps_total") or 0),
            "attempted": attempted,
            "completed": completed,
            "incomplete": incomplete,
            "zero_filled_problems": incomplete,
            "scoring_supported": True,
            "primary_metric": "whole_problem_accuracy",
            "with_background": split == "with_background",
            "official_commit": commit,
            "dataset_version": manifest.get("version"),
            "model": model,
            "max_tokens": custom.get("max_tokens"),
            "max_tokens_policy": validation["max_tokens_policy"],
            "native_context_tokens": native_context_tokens,
            "temperature": float(custom["temperature"]),
            "top_p": float(custom["top_p"]),
            "top_k": int(custom["top_k"]),
            "min_p": float(custom["min_p"]),
            "thinking": validation["thinking"],
            "validation": validation,
            "evaluation_status": "ok",
            "official_comparable": True,
        },
    }
    (output_dir / "reward.json").write_text(
        json.dumps(reward, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(reward, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
