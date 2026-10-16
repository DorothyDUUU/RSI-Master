#!/usr/bin/env python3
"""HealthBench Professional handwritten runner for BmkEverything."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "framework"))
from core.subset import select_subset, write_subset_manifest

from api import CandidateClient, CandidateResult, GraderClient
from dataset import (
    CANONICAL_SHA256,
    HealthBenchExample,
    load_dataset,
    select_examples,
)
from protocol import (
    GradedExample,
    GradingFailure,
    postprocess_candidate_response,
)
from scoring import aggregate_scores


BENCHMARK_ID = "healthbench_professional"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OPENAI_BASE = "https://api.moonshot.cn/v1"
DEFAULT_ANTHROPIC_BASE = "https://api.anthropic.com/v1"
CANONICAL_GRADER_MODEL = "gpt-5-mini"
CANONICAL_GRADER_BASE = "https://api.openai.com/v1"
TRUNCATED_FINISH_REASONS = {"length", "max_tokens", "max_output_tokens"}


def is_placeholder(value) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("<")
        and value.endswith(">")
    )


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
    env_name = str(
        resolve_custom(config, "grader_api_token_env", "OPENAI_API_TOKEN")
    )
    return os.environ.get(env_name, "")


def resolved_config(config: dict, model: str, split: str, output_dir: Path) -> dict:
    result = copy.deepcopy(config)
    general = result.setdefault("general", {})
    openai_base = resolve_general(
        config, "openai_api_base", "OPENAI_API_BASE", DEFAULT_OPENAI_BASE
    )
    anthropic_base = resolve_general(
        config,
        "anthropic_api_base",
        "ANTHROPIC_API_BASE",
        DEFAULT_ANTHROPIC_BASE,
    )
    general.update(
        {
            "model": model,
            "split": split,
            "openai_api_base": openai_base,
            "openai_api_token": "<OPENAI_API_TOKEN>",
            "anthropic_api_base": anthropic_base,
            "anthropic_api_token": "<ANTHROPIC_API_TOKEN>",
            "output_dir": str(output_dir),
        }
    )
    return result


def parse_sample_indices() -> Optional[List[int]]:
    raw = os.environ.get("SAMPLE_INDICES")
    if raw is None or not raw.strip():
        return None
    indices = [int(item) for item in raw.split(",")]
    if any(index < 0 for index in indices) or len(indices) != len(set(indices)):
        raise ValueError(
            "SAMPLE_INDICES must contain unique non-negative integers"
        )
    return sorted(indices)


def candidate_is_truncated(result: CandidateResult, max_tokens: int) -> bool:
    finish_reason = str(result.finish_reason or "").lower()
    if finish_reason in TRUNCATED_FINISH_REASONS:
        return True
    completion_tokens = (result.usage or {}).get("completion_tokens")
    return (
        isinstance(completion_tokens, int)
        and completion_tokens >= max_tokens
    )


def evaluate_candidates(
    examples: List[HealthBenchExample],
    client: CandidateClient,
    max_workers: int,
) -> List[CandidateResult]:
    if max_workers <= 1:
        return [client.complete(example.prompt) for example in examples]

    results: List[Optional[CandidateResult]] = [None] * len(examples)
    bar = tqdm(total=len(examples), desc="candidate", unit="sample",
               mininterval=1, leave=True, dynamic_ncols=True)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(client.complete, example.prompt): index
            for index, example in enumerate(examples)
        }
        for future in as_completed(futures):
            bar.update(1)
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception as exc:
                results[index] = CandidateResult(
                    content="",
                    raw_content="",
                    provider="openai-compatible",
                    finish_reason=None,
                    reasoning_content="",
                    usage={},
                    latency_s=0.0,
                    retries=0,
                    complete=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
    bar.close()
    return [result for result in results if result is not None]


def configuration_failure_example(
    example: HealthBenchExample,
    candidate: CandidateResult,
    error: str,
) -> GradedExample:
    return GradedExample(
        example_id=example.prompt_id,
        model_response=candidate.content,
        grading_results=[],
        grading_failures=[
            GradingFailure(
                criterion_id=rubric.criterion_id,
                error=error,
                attempts=0,
            )
            for rubric in example.rubrics
        ],
        total_score=None,
        max_possible_score=sum(
            rubric.points for rubric in example.rubrics if rubric.points > 0
        ),
        normalized_score=None,
    )


def build_grader(
    config: dict,
    token: str,
) -> GraderClient:
    return GraderClient(
        model=str(resolve_custom(config, "grader_model", CANONICAL_GRADER_MODEL)),
        api_base=str(
            resolve_custom(config, "grader_api_base", CANONICAL_GRADER_BASE)
        ),
        api_token=token,
        max_tokens=int(resolve_custom(config, "grader_max_tokens", 32768)),
        reasoning_max_tokens=int(
            resolve_custom(config, "grader_reasoning_max_tokens", 49152)
        ),
        timeout=int(resolve_custom(config, "grader_timeout", 300)),
        retries=int(resolve_custom(config, "grader_retries", 5)),
        max_concurrent_requests=int(
            resolve_custom(config, "grader_max_workers", 64)
        ),
    )


def grader_protocol(config: dict) -> str:
    model = str(resolve_custom(config, "grader_model", CANONICAL_GRADER_MODEL))
    base = str(
        resolve_custom(config, "grader_api_base", CANONICAL_GRADER_BASE)
    ).rstrip("/")
    if model == CANONICAL_GRADER_MODEL and base == CANONICAL_GRADER_BASE.rstrip("/"):
        return "canonical"
    return "override"


def dry_run(config: dict) -> int:
    dataset_path = resolve_repo_path(
        str(resolve_custom(config, "dataset_path", "data/datasets/healthbench_professional"))
    )
    expected_sha = str(
        resolve_custom(config, "dataset_sha256", CANONICAL_SHA256)
    )
    bundle = load_dataset(dataset_path / "test.jsonl", expected_sha)
    print(
        json.dumps(
            {
                "benchmark-id": BENCHMARK_ID,
                "status": "ok",
                "dataset": bundle.metadata,
                "adapter": "legacy_runner",
                "scorer": "unscored",
                "required-metric": "rubric_score",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run HealthBench Professional through BmkEverything"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--split")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-data", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    model = str(
        resolve_general(config, "model", "MODEL_NAME", "kimi-k2.7-code")
    )
    split = str(
        args.split
        or resolve_general(config, "split", "SPLIT_NAME", "default")
    )
    if split != "default":
        raise SystemExit(
            f"error: unsupported split {split!r}; available: ['default']"
        )

    if args.dry_run or args.validate_data:
        raise SystemExit(dry_run(config))

    started = time.time()
    trial_name = os.environ.get("TRIAL_NAME") or time.strftime("%Y%m%d_%H%M%S")
    output_dir = resolve_output_dir(config, trial_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = resolve_repo_path(
        str(resolve_custom(config, "dataset_path", "data/datasets/healthbench_professional"))
    )
    expected_sha = str(
        resolve_custom(config, "dataset_sha256", CANONICAL_SHA256)
    )
    dataset = load_dataset(dataset_path / "test.jsonl", expected_sha)
    requested_indices = parse_sample_indices()
    if requested_indices is not None:
        # Preserve the historical repair-index contract: indices address the
        # canonical seed-42 shuffled order in this benchmark.
        seed = int(resolve_custom(config, "dataset_seed", 42))
        examples = select_examples(
            dataset.rows,
            seed=seed,
            limit=args.limit,
            sample_indices=requested_indices,
        )
        original_by_id = {row.prompt_id: index for index, row in enumerate(dataset.rows)}
        chosen = [original_by_id[row.prompt_id] for row in examples]
        subset_manifest = {
            "version": 1,
            "algorithm": "explicit-indices-legacy-seed42-order",
            "dataset_fingerprint": dataset.sha256,
            "population": len(dataset.rows),
            "requested_limit": args.limit,
            "selected_count": len(examples),
            "selected_indices": chosen,
            "stratum_counts": {},
        }
    else:
        selection = select_subset(
            dataset.rows,
            limit=args.limit,
            fingerprint=dataset.sha256,
            stable_key=lambda row: row.prompt_id,
            stratum=lambda row: next(
                (tag.split(":", 1)[1] for tag in row.example_tags if tag.startswith("specialty:")),
                "unknown",
            ),
        )
        examples = [row for _, row in selection.rows]
        subset_manifest = selection.manifest
    write_subset_manifest(output_dir, subset_manifest)
    total_rubrics = sum(len(example.rubrics) for example in examples)

    openai_base = str(
        resolve_general(
            config, "openai_api_base", "OPENAI_API_BASE", DEFAULT_OPENAI_BASE
        )
    )
    anthropic_base = str(
        resolve_general(
            config,
            "anthropic_api_base",
            "ANTHROPIC_API_BASE",
            DEFAULT_ANTHROPIC_BASE,
        )
    )
    anthropic_token = str(
        resolve_general(
            config, "anthropic_api_token", "ANTHROPIC_API_TOKEN", ""
        )
    )
    candidate_max_tokens = int(
        resolve_custom(config, "candidate_max_tokens", 65536)
    )
    candidate = CandidateClient(
        model=model,
        openai_base=openai_base,
        openai_token=resolve_openai_token(config),
        anthropic_base=anthropic_base,
        anthropic_token=anthropic_token,
        max_tokens=candidate_max_tokens,
        timeout=int(resolve_custom(config, "candidate_timeout", 300)),
        retries=int(resolve_custom(config, "candidate_retries", 3)),
    )

    print(
        f"[healthbench_professional] candidate={model} split={split} "
        f"samples={len(examples)} rubrics={total_rubrics}"
    )
    candidate_results = evaluate_candidates(
        examples,
        candidate,
        int(resolve_custom(config, "candidate_workers", 1)),
    )
    processed_results = []
    candidate_truncated = []
    for result in candidate_results:
        truncated = candidate_is_truncated(result, candidate_max_tokens)
        complete = result.complete and not truncated
        error = result.error
        if truncated:
            error = (
                "CandidateTruncated: response reached the configured output "
                f"limit ({candidate_max_tokens} tokens)"
            )
        processed_results.append(CandidateResult(
            content=(
                postprocess_candidate_response(result.content)
                if complete
                else ""
            ),
            raw_content=result.raw_content,
            provider=result.provider,
            finish_reason=result.finish_reason,
            reasoning_content=result.reasoning_content,
            usage=result.usage,
            latency_s=result.latency_s,
            retries=result.retries,
            complete=complete,
            error=error,
        ))
        candidate_truncated.append(truncated)
    candidate_results = processed_results
    candidate_success = sum(result.complete for result in candidate_results)
    print(f"[healthbench_professional] candidate completed={candidate_success}/{len(examples)}")

    gradable_indices = [
        index for index, result in enumerate(candidate_results) if result.complete
    ]
    graded: List[Optional[GradedExample]] = [None] * len(examples)
    grader_token = resolve_grader_token(config)
    if gradable_indices:
        grader = build_grader(config, grader_token)
        print(
            f"[healthbench_professional] grader={grader.model} "
            f"protocol={grader_protocol(config)}"
        )
        selected_examples = [examples[index] for index in gradable_indices]
        selected_candidates = [
            candidate_results[index] for index in gradable_indices
        ]
        selected_rubrics = sum(len(e.rubrics) for e in selected_examples)
        grading_bar = tqdm(total=selected_rubrics, desc="grading", unit="rubric",
                           mininterval=0.5, leave=True, dynamic_ncols=True)
        graded_batch = grader.grade_examples_parallel(
            selected_examples,
            selected_candidates,
            example_workers=int(resolve_custom(config, "example_workers", 4)),
            criterion_workers=int(
                resolve_custom(config, "criterion_workers", 8)
            ),
            criterion_progress_callback=lambda: grading_bar.update(1),
        )
        grading_bar.close()
        for index, graded_example in zip(gradable_indices, graded_batch):
            graded[index] = graded_example

    truncated_indices = {
        index for index, truncated in enumerate(candidate_truncated) if truncated
    }
    benchmark = aggregate_scores(
        examples,
        graded,
        zero_score_indices=truncated_indices,
    )

    samples_path = output_dir / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as sink:
        for index, example in enumerate(examples):
            candidate_result = candidate_results[index]
            graded_result = graded[index]
            model_output = (
                candidate_result.content
                if candidate_result.complete
                else ""
            )
            record = {
                "index": index,
                "id": example.prompt_id,
                "prompt": example.prompt,
                "rubrics": [rubric.to_dict() for rubric in example.rubrics],
                "ideal_completions_data": example.ideal_completions_data,
                "model_output_raw": candidate_result.raw_content,
                "model_output": model_output,
                "candidate": {
                    "provider": candidate_result.provider,
                    "latency_s": round(candidate_result.latency_s, 6),
                    "retries": candidate_result.retries,
                    "finish_reason": candidate_result.finish_reason,
                    "reasoning_content": candidate_result.reasoning_content,
                    "usage": candidate_result.usage,
                    "complete": candidate_result.complete,
                    "truncated": candidate_truncated[index],
                    "error": candidate_result.error,
                },
                "grader_results": (
                    [
                        {
                            "criterion_id": result.criterion_id,
                            "criteria_met": result.criteria_met,
                            "explanation": result.explanation,
                            "weighted_score": result.weighted_score,
                            "attempts": result.attempts,
                            "raw_response": result.raw_response,
                        }
                        for result in graded_result.grading_results
                    ]
                    if graded_result
                    else []
                ),
                "grader_failures": (
                    [
                        {
                            "criterion_id": failure.criterion_id,
                            "error": failure.error,
                            "attempts": failure.attempts,
                        }
                        for failure in graded_result.grading_failures
                    ]
                    if graded_result
                    else []
                ),
                "score": (
                    0.0
                    if candidate_truncated[index]
                    else (
                        graded_result.normalized_score if graded_result else None
                    )
                ),
                "total_score": (
                    0.0
                    if candidate_truncated[index]
                    else (graded_result.total_score if graded_result else None)
                ),
                "max_possible_score": (
                    sum(
                        rubric.points
                        for rubric in example.rubrics
                        if rubric.points > 0
                    )
                    if candidate_truncated[index]
                    else (
                        graded_result.max_possible_score
                        if graded_result
                        else None
                    )
                ),
                "completed": bool(
                    candidate_truncated[index]
                    or (graded_result and graded_result.completed)
                ),
                "candidate_failed": not candidate_result.complete,
                "error": (
                    candidate_result.error
                    or (
                        "; ".join(
                            item.error for item in graded_result.grading_failures
                        )
                        if graded_result
                        else None
                    )
                ),
            }
            sink.write(json.dumps(record, ensure_ascii=True) + "\n")

    rubric_score = benchmark.accuracy
    truncated_total = sum(candidate_truncated)
    reward = rubric_score if benchmark.incomplete == 0 else None
    meta_data = {
        "benchmark": BENCHMARK_ID,
        "benchmark_id": BENCHMARK_ID,
        "model": model,
        "candidate_provider": (
            "anthropic" if model.startswith("claude-") else "openai-compatible"
        ),
        "grader_model": str(
            resolve_custom(config, "grader_model", CANONICAL_GRADER_MODEL)
        ),
        "grader_api_base": str(
            resolve_custom(config, "grader_api_base", CANONICAL_GRADER_BASE)
        ),
        "grader_protocol": grader_protocol(config),
        "metric": "rubric_score",
        "split": split,
        "total": len(examples),
        "attempted": len(examples),
        "completed": benchmark.completed,
        "incomplete": benchmark.incomplete,
        "candidate_failures": benchmark.candidate_failures,
        "grader_failures": benchmark.grader_failures,
        "truncated_total": truncated_total,
        "rubric_count": total_rubrics,
        "grader_calls": benchmark.total_grader_calls,
        "rubric_score": rubric_score,
        "accuracy": rubric_score,
        "stderr": benchmark.stderr,
        "by_theme": benchmark.by_theme,
        "by_axis": benchmark.by_axis,
        "dataset": {
            "sha256": dataset.sha256,
            "revision": str(
                resolve_custom(
                    config,
                    "dataset_revision",
                    "349962fd46dd02343a0d8a606491baf59154ea1a",
                )
            ),
            "rows": len(dataset.rows),
            "selected_rows": len(examples),
        },
        "elapsed_s": round(time.time() - started, 3),
        "scoring_supported": True,
    }
    reward_path = output_dir / "reward.json"
    reward_path.write_text(
        json.dumps(
            {"reward": reward, "meta_data": meta_data},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    resolved_config_path = output_dir / "config.yaml"
    resolved_config_path.write_text(
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
                "status": "ok" if benchmark.incomplete == 0 else "incomplete",
                "reward": reward,
                "rubric_score": rubric_score,
                "accuracy": rubric_score,
                "completed": benchmark.completed,
                "incomplete": benchmark.incomplete,
                "candidate_failures": benchmark.candidate_failures,
                "grader_failures": benchmark.grader_failures,
                "truncated_total": truncated_total,
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if benchmark.incomplete:
        raise SystemExit(
            f"error: {benchmark.incomplete} incomplete HealthBench Professional samples"
        )
    if not math.isfinite(rubric_score):
        raise SystemExit("error: HealthBench Professional reward is not finite")


if __name__ == "__main__":
    main()
