import importlib.util
import os
from pathlib import Path

import pytest

# HorizonMath ships an external data checkout that is not vendored here. Point
# $HORIZONMATH_ROOT at it; the two tests that need it skip when it is absent.
HORIZONMATH_ROOT = Path(
    os.environ.get("HORIZONMATH_ROOT") or Path(__file__).parents[2] / "external/HorizonMath"
)
requires_horizonmath = pytest.mark.skipif(
    not (HORIZONMATH_ROOT / "data" / "problems_full.json").is_file(),
    reason="HorizonMath checkout not present (set $HORIZONMATH_ROOT)",
)


RUNNER = Path(__file__).parents[2] / "format/horizonmath/run_benchmark.py"
SPEC = importlib.util.spec_from_file_location("horizonmath_runner", RUNNER)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@requires_horizonmath
def test_select_problems_counts():
    import json
    root = HORIZONMATH_ROOT
    problems = json.loads((root / "data/problems_full.json").read_text())
    assert len(MODULE.select_problems(problems, "default", None)) == 113
    assert len(MODULE.select_problems(problems, "calibration", None)) == 10
    assert len(MODULE.select_problems(problems, "numeric", None)) == 71
    assert len(MODULE.select_problems(problems, "benchmark", None)) == 33
    assert len(MODULE.select_problems(problems, "construction", None)) == 9


@requires_horizonmath
def test_official_system_messages_are_loaded():
    root = HORIZONMATH_ROOT
    messages = MODULE.load_official_system_messages(root)
    assert set(messages) == {"ground_truth_computable", "benchmark_best_known", "new_construction"}
    assert all("proposed_solution" in value for value in messages.values())


def test_merge_sample_rows_preserves_generation_and_judgement():
    responses = [{
        "problem_id": "example",
        "problem_index": 3,
        "response": "final answer",
        "reasoning_content": "reasoning",
        "finish_reason": "stop",
        "usage": {"completion_tokens": 7},
        "provider": "openai-compatible",
        "model": "model-path",
        "timestamp": "2026-08-20T00:00:00+0800",
    }]
    evaluations = [{
        "problem_id": "example",
        "problem_index": 3,
        "success": True,
        "expected_value": "1.0",
        "actual_value": "1.0",
        "matching_digits": 20,
    }]

    prompts = [{
        "problem_id": "example",
        "problem_index": 3,
        "system": "system instructions",
        "prompt": "problem statement",
    }]

    sample = MODULE.merge_sample_rows(responses, evaluations, prompts)[0]

    assert sample["response"] == "final answer"
    assert sample["reasoning_content"] == "reasoning"
    assert sample["finish_reason"] == "stop"
    assert sample["usage"] == {"completion_tokens": 7}
    assert sample["correct"] is True
    assert sample["complete"] is True
    assert sample["predicted"] == "1.0"
    assert sample["target"] == "1.0"
    assert sample["matching_digits"] == 20
    assert sample["question"] == "problem statement"
    assert sample["system_prompt"] == "system instructions"


def test_merge_sample_rows_marks_missing_generation():
    evaluations = [{"problem_id": "missing", "problem_index": 4, "success": False}]

    sample = MODULE.merge_sample_rows([], evaluations)[0]

    assert sample["response"] == ""
    assert sample["complete"] is False
    assert sample["incomplete_reason"] == "missing_generation"
    assert sample["score"] == 0
    assert sample["model_output_valid"] is False
    assert sample["failure_origin"] == "model"
    assert sample["failure_reason"] == "missing_generation"


def test_merge_sample_rows_marks_length_as_model_zero_not_run_failure():
    responses = [{
        "problem_id": "truncated",
        "problem_index": 1,
        "response": "partial function",
        "finish_reason": "length",
    }]
    evaluations = [{
        "problem_id": "truncated",
        "problem_index": 1,
        "mode": "construction",
        "valid": False,
        "error_type": "extraction",
    }]

    sample = MODULE.merge_sample_rows(responses, evaluations)[0]

    assert sample["complete"] is True
    assert sample["correct"] is False
    assert sample["score"] == 0
    assert sample["model_output_valid"] is False
    assert sample["failure_origin"] == "model"
    assert sample["failure_reason"] == "length_truncated"


def test_merge_sample_rows_marks_transport_error_as_degraded_infrastructure():
    evaluations = [{
        "problem_id": "network-error",
        "problem_index": 2,
        "mode": "runtime",
        "error_type": "runtime",
        "error_message": "RuntimeError: HTTP 503: unavailable",
    }]

    sample = MODULE.merge_sample_rows([], evaluations)[0]

    assert sample["complete"] is False
    assert sample["score"] == 0
    assert sample["failure_origin"] == "infrastructure"
    assert sample["failure_reason"] == "missing_generation"


def test_merge_sample_rows_maps_validity_when_success_is_absent():
    evaluations = [{
        "problem_id": "construction",
        "problem_index": 5,
        "valid": False,
    }]

    sample = MODULE.merge_sample_rows([], evaluations)[0]

    assert sample["correct"] is False


def test_resolve_max_output_tokens_accepts_both_names():
    assert MODULE.resolve_max_output_tokens({"custom": {"max_tokens": 8192}}) == 8192
    assert MODULE.resolve_max_output_tokens({"custom": {"max_new_tokens": 4096}}) == 4096
    assert MODULE.resolve_max_output_tokens({"custom": {}}) == 8192


def test_resolve_max_output_tokens_rejects_conflicting_names():

    with pytest.raises(ValueError, match="disagree"):
        MODULE.resolve_max_output_tokens({
            "custom": {"max_tokens": 8192, "max_new_tokens": 2048}
        })
