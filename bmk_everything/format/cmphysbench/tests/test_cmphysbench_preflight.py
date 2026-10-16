from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest


RUNNER_PATH = Path(__file__).resolve().parents[1] / "run_benchmark.py"
SPEC = importlib.util.spec_from_file_location("cmphysbench_runner", RUNNER_PATH)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_missing_scorer_python_fails_preflight(tmp_path):
    with pytest.raises(SystemExit, match="scorer Python is missing"):
        RUNNER._verify_scorer_runtime(tmp_path / "missing-python")


def test_current_python_passes_scorer_import_preflight():
    import sys

    pytest.importorskip("numpy")
    pytest.importorskip("sympy")
    RUNNER._verify_scorer_runtime(Path(sys.executable))


def test_empty_model_content_is_kept_for_scoring():
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": "", "reasoning_content": "unfinished reasoning"},
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }
    session = Mock()
    session.post.return_value = response

    record = RUNNER._call_model(
        {"id": 7, "context": "", "question": "question", "symbol": "x", "final_answer": ["x"]},
        base_url="http://localhost:8000/v1",
        token="test",
        model="test-model",
        custom={
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "max_tokens": 16384,
            "native_context_tokens": 32768,
            "request_retries": 0,
            "request_timeout": 1,
        },
        session=session,
    )

    assert record["llm_answer"] == [""]
    assert record["reasoning_content"] == "unfinished reasoning"
    assert record["finish_reason"] == "stop"
