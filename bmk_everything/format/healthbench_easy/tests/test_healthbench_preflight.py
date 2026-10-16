from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


RUNNER_PATH = Path(__file__).resolve().parents[1] / "run_benchmark.py"
sys.path.insert(0, str(RUNNER_PATH.parent))
SPEC = importlib.util.spec_from_file_location("healthbench_easy_runner", RUNNER_PATH)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_missing_grader_credential_fails_before_generation(monkeypatch):
    monkeypatch.delenv("TEST_HEALTH_GRADER_KEY", raising=False)
    config = {"custom": {"grader_api_token_env": "TEST_HEALTH_GRADER_KEY"}}
    with pytest.raises(SystemExit, match="TEST_HEALTH_GRADER_KEY"):
        RUNNER.require_grader_token(config)


def test_configured_grader_credential_is_returned(monkeypatch):
    monkeypatch.setenv("TEST_HEALTH_GRADER_KEY", "secret")
    config = {"custom": {"grader_api_token_env": "TEST_HEALTH_GRADER_KEY"}}
    assert RUNNER.require_grader_token(config) == "secret"
