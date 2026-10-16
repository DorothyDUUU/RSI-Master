import importlib.util
import sys
from pathlib import Path


RUNNER = Path(__file__).parents[2] / "format/arc-agi-1/arc_runner.py"
if not RUNNER.is_file():
    import pytest
    pytest.skip("ARC-AGI is outside the paper release scope", allow_module_level=True)
sys.path.insert(0, str(RUNNER.parent))
SPEC = importlib.util.spec_from_file_location("arc_agi_1_runner", RUNNER)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parseable_grid_is_valid():
    valid, reason = MODULE.classify_model_output("[[1, 2]]", "stop", [[1, 2]])
    assert valid is True
    assert reason is None


def test_parse_failure_is_model_zero_not_transport_failure():
    valid, reason = MODULE.classify_model_output("not json", "stop", None)
    assert valid is False
    assert reason == "parse_failure"


def test_length_is_invalid_even_if_grid_can_be_parsed():
    valid, reason = MODULE.classify_model_output("[[1, 2]]", "length", [[1, 2]])
    assert valid is False
    assert reason == "length_truncated"
