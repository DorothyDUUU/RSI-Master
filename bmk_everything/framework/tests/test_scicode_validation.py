import importlib.util
from pathlib import Path


RUNNER = Path(__file__).parents[2] / "format/scicode-verified/run_benchmark.py"
SPEC = importlib.util.spec_from_file_location("scicode_verified_runner", RUNNER)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _base_validation():
    return {
        "attempted_problems": 64,
        "completed_problems": 64,
        "generated_steps": 287,
        "expected_generated_steps": 287,
        "scored_steps": 287,
        "expected_scored_steps": 287,
        "unexpected_length_truncations": 0,
        "thinking_leaks": 0,
        "thinking": True,
        "reasoning_outputs": 0,
    }


def test_model_output_problems_are_warnings_not_fatal():
    validation = _base_validation()
    validation.update({
        "completed_problems": 63,
        "unexpected_length_truncations": 1,
        "thinking_leaks": 2,
    })
    warnings, fatal = MODULE._validation_findings(
        validation, {"2024", "2025"}, [{"2024", "2025"}]
    )
    assert "incomplete_problems" in warnings
    assert "unexpected_length_truncation" in warnings
    assert "thinking_leaked_into_final_content" in warnings
    assert fatal == []


def test_grading_environment_mismatch_remains_fatal():
    warnings, fatal = MODULE._validation_findings(
        _base_validation(), {"2024", "2025"}, [{"2024"}]
    )
    assert warnings == []
    assert fatal == ["grading_env_mismatch"]
