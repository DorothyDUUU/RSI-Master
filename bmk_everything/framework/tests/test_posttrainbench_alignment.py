from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import requests
import yaml


ROOT = Path(__file__).resolve().parents[2]


def load_runner(benchmark_id: str):
    path = ROOT / "format" / benchmark_id / "run_benchmark.py"
    spec = importlib.util.spec_from_file_location(f"test_{benchmark_id}_runner", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_aime_prompt_matches_posttrainbench_without_extra_reasoning_suffix():
    runner = load_runner("aime2025")
    prompt = runner.build_prompt("problem")
    assert prompt.startswith("Solve the following math problem step by step.")
    assert prompt.endswith("you do not need to use a \\boxed command.")
    assert runner.score_one("1.5", "2") is False


def test_gpqa_prompt_matches_inspect_multiple_choice_cot_template():
    runner = load_runner("gpqa")
    prompt = runner.build_prompt(
        {"question": "Q", "choices": ["one", "two", "three", "four"]}
    )
    assert "where LETTER is one of A,B,C,D. Think step by step before answering." in prompt
    assert prompt.endswith("A) one\nB) two\nC) three\nD) four")


def test_gsm8k_fewshots_are_a_system_message():
    runner = load_runner("gsm8k")
    messages = runner.build_messages("question", "fewshots")
    assert messages[0] == {"role": "system", "content": "fewshots"}
    assert messages[1]["role"] == "user"
    assert messages[1]["content"].endswith("Reasoning:")


def test_gsm8k_default_dataset_is_repository_local():
    runner = load_runner("gsm8k")
    resolved = Path(runner.resolve_repo_path("data/datasets/gsm8k"))
    assert resolved == ROOT / "data/datasets/gsm8k"


def test_posttrainbench_runtime_defaults_are_repository_local():
    from core import posttrainbench_inspect

    assert posttrainbench_inspect.DEFAULT_PTB_ROOT == ROOT / "external/PostTrainBench"
    assert posttrainbench_inspect.DEFAULT_RUNTIME_ROOT == ROOT / "external/PostTrainBenchRuntime"


def test_arena_null_content_becomes_one_empty_answer_instead_of_batch_crash():
    from core.posttrainbench_judge import _normalize_none_message_content

    response = requests.Response()
    response.status_code = 200
    response._content = json.dumps(
        {"choices": [{"message": {"content": None, "reasoning_content": "thought"}}]}
    ).encode()
    normalized = _normalize_none_message_content(response).json()
    assert normalized["choices"][0]["message"]["content"] == ""
    assert normalized["choices"][0]["message"]["reasoning_content"] == "thought"


def test_arena_judge_config_overrides_pinned_module_without_editing_it():
    from core.posttrainbench_judge import _apply_arena_judge_config

    module = SimpleNamespace(JUDGE_MODEL="gpt-5-mini", REASONING_EFFORT="medium")
    config = {
        "custom": {
            "judge_model": "deepseek-v4-flash",
            "reasoning_effort": None,
        }
    }
    _apply_arena_judge_config(module, config)
    assert module.JUDGE_MODEL == "deepseek-v4-flash"
    assert module.REASONING_EFFORT is None


def test_all_posttrainbench_benchmarks_are_runnable_and_pinned():
    for benchmark_id in (
        "arenahardwriting",
        "bfcl",
        "healthbench",
        "humaneval",
    ):
        directory = ROOT / "format" / benchmark_id
        manifest = json.loads((directory / "config.json").read_text())
        config = yaml.safe_load((directory / "config/config.yaml").read_text())
        assert manifest["benchmark-id"] == benchmark_id
        assert manifest["integration-status"] == "runnable"
        assert (directory / "run_benchmark.py").is_file()
        assert config["general"]["output_dir"].endswith(
            "/<BENCHMARK_ID>/<TRIAL_NAME>"
        )
