from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from core.arena_generation import generate_answers, generation_workers, install_arena_runtime


def fake_module(post, n=6):
    sessions = []

    class Session:
        def __init__(self):
            self.headers = {}
            sessions.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def post(self, *args, **kwargs):
            return post(*args, **kwargs)

    server = SimpleNamespace(start=Mock(return_value=1234), stop=Mock())
    module = SimpleNamespace(
        get_questions=lambda args: [{"uid": str(i), "prompt": str(i)} for i in range(n)],
        VLLMServer=lambda *args: server,
        requests=SimpleNamespace(Session=Session, RequestException=requests.RequestException),
        VLLM_GENERATION_RETRY=2, VLLM_REQUEST_TIMEOUT=300,
        limit_repetitions=lambda s: s,
        _make_metadata=lambda s: {"length": len(s)},
        shortuuid=SimpleNamespace(uuid=lambda: "id"),
        tqdm=lambda iterable, **kwargs: iterable,
    )
    return module, server, sessions


def response(text):
    result = requests.Response()
    result.status_code = 200
    result._content = json.dumps({
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"completion_tokens": 3},
    }).encode()
    return result


def args():
    return SimpleNamespace(model_path="model", model_alias="model", max_new_tokens=16384)


def test_parallel_requests_are_bounded_and_answers_stay_matched(tmp_path):
    barrier = threading.Barrier(3, timeout=5)
    lock = threading.Lock()
    active = peak = 0

    def post(endpoint, *, json, timeout):
        nonlocal active, peak
        assert json["max_tokens"] == 16384
        assert timeout == 300
        with lock:
            active += 1
            peak = max(peak, active)
        barrier.wait()  # A serial implementation fails instead of silently passing.
        with lock:
            active -= 1
        return response("<think>reason</think>answer " + json["messages"][0]["content"])

    module, server, sessions = fake_module(post)
    path, answers = generate_answers(module, args(), workers=3, output_dir=tmp_path)
    assert peak == 3
    assert list(answers) == [str(i) for i in range(6)]
    for uid, answer in answers.items():
        assert answer["messages"][-1]["content"]["answer"] == "answer " + uid
    assert len(path.read_text().splitlines()) == 6
    diagnostics = [json.loads(line) for line in (tmp_path / "generation.jsonl").read_text().splitlines()]
    assert all(d["finish_reason"] == "stop" and d["usage"]["completion_tokens"] == 3 for d in diagnostics)
    assert all(s.closed for s in sessions)
    server.stop.assert_called_once()


def test_retry_exhaustion_fails_and_cleans_up(tmp_path, monkeypatch):
    monkeypatch.setattr("core.arena_generation.time.sleep", lambda _: None)
    post = Mock(side_effect=requests.ConnectionError("unavailable"))
    module, server, sessions = fake_module(post, n=1)
    with pytest.raises(RuntimeError, match="uid 0 after 2 attempts"):
        generate_answers(module, args(), workers=1, output_dir=tmp_path)
    assert post.call_count == 2
    server.stop.assert_called_once()
    assert sessions[0].closed


def test_retry_then_null_content_is_saved(tmp_path, monkeypatch):
    monkeypatch.setattr("core.arena_generation.time.sleep", lambda _: None)
    post = Mock(side_effect=[requests.ConnectionError("retry"), response(None)])
    module, _, _ = fake_module(post, n=1)
    _, answers = generate_answers(module, args(), workers=1, output_dir=tmp_path)
    assert answers["0"]["messages"][-1]["content"]["answer"] == ""
    assert json.loads((tmp_path / "generation.jsonl").read_text())["attempts"] == 2


def test_runtime_counts_actual_questions_and_missing_judgments(tmp_path):
    module, _, _ = fake_module(None, n=3)
    original_generate = Mock()
    module.generate_answers = original_generate
    judgments = [{"games": [{"score": "A>B"}, {"score": "B>A"}]},
                 {"games": [{"score": "A>B"}, {"score": None}]}, None]
    original_judge = Mock(return_value=(None, judgments))
    module.judge_answers = original_judge
    counts, restore = install_arena_runtime(module, {}, tmp_path)
    path, result = module.judge_answers(args(), {})
    assert result is judgments
    assert counts == {"attempted": 3, "completed": 1, "incomplete": 2}
    assert len(path.read_text().splitlines()) == 3
    restore()
    assert module.generate_answers is original_generate
    assert module.judge_answers is original_judge


@pytest.mark.parametrize("value,expected", [(None, 16), (0, 16), (1, 1), (8, 8), (32, 32)])
def test_workers_config(value, expected):
    config = {} if value is None else {"custom": {"max_workers": value}}
    assert generation_workers(config) == expected


def test_negative_workers_rejected():
    with pytest.raises(ValueError):
        generation_workers({"custom": {"max_workers": -1}})


@pytest.mark.parametrize("failed_judge", [False, True])
def test_adapter_entrypoint_writes_real_counts_and_restores_hooks(tmp_path, monkeypatch, failed_judge):
    import sys
    import yaml
    from core import posttrainbench_judge as adapter

    module, _, _ = fake_module(lambda *a, **kw: response("answer"), n=2)
    original_generate = Mock()
    module.generate_answers = original_generate
    judgments = [
        {"games": [{"score": "A>B"}, {"score": "B>A"}]},
        {"games": [{"score": "A>B"}, {"score": None if failed_judge else "A=B"}]},
    ]
    original_judge = Mock(return_value=(None, judgments))
    module.judge_answers = original_judge
    module.JUDGE_MODEL = "judge"
    module.REASONING_EFFORT = None

    def official_main():
        _, answers = module.generate_answers(args())
        module.judge_answers(args(), answers)
        metrics_path = sys.argv[sys.argv.index("--json-output-file") + 1]
        from pathlib import Path
        Path(metrics_path).write_text('{"accuracy": 0.25}')

    module.main = official_main
    monkeypatch.setattr(adapter, "_ensure_runtime", lambda: None)
    monkeypatch.setattr(adapter, "_verify_pins", lambda: None)
    monkeypatch.setattr(adapter, "_load_official_module", lambda _: (module, tmp_path))
    config_path = tmp_path / "config.yaml"
    output = tmp_path / "output"
    config_path.write_text(yaml.safe_dump({
        "general": {"model": "model", "openai_api_base": "http://127.0.0.1:1234/v1", "output_dir": str(output)},
        "custom": {"max_workers": 2},
    }))
    monkeypatch.setattr(sys, "argv", ["runner", "--config", str(config_path), "--limit", "500"])
    saved_argv = sys.argv
    if failed_judge:
        with pytest.raises(SystemExit) as exc:
            adapter.main("arenahardwriting")
        assert exc.value.code == 1
    else:
        adapter.main("arenahardwriting")
    reward = json.loads((output / "reward.json").read_text())
    assert reward["meta_data"]["attempted"] == 2
    assert reward["meta_data"]["completed"] == (1 if failed_judge else 2)
    assert reward["meta_data"]["incomplete"] == int(failed_judge)
    assert reward["meta_data"]["accuracy"] == 0.25
    assert reward["reward"] == (None if failed_judge else 0.25)
    assert module.generate_answers is original_generate
    assert module.judge_answers is original_judge
    assert sys.argv is saved_argv


@pytest.mark.parametrize("text", [" plain answer ", "<think>reason</think> final", "<think>unfinished", ""])
def test_generation_matches_pinned_upstream_request_and_answer(tmp_path, text):
    # Execute the real upstream function without importing its heavy judge stack.
    import ast
    import os
    import time
    from pathlib import Path
    from typing import Dict, Optional

    source = Path(__file__).resolve().parents[2] / "external/PostTrainBench/src/eval/tasks/arenahardwriting/evaluate.py"
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "generate_answers")
    post = Mock(return_value=response(text))
    module, _, _ = fake_module(post, n=1)
    namespace = {**vars(module), "DATA_PATH": tmp_path, "time": time, "json": json, "os": os,
                 "Dict": Dict, "Optional": Optional}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    upstream_args = args()
    upstream_args.store_outputs = False
    _, upstream = namespace["generate_answers"](upstream_args)
    upstream_request = post.call_args
    _, concurrent = generate_answers(module, args(), workers=2, output_dir=tmp_path)
    assert post.call_args == upstream_request
    assert concurrent["0"]["messages"] == upstream["0"]["messages"]
    assert concurrent["0"]["metadata"] == upstream["0"]["metadata"]
