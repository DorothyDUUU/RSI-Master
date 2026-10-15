import importlib.util
import json
from pathlib import Path


RUNNER = Path(__file__).resolve().parents[2] / "format" / "livecodebench" / "run_benchmark.py"
spec = importlib.util.spec_from_file_location("livecodebench_runner", RUNNER)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_recent_date_window(tmp_path):
    rows = [
        {"question_id": "old", "contest_date": "2024-12-31T00:00:00"},
        {"question_id": "new", "contest_date": "2025-01-01T00:00:00"},
        {"question_id": "last", "contest_date": "2025-04-30T12:00:00"},
        {"question_id": "future", "contest_date": "2025-05-01T00:00:00"},
    ]
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert [row["question_id"] for row in module.load_rows(path, "recent_2025")] == ["new", "last"]


def test_full_v6_keeps_all_rows(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({"question_id": "x", "contest_date": "2023-05-01T00:00:00"}) + "\n")
    assert len(module.load_rows(path, "full_v6")) == 1


def test_api_payload_can_disable_thinking(monkeypatch):
    class Response:
        def raise_for_status(self): pass
        def json(self):
            return {
                "choices": [{
                    "message": {
                        "content": "```python\npass\n```",
                        "reasoning_content": "brief reasoning",
                    },
                    "finish_reason": "stop",
                }],
                "usage": {"completion_tokens": 4},
            }
    class Session:
        def post(self, url, **kwargs):
            assert kwargs["json"]["chat_template_kwargs"] == {"enable_thinking": False}
            return Response()
    custom = {"temperature": 0, "max_tokens": 32, "samples_per_problem": 1,
              "enable_thinking": False, "request_timeout": 1}
    outputs, reasons, reasoning, usage = module.api_completion(
        "http://localhost/v1", "x", "m", [], custom, Session()
    )
    assert outputs == ["```python\npass\n```"]
    assert reasons == ["stop"]
    assert reasoning == ["brief reasoning"]
    assert usage == {"completion_tokens": 4}


def test_sandbox_evaluate_one_uses_run_code_api():
    class Response:
        def raise_for_status(self): pass
        def json(self):
            return {
                "status": "Success",
                "run_result": {
                    "status": "Finished",
                    "return_code": 0,
                    "stdout": module.SANDBOX_RESULT_PREFIX + json.dumps({
                        "results": [True], "metadata": {"execution time": 0.01}
                    }) + "\n",
                    "stderr": "",
                },
            }

    class Session:
        def post(self, url, **kwargs):
            assert url == "http://sandbox:18080/run_code"
            assert kwargs["json"]["language"] == "python"
            assert "testing_util.py" in kwargs["json"]["files"]
            return Response()

    sample = {"input_output": json.dumps({"inputs": ["1\n"], "outputs": ["1\n"]})}
    results, metadata = module.sandbox_evaluate_one(
        "http://sandbox:18080", sample, "print(input())", "encoded", 6, Session()
    )
    assert results == [True]
    assert metadata == {"execution time": 0.01}


def test_empty_extracted_code_is_a_scored_model_failure():
    def compute_metrics(results, k_list):
        assert results == {0: [[-2]]}
        assert k_list == [1]
        return {"pass@1": 0.0}

    metrics, results, metadata = module.sandbox_codegen_metrics(
        [{"input_output": json.dumps({"inputs": [""], "outputs": [""]})}],
        [[""]],
        base_url="http://sandbox:18080",
        testing_util_b64="encoded",
        workers=1,
        timeout=6,
        compute_metrics=compute_metrics,
    )
    assert metrics == {"pass@1": 0.0}
    assert results == {0: [[-2]]}
    detail = json.loads(metadata[0][0])
    assert detail["execution_incomplete"] is False
    assert detail["model_failure_reason"] == "no_code_extracted"
