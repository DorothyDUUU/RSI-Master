import unittest
import importlib.util
from pathlib import Path
from unittest.mock import patch

import requests

RUNNER_PATH = Path(__file__).resolve().parents[1] / "run_benchmark.py"
SPEC = importlib.util.spec_from_file_location("bfcl_runner", RUNNER_PATH)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)

call_model = RUNNER.call_model
is_context_overflow_response = RUNNER.is_context_overflow_response
is_truncated = RUNNER.is_truncated
score_with_truncation = RUNNER.score_with_truncation


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self.payload = payload

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}", response=self)


class TruncationPolicyTest(unittest.TestCase):
    def test_detects_finish_reason_and_token_limit(self):
        self.assertTrue(is_truncated("length", {}, 100))
        self.assertTrue(is_truncated("stop", {"completion_tokens": 100}, 100))
        self.assertFalse(is_truncated("stop", {"completion_tokens": 99}, 100))

    def test_truncated_scoreable_response_gets_zero(self):
        self.assertFalse(score_with_truncation(True, True))
        self.assertFalse(score_with_truncation(False, True))
        self.assertIsNone(score_with_truncation(None, True))
        self.assertTrue(score_with_truncation(True, False))

    def test_context_overflow_detection_is_specific(self):
        overflow = FakeResponse(400, {"error": {"message": "This model's maximum context length is 32000 tokens. You requested 31600 output tokens."}})
        oversized_budget = FakeResponse(400, {"error": {"message": "max_tokens is too large"}})
        other_bad_request = FakeResponse(400, {"error": {"message": "invalid tool schema"}})
        self.assertTrue(is_context_overflow_response(overflow))
        self.assertTrue(is_context_overflow_response(oversized_budget))
        self.assertFalse(is_context_overflow_response(other_bad_request))

    @patch.object(RUNNER.requests, "post")
    def test_uses_highest_accepted_fallback_budget(self, post):
        post.side_effect = [
            FakeResponse(400, {"error": {"message": "Maximum context length exceeded after requesting output tokens"}}),
            FakeResponse(200, {"choices": []}),
        ]
        result = call_model(
            [{"role": "user", "content": "test"}],
            base_url="https://example.test/v1",
            model="Qwen3-8B",
            token="",
            tools=None,
            max_tokens=31600,
            fallback_max_tokens=[28672, 26624, 24576],
            temperature=0.0,
            timeout=10,
            retries=0,
        )
        self.assertEqual([call.kwargs["json"]["max_tokens"] for call in post.call_args_list], [31600, 28672])
        self.assertEqual(result["_bmk_requested_max_tokens"], 28672)

    @patch.object(RUNNER.requests, "post")
    def test_does_not_fallback_for_unrelated_bad_request(self, post):
        post.return_value = FakeResponse(400, {"error": {"message": "invalid tool schema"}})
        with self.assertRaises(requests.HTTPError):
            call_model(
                [{"role": "user", "content": "test"}],
                base_url="https://example.test/v1",
                model="Qwen3-8B",
                token="",
                tools=None,
                max_tokens=31600,
                fallback_max_tokens=[28672],
                temperature=0.0,
                timeout=10,
                retries=0,
            )
        self.assertEqual(post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
