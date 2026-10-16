import unittest
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_benchmark import (
    chat_completion,
    is_aborted,
    is_context_overflow_response,
    is_truncated,
    load_candidate_checkpoint,
    load_resume_state,
    record_candidate_is_truncated,
    scored_outcome,
    summarize_records,
)


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
    def test_detects_aborted_generation(self):
        self.assertTrue(is_aborted("abort"))
        self.assertFalse(is_aborted("stop"))

    def test_detects_finish_reason_and_token_limit(self):
        self.assertTrue(is_truncated("length", {}, 100))
        self.assertTrue(is_truncated("stop", {"completion_tokens": 100}, 100))
        self.assertFalse(is_truncated("stop", {"completion_tokens": 99}, 100))

    def test_record_truncation_detection_supports_paper_rejudge(self):
        self.assertTrue(
            record_candidate_is_truncated(
                {
                    "candidate": {
                        "finish_reason": "length",
                        "usage": {"completion_tokens": 100},
                        "requested_max_tokens": 100,
                    }
                }
            )
        )
        self.assertFalse(
            record_candidate_is_truncated(
                {
                    "candidate": {
                        "finish_reason": "stop",
                        "usage": {"completion_tokens": 99},
                        "requested_max_tokens": 100,
                    }
                }
            )
        )

    def test_truncation_scores_zero_but_transport_failure_is_incomplete(self):
        candidate = {"complete": True}
        judge = {"complete": False, "correct": None}
        self.assertEqual(scored_outcome(candidate, judge, True), (True, False))
        self.assertEqual(
            scored_outcome({"complete": False}, judge, False),
            (False, None),
        )

    def test_summary_always_returns_numeric_lower_bound(self):
        records = [
            {
                "complete": True,
                "correct": True,
                "candidate": {"complete": True, "truncated": False},
                "judge": {"complete": True},
            },
            {
                "complete": True,
                "correct": False,
                "candidate": {"complete": True, "truncated": True},
                "judge": {"complete": False},
            },
            {
                "complete": False,
                "correct": None,
                "candidate": {"complete": False, "truncated": False},
                "judge": {"complete": False},
            },
        ]
        summary = summarize_records(records)
        self.assertEqual(summary["accuracy"], 1 / 3)
        self.assertEqual(summary["observed_accuracy"], 1 / 2)
        self.assertEqual(summary["candidate_failures"], 1)
        self.assertEqual(summary["judge_failures"], 0)
        self.assertEqual(summary["evaluation_status"], "degraded")
        self.assertFalse(summary["official_comparable"])

    def test_candidate_checkpoint_validates_signature_and_row(self):
        rows = [{"id": "sample-a"}]
        signature = {"model": "model-a", "max_tokens": 100}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "candidates.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "signature": signature,
                        "index": 0,
                        "id": "sample-a",
                        "result": {"complete": True, "content": "answer"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            results = load_candidate_checkpoint(path, rows, signature)
            self.assertEqual(results[0]["content"], "answer")
            with self.assertRaises(RuntimeError):
                load_candidate_checkpoint(path, rows, {"model": "model-b"})

    def test_candidate_checkpoint_does_not_preserve_abort(self):
        rows = [{"id": "sample-a"}]
        signature = {"model": "model-a", "max_tokens": 100}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "candidates.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "signature": signature,
                        "index": 0,
                        "id": "sample-a",
                        "result": {"complete": True, "finish_reason": "abort"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertIsNone(load_candidate_checkpoint(path, rows, signature)[0])

    def test_context_overflow_detection_is_specific(self):
        overflow = FakeResponse(
            400,
            {
                "error": {
                    "message": "Maximum context length is 32000 tokens; requested output tokens exceed it"
                }
            },
        )
        unrelated = FakeResponse(400, {"error": {"message": "invalid request"}})
        self.assertTrue(is_context_overflow_response(overflow))
        self.assertFalse(is_context_overflow_response(unrelated))

    @patch("run_benchmark.requests.post")
    @patch("run_benchmark.time.sleep")
    def test_retries_aborted_generation(self, sleep, post):
        post.side_effect = [
            FakeResponse(
                200,
                {
                    "choices": [
                        {
                            "message": {"content": "unfinished"},
                            "finish_reason": "abort",
                        }
                    ],
                    "usage": {"completion_tokens": 10},
                },
            ),
            FakeResponse(
                200,
                {
                    "choices": [
                        {
                            "message": {"content": "ANSWER: A"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"completion_tokens": 20},
                },
            ),
        ]
        result = chat_completion(
            [{"role": "user", "content": "test"}],
            base_url="https://example.test/v1",
            token="",
            model="Qwen3.5-35B-A3B",
            max_tokens=65536,
            temperature=0.0,
            timeout=10,
            retries=2,
        )
        self.assertEqual(result["content"], "ANSWER: A")
        self.assertEqual(result["attempts"], 2)
        sleep.assert_called_once()

    @patch("run_benchmark.requests.post")
    def test_uses_highest_accepted_output_budget(self, post):
        post.side_effect = [
            FakeResponse(
                400,
                {
                    "error": {
                        "message": "Maximum context length exceeded by requested output tokens"
                    }
                },
            ),
            FakeResponse(
                200,
                {
                    "choices": [
                        {
                            "message": {"content": "ANSWER: A"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {},
                },
            ),
        ]
        result = chat_completion(
            [{"role": "user", "content": "test"}],
            base_url="https://example.test/v1",
            token="",
            model="Qwen3-8B",
            max_tokens=31600,
            fallback_max_tokens=[28672, 26624, 24576],
            temperature=0.0,
            timeout=10,
            retries=1,
        )
        self.assertEqual(
            [call.kwargs["json"]["max_tokens"] for call in post.call_args_list],
            [31600, 28672],
        )
        self.assertEqual(result["requested_max_tokens"], 28672)

    def test_resume_preserves_only_complete_nontruncated_rows(self):
        rows = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        previous = [
            {
                "index": 0,
                "id": "a",
                "complete": True,
                "candidate": {
                    "complete": True,
                    "finish_reason": "stop",
                    "content": "line\u2028separator",
                },
                "judge": {"complete": True},
                "correct": True,
            },
            {
                "index": 1,
                "id": "b",
                "complete": True,
                "candidate": {"complete": True, "finish_reason": "length"},
                "judge": {"complete": True},
                "correct": False,
            },
            {
                "index": 2,
                "id": "c",
                "complete": False,
                "candidate": {"complete": True, "finish_reason": "stop"},
                "judge": {"complete": False},
                "correct": None,
            },
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in previous) + "\n",
                encoding="utf-8",
            )
            preserved, candidates, judges = load_resume_state(path, rows)
        self.assertEqual(set(preserved), {0})
        self.assertIsNotNone(candidates[0])
        self.assertIsNone(candidates[1])
        self.assertIsNotNone(candidates[2])
        self.assertIsNotNone(judges[0])
        self.assertIsNone(judges[2])

        with TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in previous) + "\n",
                encoding="utf-8",
            )
            preserved, candidates, _ = load_resume_state(
                path, rows, preserve_truncated=True
            )
        self.assertEqual(set(preserved), {0, 1})
        self.assertIsNotNone(candidates[1])

    def test_resume_does_not_preserve_abort(self):
        rows = [{"id": "a"}]
        previous = [
            {
                "index": 0,
                "id": "a",
                "complete": True,
                "candidate": {"complete": True, "finish_reason": "abort"},
                "judge": {"complete": True},
                "correct": False,
            }
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            path.write_text(json.dumps(previous[0]) + "\n", encoding="utf-8")
            preserved, candidates, judges = load_resume_state(
                path, rows, preserve_truncated=True
            )
        self.assertEqual(preserved, {})
        self.assertIsNone(candidates[0])
        self.assertIsNone(judges[0])


if __name__ == "__main__":
    unittest.main()
