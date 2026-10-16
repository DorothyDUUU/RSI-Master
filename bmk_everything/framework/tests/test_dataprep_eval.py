from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[2]
FORMAT = REPO_ROOT / "format"
sys.path.insert(0, str(REPO_ROOT / "framework"))

from automation.bootstrap_dataprep_eval import FILES, TARGETS  # noqa: E402
from core.dataprep_eval import (  # noqa: E402
    _judge,
    aggregate,
    balanced_accuracy,
    clean_legal,
    extract_choice,
    extract_tag,
    parse_lexglue,
    normalize_label,
    run,
)
from core.provider import ChatResult  # noqa: E402


class DataPrepEvalTests(unittest.TestCase):
    def test_paper_integrations_are_pinned_and_registered(self):
        expected = {"legalbench", "medcasereasoning", "ckqa"}
        registry = set(json.loads((REPO_ROOT / "paper_manifest.json").read_text())["benchmark_ids"])
        self.assertTrue(expected <= registry)
        for benchmark_id in expected:
            directory = FORMAT / benchmark_id
            for name in ("config.json", "spec.json", "SOURCE.json", "config/config.yaml",
                         "run_benchmark.py", "bootstrap.py", "BENCHMARK.md"):
                self.assertTrue((directory / name).is_file(), f"{benchmark_id}: missing {name}")
            spec = json.loads((directory / "spec.json").read_text())
            self.assertEqual(TARGETS[benchmark_id][2], spec["expected-rows"])
            self.assertEqual(40, len(spec["official-evaluation-revision"]))

    def test_source_checksums_and_expected_counts_are_fixed(self):
        self.assertEqual({"business", "law", "medicine"}, set(FILES))
        self.assertTrue(all(len(value) == 64 for value in FILES.values()))
        self.assertEqual(1621, sum(item[2] for item in TARGETS.values()))

    def test_deterministic_answer_parsers(self):
        self.assertEqual("B", extract_choice('{"answer":"B","confidence":"high"}'))
        self.assertEqual("C", extract_choice("Answer: C"))
        self.assertEqual("UCC", clean_legal("<think>x</think> The answer is UCC"))
        self.assertEqual("diagnosis", extract_tag("<answer>diagnosis</answer>", "answer"))
        self.assertEqual(["100170", "100191"], parse_lexglue('["100191", "100170"]', ["100170", "100191"]))
        self.assertEqual("governing laws", normalize_label("  Governing   Laws "))

    def test_ledgar_labels_are_compared_case_insensitively(self):
        from core.dataprep_eval import _score

        record = {"metadata": {"gold": "governing laws", "task_name": "ledgar"}}
        score = _score("lexglue", record, "Governing Laws", "fake", {})
        self.assertEqual("governing laws", score["gold"])
        self.assertEqual("governing laws", score["prediction"])
        self.assertTrue(score["correct"])

    def test_official_style_aggregates(self):
        legal = [
            {"task": "a", "gold": "yes", "prediction": "yes", "correct": True},
            {"task": "a", "gold": "yes", "prediction": "no", "correct": False},
            {"task": "a", "gold": "no", "prediction": "no", "correct": True},
        ]
        self.assertAlmostEqual(0.75, balanced_accuracy(legal))
        reward, meta = aggregate("legalbench", legal)
        self.assertAlmostEqual(0.75, reward)
        self.assertEqual("weighted_balanced_accuracy", meta["metric"])

        lex = [
            {"task": "eurlex", "gold": ["a", "b"], "prediction": ["a"], "correct": False},
            {"task": "ledgar", "gold": "x", "prediction": "x", "correct": True},
        ]
        reward, meta = aggregate("lexglue", lex)
        self.assertAlmostEqual(0.75, reward)
        self.assertEqual("average_of_subsets", meta["metric"])

        ledgar_with_empty_prediction = [
            {"task": "ledgar", "gold": "x", "prediction": "x", "correct": True},
            {"task": "ledgar", "gold": "y", "prediction": "", "correct": False},
        ]
        reward, meta = aggregate("lexglue", ledgar_with_empty_prediction)
        self.assertEqual(0.5, reward)
        self.assertEqual(2, meta["subsets"]["ledgar"]["total"])
        self.assertEqual(0, meta["subsets"]["ledgar"]["skipped"])

    def test_ckqa_fake_provider_end_to_end_reward_contract(self):
        record = {
            "messages": [{"role": "user", "content": "Choose one."}],
            "completion_args": {"temperature": 0, "max_tokens": 16},
            "metadata": {"source": "fincdm", "standard_answer": "B"},
        }
        completion = ChatResult(content='{"answer":"B"}', provider="fake", finish_reason="stop",
                                reasoning_content="", usage={"completion_tokens": 3})
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            config = Path(temporary) / "config.yaml"
            config.write_text(
                f"general:\n  model: fake-model\n  split: all\n  output_dir: {output}\n"
                "custom:\n  max_workers: 1\n  max_tokens: 32\n  timeout: 1\n",
                encoding="utf-8",
            )
            observed = {}
            def fake_chat(*args, **kwargs):
                observed["max_tokens"] = kwargs["max_tokens"]
                return completion
            with patch("core.dataprep_eval._load", return_value=[(0, record)]), \
                    patch("core.dataprep_eval.chat_detailed", side_effect=fake_chat):
                run(FORMAT / "ckqa", config, 1)
            reward = json.loads((output / "reward.json").read_text())
            self.assertEqual(1.0, reward["reward"])
            self.assertEqual(1, reward["meta_data"]["completed"])
            sample = json.loads((output / "samples.jsonl").read_text())
            self.assertTrue(sample["score"]["correct"])
            self.assertEqual(32, observed["max_tokens"])

    def test_truncated_candidate_is_scored_as_incorrect(self):
        record = {
            "messages": [{"role": "user", "content": "Choose one."}],
            "completion_args": {"temperature": 0, "max_tokens": 16},
            "metadata": {"source": "fincdm", "standard_answer": "A"},
        }
        completion = ChatResult(content="I considered A but", provider="fake", finish_reason="length",
                                reasoning_content="", usage={"completion_tokens": 16})
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            config = Path(temporary) / "config.yaml"
            config.write_text(
                f"general:\n  model: fake-model\n  split: all\n  output_dir: {output}\n"
                "custom:\n  max_workers: 1\n  timeout: 1\n",
                encoding="utf-8",
            )
            with patch("core.dataprep_eval._load", return_value=[(0, record)]), \
                    patch("core.dataprep_eval.chat_detailed", return_value=completion):
                run(FORMAT / "ckqa", config, 1)
            reward = json.loads((output / "reward.json").read_text())
            self.assertEqual(0.0, reward["reward"])
            self.assertEqual(0, reward["meta_data"]["completed"])
            self.assertEqual(1, reward["meta_data"]["scored"])
            self.assertEqual(1, reward["meta_data"]["truncated_as_incorrect"])
            self.assertEqual(0, reward["meta_data"]["incomplete"])
            sample = json.loads((output / "samples.jsonl").read_text())
            self.assertFalse(sample["score"]["correct"])
            self.assertTrue(sample["scored_as_incorrect"])

    def test_judge_uses_explicit_endpoint_without_changing_candidate_environment(self):
        completion = ChatResult(content="y", provider="fake", finish_reason="stop",
                                reasoning_content="", usage={})
        observed = {}

        def fake_chat(*args, **kwargs):
            observed["argument"] = kwargs["model"]
            observed["environment"] = os.environ.get("MODEL_NAME")
            observed["endpoint"] = kwargs["endpoint"]
            return completion

        env = {
            "MODEL_NAME": "candidate-local-path",
            "OPENAI_API_BASE": "http://candidate/v1",
            "OPENAI_API_TOKEN": "candidate-token",
            "JUDGE_OPENAI_API_BASE": "http://judge/v1",
            "JUDGE_OPENAI_API_TOKEN": "judge-token",
            "JUDGE_MODEL": "independent-judge",
        }
        with patch.dict(os.environ, env, clear=True), \
                patch("core.dataprep_eval.chat_detailed", side_effect=fake_chat):
            self.assertEqual("y", _judge([{"role": "user", "content": "check"}],
                                         "candidate-local-path", {}, 8))
            self.assertEqual("candidate-local-path", os.environ["MODEL_NAME"])
            self.assertEqual("http://candidate/v1", os.environ["OPENAI_API_BASE"])
            self.assertEqual("candidate-token", os.environ["OPENAI_API_TOKEN"])
        self.assertEqual("independent-judge", observed["argument"])
        self.assertEqual("candidate-local-path", observed["environment"])
        self.assertEqual("http://judge/v1", observed["endpoint"].base_url)
        self.assertEqual("JUDGE_OPENAI_API_TOKEN", observed["endpoint"].api_key_env)

    def test_judge_rejects_empty_or_truncated_decision(self):
        env = {
            "JUDGE_OPENAI_API_BASE": "http://judge/v1",
            "JUDGE_OPENAI_API_TOKEN": "judge-token",
            "JUDGE_MODEL": "independent-judge",
            "JUDGE_MIN_TOKENS": "256",
        }
        for completion in (
            ChatResult(content="", provider="fake", finish_reason="length",
                       reasoning_content="", usage={}),
            ChatResult(content="y", provider="fake", finish_reason="length",
                       reasoning_content="", usage={}),
        ):
            with self.subTest(content=completion.content), patch.dict(os.environ, env, clear=True), \
                    patch("core.dataprep_eval.chat_detailed", return_value=completion) as mocked:
                with self.assertRaisesRegex(RuntimeError, "no complete decision"):
                    _judge([{"role": "user", "content": "check"}], "candidate", {}, 8)
                self.assertEqual(256, mocked.call_args.kwargs["max_tokens"])


if __name__ == "__main__":
    unittest.main()
