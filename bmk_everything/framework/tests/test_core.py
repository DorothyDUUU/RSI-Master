from __future__ import annotations

import sys
import json
import tempfile
import unittest
from unittest.mock import patch
from io import BytesIO
from collections import Counter
from pathlib import Path
from types import SimpleNamespace


REPO_DIR = Path(__file__).resolve().parents[2]
FRAMEWORK_DIR = REPO_DIR / "framework"
FORMAT_DIR = REPO_DIR / "format"
sys.path.insert(0, str(FRAMEWORK_DIR))

from core.catalog import discover, find
from core.config import deep_merge, redact
from core.provider import ProviderError, chat, chat_detailed, completion_from_payload
from core.generic_runner import _choices, _load_rows, _prompt, _targets, run as run_generic
from core.legacy_bridge import completion_fields, parse_sample_indices
from core.scoring import (boolean_match, exact_match, json_match, label_match,
                          multiple_choice_match, numeric_match, squad_match)
from automation.merge_incomplete_repair import merge_sample
from automation.spec_fixes_data import apply_spec_fixes
from automation.validate_local_datasets import validate_legacy_runner


class CoreTests(unittest.TestCase):
    @patch("core.generic_runner._load_rows")
    @patch("core.generic_runner.chat_detailed")
    def test_empty_model_output_counts_zero_without_invalidating_run(self, chat_mock, rows_mock):
        rows_mock.return_value = [
            {"question": "q1", "answer": "yes"},
            {"question": "q2", "answer": "yes"},
        ]
        chat_mock.side_effect = [
            SimpleNamespace(content="yes", reasoning_content="", finish_reason="stop",
                            usage={}, complete=True, provider="test"),
            SimpleNamespace(content="", reasoning_content="unfinished reasoning", finish_reason="length",
                            usage={}, complete=False, provider="test"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_dir = root / "bench"
            output_dir = root / "output"
            bench_dir.mkdir()
            (bench_dir / "spec.json").write_text(json.dumps({
                "benchmark-id": "test-bench",
                "adapter": "hf_exact_match",
                "dataset": {"path": "unused", "split": "test"},
                "scorer": "exact_match",
                "license": "test",
                "resource-class": "light",
                "fields": {"prompt": "question", "target": "answer"},
            }), encoding="utf-8")
            config_path = root / "config.yaml"
            config_path.write_text(
                f"general:\n  output_dir: {output_dir}\n  model: test\n"
                "custom:\n  max_workers: 1\n",
                encoding="utf-8",
            )
            run_generic(bench_dir, config_path, 2)
            reward = json.loads((output_dir / "reward.json").read_text())
            self.assertEqual(reward["reward"], 0.5)
            self.assertEqual(reward["meta_data"]["observed_reward"], 0.5)
            self.assertEqual(reward["meta_data"]["completed"], 2)
            self.assertEqual(reward["meta_data"]["incomplete"], 0)
            samples = [json.loads(line) for line in (output_dir / "samples.jsonl").read_text().splitlines()]
            self.assertEqual(samples[1]["model_failure_reason"], "length_without_answer")
            self.assertFalse(samples[1]["correct"])

    @patch("core.generic_runner._load_rows")
    @patch("core.generic_runner.chat_detailed")
    def test_request_error_uses_attempted_denominator_and_fails(self, chat_mock, rows_mock):
        rows_mock.return_value = [
            {"question": "q1", "answer": "yes"},
            {"question": "q2", "answer": "yes"},
        ]
        chat_mock.side_effect = [
            SimpleNamespace(content="yes", reasoning_content="", finish_reason="stop",
                            usage={}, complete=True, provider="test"),
            RuntimeError("provider unavailable"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_dir = root / "bench"
            output_dir = root / "output"
            bench_dir.mkdir()
            (bench_dir / "spec.json").write_text(json.dumps({
                "benchmark-id": "test-bench",
                "adapter": "hf_exact_match",
                "dataset": {"path": "unused", "split": "test"},
                "scorer": "exact_match",
                "license": "test",
                "resource-class": "light",
                "fields": {"prompt": "question", "target": "answer"},
            }), encoding="utf-8")
            config_path = root / "config.yaml"
            config_path.write_text(
                f"general:\n  output_dir: {output_dir}\n  model: test\n"
                "custom:\n  max_workers: 1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "evaluation incomplete"):
                run_generic(bench_dir, config_path, 2)
            reward = json.loads((output_dir / "reward.json").read_text())
            self.assertIsNone(reward["reward"])
            self.assertEqual(reward["meta_data"]["observed_reward"], 0.5)
            self.assertEqual(reward["meta_data"]["completed"], 1)
            self.assertEqual(reward["meta_data"]["incomplete"], 1)

    def test_catalog_excludes_template(self):
        manifests = discover(FORMAT_DIR)
        ids = {item["benchmark-id"] for item in manifests}
        self.assertNotIn("unique-benchmark-id", ids)
        self.assertTrue({"gsm8k", "gpqa", "aime2025"}.issubset(ids))
        # the catalogue grew beyond 200 (2026 domain-knowledge survey added 53);
        # assert consistency with the manifest rather than a hardcoded count
        self.assertEqual(ids, set(json.loads((REPO_DIR / "paper_manifest.json").read_text())["benchmark_ids"]))
        self.assertEqual(Counter(item["integration-status"] for item in manifests),
                         {"runnable": len(manifests)})

    def test_find_existing_benchmark(self):
        self.assertEqual(find(FORMAT_DIR, "gsm8k")["integration-status"], "runnable")

    def test_deep_merge_preserves_nested_defaults(self):
        merged = deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}})
        self.assertEqual(merged, {"a": {"b": 3, "c": 2}})

    def test_redact_does_not_mutate_input(self):
        config = {"general": {"openai_api_token": "secret", "model": "m"}}
        safe = redact(config)
        self.assertEqual(config["general"]["openai_api_token"], "secret")
        self.assertEqual(safe["general"]["openai_api_token"], "<OPENAI_API_TOKEN>")

    def test_deterministic_scorers(self):
        self.assertTrue(exact_match("ANSWER: Paris", ["Paris"]))
        self.assertTrue(numeric_match("final answer: 1,234.5", ["1234.5"]))
        self.assertTrue(multiple_choice_match("ANSWER: C", ["C"]))
        self.assertTrue(json_match('ANSWER: {"a": 1}', ['{"a": 1}']))
        self.assertTrue(boolean_match("ANSWER: No.", ["no"]))
        self.assertTrue(label_match("ANSWER: Generic—not protectable", ["generic"]))
        self.assertTrue(squad_match("ANSWER: The Denver Broncos.", ["Denver Broncos"]))
        self.assertTrue(squad_match("ANSWER: The passage does not say.",
                                    ["unanswerable", "no answer"]))

    def test_reviewed_target_transforms(self):
        squad = {"adapter": "hf_exact_match", "fields": {"target": "answers"},
                 "target-transform": "answers_text"}
        self.assertEqual(_targets({"answers": {"text": ["France", "France"]}}, squad),
                         ["France", "France"])
        gsm = {"adapter": "hf_numeric", "fields": {"target": "answer"},
               "target-transform": "gsm8k_final"}
        self.assertEqual(_targets({"answer": "work 16 then 9\n#### 18"}, gsm), ["18"])
        minerva = {"adapter": "hf_numeric", "fields": {"target": "solution"},
                   "target-transform": "boxed_final"}
        self.assertEqual(_targets({"solution": r"45/3600=\boxed{1.6}"}, minerva), ["1.6"])
        drop = {"adapter": "hf_exact_match", "fields": {"target": "answer"},
                "target-transform": "drop_answer"}
        self.assertEqual(_targets({"answer": {"spans": [], "number": "3", "date": {}}}, drop),
                         ["3"])

    def test_reviewed_choice_and_prompt_transforms(self):
        thai = apply_spec_fixes({"benchmark-id": "thai-onet", "adapter": "hf_exact_match",
                                 "fields": {}, "scorer": "exact_match"})
        row = {"instruction": "Choose blank 1", "input": "1. No\n2. Yes",
               "result": "2. Yes"}
        self.assertEqual(_choices(row, thai), ["No", "Yes"])
        self.assertEqual(_targets(row, thai), ["B"])
        xcopa = apply_spec_fixes({"benchmark-id": "xcopa", "adapter": "hf_exact_match",
                                  "fields": {}, "scorer": "exact_match"})
        prompt = _prompt({"premise": "It broke.", "choice1": "It fell.",
                          "choice2": "It sang.", "question": "cause", "label": 0}, xcopa)
        self.assertIn("most plausible cause", prompt)
        self.assertIn("A. It fell.", prompt)

    def test_calibrated_domain_benchmark_contracts(self):
        econ_spec = {
            "adapter": "hf_exact_match",
            "benchmark-id": "econlogicqa",
            "fields": {"choice_fields": ["A", "B", "C", "D"],
                       "prompt": "Question", "target": "Answer"},
            "prompt-transform": "event_sequence",
            "scorer": "exact_match",
        }
        econ_row = {
            "Question": "Order the events.",
            "A": "Policy is announced.", "B": "Prices adjust.",
            "C": "Firms respond.", "D": "A shock occurs.",
            "Answer": "D, A, C, B",
        }
        econ_prompt = _prompt(econ_row, econ_spec)
        self.assertIn("A. Policy is announced.", econ_prompt)
        self.assertIn("D. A shock occurs.", econ_prompt)
        self.assertIn("comma-separated", econ_prompt)
        self.assertEqual(_targets(econ_row, econ_spec), ["D, A, C, B"])
        self.assertTrue(exact_match("ANSWER: D, A, C, B", _targets(econ_row, econ_spec)))

        cti_spec = {
            "adapter": "hf_multiple_choice",
            "fields": {"choice_fields": ["Option A", "Option B", "Option C", "Option D"],
                       "prompt": "Question", "target": "GT"},
            "scorer": "multiple_choice",
        }
        cti_row = {
            "Question": "Which mitigation blocks untrusted applications?",
            "Option A": "Audit", "Option B": "Execution Prevention",
            "Option C": "OS Configuration", "Option D": "User Account Control",
            "GT": "B",
        }
        cti_prompt = _prompt(cti_row, cti_spec)
        self.assertEqual(cti_prompt.count("Return the single correct option letter"), 1)
        self.assertIn("B. Execution Prevention", cti_prompt)
        self.assertTrue(multiple_choice_match("ANSWER: B", _targets(cti_row, cti_spec)))

        lexam_spec = {
            "adapter": "hf_multiple_choice",
            "fields": {"choices": "choices", "prompt": "question", "target": "gold"},
            "scorer": "multiple_choice",
        }
        lexam_row = {
            "question": "Which statements are correct?",
            "choices": repr([f"choice {i}" for i in range(16)]),
            "gold": 15,
        }
        self.assertEqual(len(_choices(lexam_row, lexam_spec)), 16)
        self.assertEqual(_targets(lexam_row, lexam_spec), ["P"])
        self.assertIn("P. choice 15", _prompt(lexam_row, lexam_spec))

    def test_legitimate_none_choice_can_be_preserved(self):
        spec = {
            "adapter": "hf_multiple_choice",
            "fields": {"choices": "choices", "prompt": "question", "target": "answer"},
            "preserve-none-choice": True,
            "scorer": "multiple_choice",
            "target-transform": "choice_index",
        }
        row = {"question": "How many?", "choices": ["One", "Two", "Three", "None"],
               "answer": 3}
        self.assertEqual(_choices(row, spec), ["One", "Two", "Three", "None"])
        self.assertEqual(_targets(row, spec), ["D"])
        self.assertIn("D. None", _prompt(row, spec))

    def test_local_dataset_can_exclude_pinned_invalid_rows(self):
        spec = {
            "adapter": "hf_multiple_choice",
            "dataset": {"path": "data/datasets/cti-mcq", "split": "test"},
            "exclude-indices": [0],
        }
        rows = _load_rows(spec, 1)
        self.assertEqual(len(rows), 1)
        self.assertNotIn("preventing applications from running", rows[0]["Question"])

    def test_completion_metadata_preserves_incomplete_reasoning(self):
        result = completion_from_payload({
            "choices": [{"finish_reason": "length", "message": {
                "content": "", "reasoning_content": "unfinished reasoning"}}],
            "usage": {"completion_tokens": 2048},
        }, "moonshot")
        self.assertFalse(result.complete)
        self.assertEqual(result.finish_reason, "length")
        self.assertEqual(result.reasoning_content, "unfinished reasoning")
        self.assertEqual(result.usage["completion_tokens"], 2048)
        fields = completion_fields(result)
        self.assertTrue(fields["complete"])
        self.assertFalse(fields["answer_present"])
        self.assertEqual(fields["model_failure_reason"], "length_without_answer")
        self.assertIsNone(fields["incomplete_reason"])

    def test_targeted_sample_indices_are_validated(self):
        with patch.dict("os.environ", {"SAMPLE_INDICES": "3,1"}):
            self.assertEqual(parse_sample_indices(), [1, 3])
        with patch.dict("os.environ", {"SAMPLE_INDICES": "1,1"}):
            with self.assertRaises(ValueError):
                parse_sample_indices()

    def test_legacy_runner_dataset_validation_uses_spec_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            dataset_dir = data_dir / "legacy-test"
            dataset_dir.mkdir()
            row = {
                "prompt": [{"role": "user", "content": "question"}],
                "rubrics": [{"criterion": "answers", "points": 1, "tags": []}],
            }
            jsonl = dataset_dir / "default.jsonl"
            jsonl.write_text(json.dumps(row) + "\n", encoding="utf-8")
            spec = {
                "adapter": "legacy_runner",
                "dataset": {
                    "path": "data/datasets/legacy-test",
                    "split": "default",
                    "local_rows": 1,
                },
                "validation": {
                    "rubrics-field": "rubrics",
                    "required-row-fields": ["prompt", "rubrics"],
                    "required-rubric-fields": ["criterion", "points", "tags"],
                },
            }
            result = validate_legacy_runner("legacy-test", spec, data_dir)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["rows"], 1)

    def test_repair_overlay_preserves_targets(self):
        original = {"index": 1, "response": "", "targets": ["C"], "correct": None}
        replacement = {"index": 1, "response": "ANSWER: C", "targets": [],
                       "method": "finalizer"}
        merged = merge_sample(original, replacement)
        self.assertEqual(merged["targets"], ["C"])
        self.assertEqual(merged["response"], "ANSWER: C")

    @patch("core.generic_runner.urllib.request.urlopen")
    def test_bigbench_rows_use_dataset_server(self, urlopen):
        payload = b'{"rows":[{"row":{"inputs":"q","targets":["a"]}}]}'
        response = BytesIO(payload)
        response.__enter__ = lambda value: value
        response.__exit__ = lambda *args: None
        urlopen.return_value = response
        spec = {
            "adapter": "bigbench",
            "dataset": {
                "path": "tasksource/bigbench",
                "name": "example",
                "split": "train",
                "revision": "abc123",
            },
        }
        self.assertEqual(_load_rows(spec, 3), [{"inputs": "q", "targets": ["a"]}])
        request = urlopen.call_args.args[0]
        self.assertIn("datasets-server.huggingface.co/rows", request.full_url)
        self.assertIn("revision=abc123", request.full_url)

    @patch("core.provider._request")
    def test_content_filter_uses_fallback(self, request):
        request.side_effect = [ProviderError("content_filter"),
                               {"choices": [{"message": {"content": "ok"}}]}]
        with patch.dict("os.environ", {"KIMI_MOONSHOT_API_KEY": "x", "KIMI_KSYUN_API_KEY": "y"}, clear=True):
            response, alias = chat([{"role": "user", "content": "x"}], retries=0)
        self.assertEqual((response, alias), ("ok", "kimi-k2.7-code-ksyun"))
        self.assertEqual(request.call_count, 2)

    @patch("core.provider._request", side_effect=ProviderError("authentication failed", status=401))
    def test_non_filter_error_never_uses_fallback(self, request):
        with self.assertRaises(ProviderError):
            chat([{"role": "user", "content": "x"}], retries=0)
        self.assertEqual(request.call_count, 1)

    @patch("core.provider.time.sleep")
    @patch("core.provider._request")
    def test_network_error_retries_primary_without_fallback(self, request, sleep):
        request.side_effect = [
            ProviderError("temporary SSL EOF", retryable=True),
            {"choices": [{"message": {"content": "ok"}}]},
        ]
        with patch.dict("os.environ", {"KIMI_MOONSHOT_API_KEY": "x", "KIMI_KSYUN_API_KEY": "y"}, clear=True):
            response, alias = chat([{"role": "user", "content": "x"}], retries=1)
        self.assertEqual((response, alias), ("ok", "kimi-k2.7-code-moonshot"))
        self.assertEqual(request.call_count, 2)
        self.assertEqual({call.args[0].alias for call in request.call_args_list},
                         {"kimi-k2.7-code-moonshot"})

    @patch("core.provider._request")
    def test_instant_final_only_request(self, request):
        request.return_value = {"choices": [{"message": {"content": "ANSWER: A"}}]}
        env = {"KIMI_MOONSHOT_API_KEY": "x", "KIMI_KSYUN_API_KEY": "y",
               "KIMI_THINKING_TYPE": "disabled", "KIMI_FINAL_ONLY": "1"}
        with patch.dict("os.environ", env, clear=True):
            result = chat_detailed([{"role": "user", "content": "question"}], retries=0)
        payload = request.call_args.args[2]
        self.assertEqual(result.content, "ANSWER: A")
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["messages"][0]["role"], "system")

    @patch("core.provider._request")
    def test_ksyun_can_be_selected_as_primary(self, request):
        request.return_value = {"choices": [{"message": {"content": "ANSWER: A"}}]}
        env = {"KIMI_MOONSHOT_API_KEY": "x", "KIMI_KSYUN_API_KEY": "y",
               "KIMI_PRIMARY_PROVIDER": "ksyun"}
        with patch.dict("os.environ", env, clear=True):
            result = chat_detailed([{"role": "user", "content": "question"}], retries=0)
        self.assertEqual(result.provider, "kimi-k2.7-code-ksyun")
        self.assertEqual(request.call_args.args[0].alias, "kimi-k2.7-code-ksyun")

    @patch("core.provider._anthropic_request")
    def test_claude_model_uses_anthropic_branch(self, request):
        request.return_value = {
            "content": [{"type": "text", "text": "ANSWER: A"},
                        {"type": "tool_use", "name": "x"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        env = {"ANTHROPIC_API_BASE": "https://api.anthropic.com", "ANTHROPIC_API_TOKEN": "t"}
        with patch.dict("os.environ", env, clear=True):
            result = chat_detailed([{"role": "user", "content": "q"}],
                                   model="claude-sonnet-5", retries=0)
        self.assertEqual(result.content, "ANSWER: A")
        self.assertEqual(result.provider, "anthropic")
        self.assertEqual(result.finish_reason, "end_turn")
        payload = request.call_args.args[1]
        self.assertEqual(payload["model"], "claude-sonnet-5")
        self.assertEqual(request.call_args.args[0].base_url, "https://api.anthropic.com/v1")

    @patch("core.provider._request")
    def test_openai_base_custom_endpoint(self, request):
        request.return_value = {"choices": [{"message": {"content": "ANSWER: B"}}], "usage": {}}
        env = {"OPENAI_API_BASE": "https://vllm.internal/v1", "OPENAI_API_TOKEN": "sk-xyz"}
        with patch.dict("os.environ", env, clear=True):
            result = chat_detailed([{"role": "user", "content": "q"}], retries=0)
        self.assertEqual(result.content, "ANSWER: B")
        endpoint = request.call_args.args[0]
        self.assertEqual(endpoint.alias, "openai")
        self.assertEqual(endpoint.base_url, "https://vllm.internal/v1")

    @patch("core.provider._request")
    def test_model_name_env_override(self, request):
        request.return_value = {"choices": [{"message": {"content": "ok"}}], "usage": {}}
        env = {"OPENAI_API_BASE": "https://vllm.internal/v1", "OPENAI_API_TOKEN": "sk-xyz",
               "MODEL_NAME": "my-model"}
        with patch.dict("os.environ", env, clear=True):
            chat_detailed([{"role": "user", "content": "q"}], model="ignored", retries=0)
        self.assertEqual(request.call_args.args[2]["model"], "my-model")


if __name__ == "__main__":
    unittest.main()
