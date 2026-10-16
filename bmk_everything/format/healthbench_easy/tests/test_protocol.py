import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import HealthBenchExample
from protocol import (
    GradedExample,
    GradingFailure,
    GradingResult,
    RubricCriterion,
    make_grader_prompt,
    parse_grader_json,
    postprocess_candidate_response,
)
from scoring import aggregate_scores


class ProtocolTests(unittest.TestCase):
    def test_grader_parser_requires_real_boolean(self):
        self.assertEqual(
            parse_grader_json('```json\n{"criteria_met": true}\n```')[
                "criteria_met"
            ],
            True,
        )
        self.assertEqual(
            parse_grader_json('{"criteria_met": "true"}'),
            {"criteria_met": "true"},
        )

    def test_grader_prompt_preserves_rubric_points_and_turn_roles(self):
        prompt = make_grader_prompt(
            [
                {"role": "user", "content": "Is smoking safe?"},
                {"role": "assistant", "content": "Yes."},
            ],
            "No, smoking is harmful.",
            RubricCriterion("Gives accurate safety guidance", 7, ["axis:accuracy"]),
        )
        self.assertIn("user: Is smoking safe?", prompt)
        self.assertIn("assistant: Yes.", prompt)
        self.assertIn("assistant: No, smoking is harmful.", prompt)
        self.assertIn("[7] Gives accurate safety guidance", prompt)

    def test_repetition_limiter_matches_canonical_behavior(self):
        text = "\n".join(["repeated line"] * 7)
        self.assertEqual(
            postprocess_candidate_response(text),
            "\n".join(["repeated line"] * 5),
        )

    def test_aggregation_uses_signed_points_and_positive_denominator(self):
        example = HealthBenchExample(
            prompt_id="sample",
            prompt=[{"role": "user", "content": "question"}],
            rubrics=[
                RubricCriterion("positive", 4, ["axis:completeness"]),
                RubricCriterion("negative", -2, ["axis:accuracy"]),
            ],
            example_tags=["theme:communication"],
            ideal_completions_data=None,
            canary="test",
        )
        graded = GradedExample(
            example_id="sample",
            model_response="answer",
            grading_results=[
                GradingResult("positive", True, "", 4),
                GradingResult("negative", False, "", 0),
            ],
            total_score=4,
            max_possible_score=4,
            normalized_score=1.0,
        )
        result = aggregate_scores([example], [graded])
        self.assertEqual(result.accuracy, 1.0)
        self.assertEqual(result.completed, 1)
        self.assertEqual(result.incomplete, 0)
        self.assertEqual(result.by_axis["completeness"], 1.0)

    def test_grader_failure_marks_example_incomplete(self):
        example = HealthBenchExample(
            prompt_id="sample",
            prompt=[{"role": "user", "content": "question"}],
            rubrics=[RubricCriterion("positive", 1, ["axis:accuracy"])],
            example_tags=["theme:communication"],
            ideal_completions_data=None,
            canary="test",
        )
        graded = GradedExample(
            example_id="sample",
            model_response="answer",
            grading_failures=[
                GradingFailure("positive", "malformed response", 5)
            ],
            normalized_score=0.0,
        )
        result = aggregate_scores([example], [graded])
        self.assertEqual(result.completed, 0)
        self.assertEqual(result.incomplete, 1)
        self.assertEqual(result.grader_failures, 1)


if __name__ == "__main__":
    unittest.main()
