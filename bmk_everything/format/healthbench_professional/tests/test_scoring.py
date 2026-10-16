import unittest

from dataset import HealthBenchExample
from protocol import GradedExample, RubricCriterion
from scoring import aggregate_scores


def example(example_id: str, points: int = 1) -> HealthBenchExample:
    return HealthBenchExample(
        prompt_id=example_id,
        prompt=[{"role": "user", "content": "question"}],
        rubrics=[
            RubricCriterion(
                criterion="criterion",
                points=points,
                tags=["axis:professional_rubric"],
            )
        ],
        example_tags=[],
        ideal_completions_data=None,
        canary="",
    )


class AggregateScoresTest(unittest.TestCase):
    def test_truncation_is_zero_and_transport_failure_is_incomplete(self):
        examples = [example("ok"), example("truncated"), example("failed")]
        graded = [
            GradedExample(
                example_id="ok",
                model_response="answer",
                total_score=1.0,
                max_possible_score=1.0,
                normalized_score=1.0,
            ),
            None,
            None,
        ]

        result = aggregate_scores(examples, graded, zero_score_indices={1})

        self.assertEqual(result.accuracy, 0.5)
        self.assertEqual(result.completed, 2)
        self.assertEqual(result.incomplete, 1)
        self.assertEqual(result.candidate_failures, 2)
        self.assertEqual(result.rubric_count, 3)
        self.assertEqual(result.by_theme, {"unknown": 0.5})
        self.assertEqual(result.by_axis, {"professional_rubric": 0.0})


if __name__ == "__main__":
    unittest.main()
