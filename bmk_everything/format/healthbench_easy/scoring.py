"""Canonical HealthBench Easy score aggregation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from dataset import HealthBenchExample
from protocol import GradedExample


@dataclass
class BenchmarkResult:
    accuracy: float
    stderr: float
    completed: int
    incomplete: int
    candidate_failures: int
    grader_failures: int
    rubric_count: int
    total_grader_calls: int
    by_theme: Dict[str, float]
    by_axis: Dict[str, float]


def _clip(value: float) -> float:
    return float(np.clip(value, 0, 1))


def compute_bootstrap_std(scores: List[float], n_bootstrap: int = 1000) -> float:
    if len(scores) < 2:
        return 0.0
    values = np.asarray(scores)
    rng = np.random.default_rng(42)
    bootstrap_means = [
        _clip(float(np.mean(rng.choice(values, size=len(values), replace=True))))
        for _ in range(n_bootstrap)
    ]
    return float(np.std(bootstrap_means))


def compute_scores_by_theme(
    completed: List[tuple[HealthBenchExample, GradedExample]],
) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for example, graded in completed:
        if graded.normalized_score is not None:
            grouped[example.theme].append(graded.normalized_score)
    return {
        theme: _clip(float(np.mean(scores)))
        for theme, scores in sorted(grouped.items())
    }


def compute_scores_by_axis(
    completed: List[tuple[HealthBenchExample, GradedExample]],
) -> Dict[str, float]:
    axis_items: Dict[str, List[tuple[float, bool]]] = defaultdict(list)
    for example, graded in completed:
        results = graded.grading_results
        for index, rubric in enumerate(example.rubrics):
            if index >= len(results):
                continue
            result = results[index]
            axis_items[rubric.axis].append(
                (rubric.points, result.criteria_met)
            )

    by_axis = {}
    for axis, items in axis_items.items():
        total_possible = sum(points for points, _ in items if points > 0)
        if total_possible == 0:
            continue
        achieved = sum(
            points for points, criteria_met in items if criteria_met
        )
        by_axis[axis] = _clip(achieved / total_possible)
    return dict(sorted(by_axis.items()))


def aggregate_scores(
    examples: List[HealthBenchExample],
    graded_examples: List[GradedExample],
) -> BenchmarkResult:
    completed_pairs = []
    candidate_failures = 0
    grader_failures = 0
    rubric_count = 0
    total_grader_calls = 0

    for index, graded in enumerate(graded_examples):
        if index >= len(examples):
            raise ValueError("graded examples exceed dataset examples")
        if graded.completed:
            completed_pairs.append((examples[index], graded))
        grader_failures += len(graded.grading_failures)
        total_grader_calls += len(graded.grading_results)
        total_grader_calls += len(graded.grading_failures)
        rubric_count += len(examples[index].rubrics)

    candidate_failures = len(examples) - len(graded_examples)
    scores = [
        graded.normalized_score
        for _, graded in completed_pairs
        if graded.normalized_score is not None
    ]
    accuracy = _clip(float(np.mean(scores))) if scores else 0.0
    stderr = compute_bootstrap_std(scores)
    completed = len(completed_pairs)

    return BenchmarkResult(
        accuracy=accuracy,
        stderr=stderr,
        completed=completed,
        incomplete=len(examples) - completed,
        candidate_failures=candidate_failures,
        grader_failures=grader_failures,
        rubric_count=rubric_count,
        total_grader_calls=total_grader_calls,
        by_theme=compute_scores_by_theme(completed_pairs),
        by_axis=compute_scores_by_axis(completed_pairs),
    )
