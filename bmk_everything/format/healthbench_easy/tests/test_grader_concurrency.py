from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api import GraderClient, GraderRequestError
from protocol import GradingResult, RubricCriterion


def client(workers=32):
    return GraderClient(model="test", api_base="http://unused", api_token="test",
                        max_tokens=512, reasoning_max_tokens=512, timeout=1,
                        retries=2, max_concurrent_requests=workers)


def example(index, points):
    return SimpleNamespace(prompt_id=str(index), prompt=[], rubrics=[
        RubricCriterion(f"{index}-{j}", point, []) for j, point in enumerate(points)
    ])


def grade(conversation, response, rubric):
    return GradingResult(rubric.criterion_id, True, response, rubric.points)


def test_flat_queue_uses_32_workers_across_examples_and_preserves_order(monkeypatch):
    grader = client()
    barrier = threading.Barrier(32)
    lock = threading.Lock()
    active = peak = 0

    def controlled(*args):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            barrier.wait(timeout=10)
            return grade(*args)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(grader, "grade_criterion", controlled)
    examples = [example(i, [1]) for i in range(64)]
    responses = [SimpleNamespace(content=str(i)) for i in range(64)]
    progress, criteria = [], []
    results = grader.grade_examples_parallel(
        examples, responses, example_workers=2, criterion_workers=8,
        progress_callback=lambda done, total: progress.append((done, total)),
        criterion_progress_callback=lambda: criteria.append(1),
    )
    assert peak == 32
    assert [r.example_id for r in results] == [str(i) for i in range(64)]
    assert [r.grading_results[0].explanation for r in results] == [str(i) for i in range(64)]
    assert all(r.completed and r.normalized_score == 1 for r in results)
    assert progress == [(i, 64) for i in range(1, 65)]
    assert len(criteria) == 64


def test_scores_failures_and_rubric_order_match_single_example_grading(monkeypatch):
    grader = client(4)

    def controlled(conversation, response, rubric):
        if rubric.points == 9:
            raise GraderRequestError("retry exhausted", attempts=5)
        if rubric.points == 7:
            raise ValueError("bad result")
        return grade(conversation, response, rubric)

    monkeypatch.setattr(grader, "grade_criterion", controlled)
    examples = [example(0, [2, -1, 3]), example(1, [1, 9, 7])]
    responses = [SimpleNamespace(content="a"), SimpleNamespace(content="b")]
    expected = [grader.grade_example(e.prompt_id, e.prompt, r.content, e.rubrics, 1)
                for e, r in zip(examples, responses)]
    actual = grader.grade_examples_parallel(examples, responses)
    assert actual == expected
    assert actual[0].normalized_score == 4 / 5
    assert actual[0].completed
    assert not actual[1].completed
    assert [f.attempts for f in actual[1].grading_failures] == [5, 1]


def test_empty_inputs_and_empty_rubrics():
    grader = client()
    assert grader.grade_examples_parallel([], []) == []
    progress = []
    results = grader.grade_examples_parallel(
        [example(0, [])], [SimpleNamespace(content="a")],
        progress_callback=lambda *args: progress.append(args),
    )
    assert len(results) == 1
    assert results[0].normalized_score == 0
    assert progress == [(1, 1)]
    with pytest.raises(ValueError, match="same length"):
        grader.grade_examples_parallel([example(0, [])], [])


@pytest.mark.parametrize("workers", [0, -1])
def test_invalid_limit_fails_fast(workers):
    with pytest.raises(ValueError, match="positive"):
        client(workers)


def test_clients_have_independent_request_limits():
    first, second = client(1), client(2)
    assert first._rate_limit.acquire(blocking=False)
    assert not first._rate_limit.acquire(blocking=False)
    assert second._rate_limit.acquire(blocking=False)
    assert second._rate_limit.acquire(blocking=False)
    assert not second._rate_limit.acquire(blocking=False)
    first._rate_limit.release()
    second._rate_limit.release()
    second._rate_limit.release()
