"""OpenAI-compatible and Anthropic API clients used by HealthBench."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, List, Optional

import requests

from protocol import (
    GradedExample,
    GradingFailure,
    GradingResult,
    RubricCriterion,
    is_reasoning_grader,
    make_grader_prompt,
    parse_grader_json,
)


class CandidateRequestError(RuntimeError):
    """Raised when a candidate API call cannot produce a response."""


class GraderRequestError(RuntimeError):
    def __init__(self, message: str, attempts: int):
        super().__init__(message)
        self.attempts = attempts


@dataclass
class CandidateResult:
    content: str
    raw_content: str
    provider: str
    finish_reason: Optional[str]
    reasoning_content: str
    usage: dict
    latency_s: float
    retries: int
    complete: bool
    error: Optional[str] = None


def _join_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text") or "")
            for item in content
            if isinstance(item, dict)
        )
    return str(content)


class CandidateClient:
    def __init__(
        self,
        *,
        model: str,
        openai_base: str,
        openai_token: str,
        anthropic_base: str,
        anthropic_token: str,
        max_tokens: int,
        timeout: int,
        retries: int,
    ):
        self.model = model
        self.openai_base = openai_base.rstrip("/")
        self.openai_token = openai_token
        self.anthropic_base = anthropic_base.rstrip("/")
        self.anthropic_token = anthropic_token
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()

    def complete(self, messages: List[dict]) -> CandidateResult:
        started = time.perf_counter()
        attempts = 0
        last_error: Optional[Exception] = None

        if self.model.startswith("claude-"):
            for attempt in range(1, self.retries + 1):
                attempts = attempt
                try:
                    payload = self._anthropic_payload(messages)
                    response = self.session.post(
                        f"{self.anthropic_base}/messages",
                        json=payload,
                        headers={
                            "x-api-key": self.anthropic_token,
                            "anthropic-version": "2023-06-01",
                            "Content-Type": "application/json",
                        },
                        timeout=self.timeout,
                    )
                    response.raise_for_status()
                    body = response.json()
                    content = "".join(
                        block.get("text", "")
                        for block in body.get("content") or []
                        if isinstance(block, dict) and block.get("type") == "text"
                    )
                    if not content.strip():
                        raise ValueError("candidate returned empty content")
                    return CandidateResult(
                        content=content.strip(),
                        raw_content=content,
                        provider="anthropic",
                        finish_reason=body.get("stop_reason"),
                        reasoning_content="",
                        usage=body.get("usage") or {},
                        latency_s=time.perf_counter() - started,
                        retries=attempt - 1,
                        complete=True,
                    )
                except (requests.RequestException, ValueError, KeyError) as exc:
                    last_error = exc
                    if attempt < self.retries:
                        time.sleep(2**attempt)
        else:
            for attempt in range(1, self.retries + 1):
                attempts = attempt
                try:
                    payload = {
                        "model": self.model,
                        "messages": messages,
                        "max_tokens": self.max_tokens,
                    }
                    response = self.session.post(
                        f"{self.openai_base}/chat/completions",
                        json=payload,
                        headers={
                            "Authorization": f"Bearer {self.openai_token}",
                            "Content-Type": "application/json",
                        },
                        timeout=self.timeout,
                    )
                    response.raise_for_status()
                    body = response.json()
                    choices = body.get("choices") or []
                    if not choices:
                        raise ValueError("candidate response missing choices")
                    message = choices[0].get("message") or {}
                    content = _join_content(message.get("content"))
                    reasoning = (
                        message.get("reasoning_content")
                        or message.get("reasoning")
                        or message.get("reasoning_text")
                        or ""
                    )
                    if not content.strip() and reasoning:
                        content = str(reasoning).strip()
                    if not content.strip():
                        raise ValueError("candidate returned empty content")
                    return CandidateResult(
                        content=content.strip(),
                        raw_content=content,
                        provider="openai-compatible",
                        finish_reason=choices[0].get("finish_reason"),
                        reasoning_content=reasoning,
                        usage=body.get("usage") or {},
                        latency_s=time.perf_counter() - started,
                        retries=attempt - 1,
                        complete=True,
                    )
                except (requests.RequestException, ValueError, KeyError) as exc:
                    last_error = exc
                    if attempt < self.retries:
                        time.sleep(2**attempt)

        provider = "anthropic" if self.model.startswith("claude-") else "openai-compatible"
        error = last_error or CandidateRequestError(
            f"candidate request failed after {attempts} attempts"
        )
        return CandidateResult(
            content="",
            raw_content="",
            provider=provider,
            finish_reason=None,
            reasoning_content="",
            usage={},
            latency_s=time.perf_counter() - started,
            retries=attempts - 1 if attempts else 0,
            complete=False,
            error=f"{type(error).__name__}: {error}",
        )

    def _anthropic_payload(self, messages: List[dict]) -> dict:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }


_rate_limit: Optional[threading.Semaphore] = None
_rate_limit_lock = threading.Lock()


def _set_rate_limit(max_concurrent: int) -> None:
    global _rate_limit
    with _rate_limit_lock:
        _rate_limit = threading.Semaphore(max_concurrent)


class GraderClient:
    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_token: str,
        max_tokens: int,
        reasoning_max_tokens: int,
        timeout: int,
        retries: int,
        max_concurrent_requests: int,
    ):
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_token = api_token
        self.max_tokens = max_tokens
        self.reasoning_max_tokens = reasoning_max_tokens
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()
        _set_rate_limit(max_concurrent_requests)

    def grade_criterion(
        self,
        conversation: List[dict],
        model_response: str,
        criterion: RubricCriterion,
    ) -> GradingResult:
        grader_prompt = make_grader_prompt(
            conversation, model_response, criterion
        )
        payload_fields = (
            {"max_completion_tokens": self.reasoning_max_tokens}
            if is_reasoning_grader(self.model)
            else {"temperature": 0, "max_tokens": self.max_tokens}
        )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": grader_prompt}],
            **payload_fields,
        }
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

        last_error: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            if _rate_limit is not None:
                _rate_limit.acquire()
            try:
                response = self.session.post(
                    f"{self.api_base}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                choices = response.json().get("choices") or []
                if not choices:
                    raise ValueError("grader response missing choices")
                message = choices[0].get("message") or {}
                raw_response = _join_content(message.get("content"))
                grading_dict = parse_grader_json(raw_response)
                if "criteria_met" in grading_dict:
                    label = grading_dict["criteria_met"]
                    if label is True or label is False:
                        criteria_met = bool(label)
                        return GradingResult(
                            criterion_id=criterion.criterion_id,
                            criteria_met=criteria_met,
                            explanation=grading_dict.get(
                                "explanation", "No explanation provided"
                            ),
                            weighted_score=(
                                criterion.points if criteria_met else 0
                            ),
                            attempts=attempt,
                            raw_response=raw_response,
                        )
                last_error = ValueError(
                    "grader returned malformed JSON or a non-boolean criteria_met"
                )
            except (requests.RequestException, ValueError, KeyError) as exc:
                last_error = exc
            finally:
                if _rate_limit is not None:
                    _rate_limit.release()

            if attempt < self.retries:
                time.sleep(2 * attempt)

        error = last_error or RuntimeError("grader retries exhausted")
        raise GraderRequestError(
            f"{criterion.criterion_id}: {type(error).__name__}: {error}",
            attempts=self.retries,
        )

    def grade_example(
        self,
        example_id: str,
        conversation: List[dict],
        model_response: str,
        rubrics: List[RubricCriterion],
        criterion_workers: int,
        criterion_progress_callback: Optional[Callable[[], None]] = None,
    ) -> GradedExample:
        results: List[Optional[GradingResult]] = [None] * len(rubrics)
        failures: List[GradingFailure] = []

        def grade_one(index: int, rubric: RubricCriterion):
            return index, self.grade_criterion(
                conversation, model_response, rubric
            )

        workers = min(max(int(criterion_workers), 1), len(rubrics))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(grade_one, index, rubric): index
                for index, rubric in enumerate(rubrics)
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    _, result = future.result()
                    results[index] = result
                except GraderRequestError as exc:
                    failures.append(
                        GradingFailure(
                            criterion_id=rubrics[index].criterion_id,
                            error=str(exc),
                            attempts=exc.attempts,
                        )
                    )
                except Exception as exc:
                    failures.append(
                        GradingFailure(
                            criterion_id=rubrics[index].criterion_id,
                            error=f"{type(exc).__name__}: {exc}",
                            attempts=1,
                        )
                    )
                if criterion_progress_callback:
                    criterion_progress_callback()

        successful = [result for result in results if result is not None]
        max_possible_score = sum(
            rubric.points for rubric in rubrics if rubric.points > 0
        )
        total_score = sum(result.weighted_score for result in successful)
        normalized_score = (
            total_score / max_possible_score if max_possible_score > 0 else 0.0
        )
        failures.sort(key=lambda item: item.criterion_id)
        return GradedExample(
            example_id=example_id,
            model_response=model_response,
            grading_results=successful,
            grading_failures=failures,
            total_score=total_score,
            max_possible_score=max_possible_score,
            normalized_score=normalized_score,
        )

    def grade_examples_parallel(
        self,
        examples: list,
        responses: List[CandidateResult],
        *,
        example_workers: int,
        criterion_workers: int,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        criterion_progress_callback: Optional[Callable[[], None]] = None,
    ) -> List[GradedExample]:
        if len(examples) != len(responses):
            raise ValueError("examples and responses must have the same length")

        results: List[Optional[GradedExample]] = [None] * len(examples)
        completed = 0
        workers = min(max(int(example_workers), 1), len(examples))

        def grade_one(index, example, candidate):
            return index, self.grade_example(
                example.prompt_id,
                example.prompt,
                candidate.content,
                example.rubrics,
                criterion_workers,
                criterion_progress_callback,
            )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    grade_one, index, example, responses[index]
                ): index
                for index, example in enumerate(examples)
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    _, graded = future.result()
                    results[index] = graded
                except Exception as exc:
                    example = examples[index]
                    results[index] = GradedExample(
                        example_id=example.prompt_id,
                        model_response=responses[index].content,
                        grading_failures=[
                            GradingFailure(
                                criterion_id="example",
                                error=f"{type(exc).__name__}: {exc}",
                                attempts=1,
                            )
                        ],
                    )
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(examples))

        return [result for result in results if result is not None]
