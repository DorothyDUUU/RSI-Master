"""Canonical HealthBench Easy data and grading protocol primitives."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import List, Optional


# Kept byte-for-byte equivalent to OpenAI simple-evals/healthbench_eval.py.
GRADER_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:

```json
{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteria_met": false
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{
  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
  "criteria_met": false
}
```

As another example, if the critera says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()


@dataclass(frozen=True)
class RubricCriterion:
    criterion: str
    points: int
    tags: List[str]

    @property
    def axis(self) -> str:
        for tag in self.tags:
            if tag.startswith("axis:"):
                return tag.split(":", 1)[1]
        return "unknown"

    @property
    def criterion_id(self) -> str:
        return self.criterion[:50].replace(" ", "_").lower()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GradingResult:
    criterion_id: str
    criteria_met: bool
    explanation: str
    weighted_score: float
    attempts: int = 1
    raw_response: Optional[str] = None


@dataclass
class GradingFailure:
    criterion_id: str
    error: str
    attempts: int


@dataclass
class GradedExample:
    example_id: str
    model_response: str
    grading_results: List[GradingResult] = field(default_factory=list)
    grading_failures: List[GradingFailure] = field(default_factory=list)
    total_score: Optional[float] = None
    max_possible_score: Optional[float] = None
    normalized_score: Optional[float] = None

    @property
    def completed(self) -> bool:
        return not self.grading_failures and self.normalized_score is not None


def parse_grader_json(json_string: str) -> dict:
    """Parse the canonical markdown-fenced JSON response."""
    json_cleaned = re.sub(r"^```json\s*|\s*```$", "", json_string.strip())
    try:
        return json.loads(json_cleaned)
    except json.JSONDecodeError:
        return {}


def format_conversation_for_grader(
    conversation: List[dict], model_response: str
) -> str:
    convo_with_response = conversation + [
        {"role": "assistant", "content": model_response}
    ]
    return "\n\n".join(
        f"{turn['role']}: {turn['content']}" for turn in convo_with_response
    )


def format_rubric_item(criterion_text: str, points: int) -> str:
    return f"[{points}] {criterion_text}"


def make_grader_prompt(
    conversation: List[dict],
    model_response: str,
    criterion: RubricCriterion,
) -> str:
    conversation_text = format_conversation_for_grader(
        conversation, model_response
    )
    rubric_text = format_rubric_item(criterion.criterion, criterion.points)
    return GRADER_TEMPLATE.replace(
        "<<conversation>>", conversation_text
    ).replace("<<rubric_item>>", rubric_text)


def is_reasoning_grader(grader_model: str) -> bool:
    return any(token in grader_model for token in ("gpt-5", "o1", "o3"))


MAX_REPETITIONS = 5


def limit_repetitions(text: str, max_reps: int = MAX_REPETITIONS) -> str:
    """Apply the PostTrainBench repetition limiter unchanged."""

    def _limit_consecutive_lines(txt: str) -> tuple:
        lines = txt.split("\n")
        result = []
        i = 0
        modified = False

        while i < len(lines):
            line = lines[i]
            count = 1
            j = i + 1
            while j < len(lines) and lines[j] == line:
                count += 1
                j += 1

            if count > max_reps:
                result.extend([line] * max_reps)
                modified = True
            else:
                result.extend([line] * count)
            i = j

        return "\n".join(result), modified

    def _limit_block_patterns(txt: str) -> tuple:
        lines = txt.split("\n")
        modified = False

        for block_size in range(2, min(30, len(lines) // 5 + 1)):
            i = 0
            new_lines = []
            last_end = 0

            while i <= len(lines) - block_size:
                block = lines[i : i + block_size]
                count = 1
                j = i + block_size
                while j + block_size <= len(lines):
                    next_block = lines[j : j + block_size]
                    if next_block == block:
                        count += 1
                        j += block_size
                    else:
                        break

                if count > max_reps:
                    new_lines.extend(lines[last_end:i])
                    for _ in range(max_reps):
                        new_lines.extend(block)

                    remaining_lines = lines[j:]
                    if remaining_lines:
                        is_partial_repeat = True
                        for k, rem_line in enumerate(remaining_lines):
                            if k >= len(block):
                                is_partial_repeat = False
                                break
                            block_line = block[k % len(block)]
                            if rem_line != block_line:
                                if not (
                                    rem_line.strip()
                                    and block_line.startswith(rem_line.strip())
                                ):
                                    if not (
                                        block_line.strip()
                                        and rem_line.startswith(block_line.strip())
                                    ):
                                        is_partial_repeat = False
                                        break
                        if is_partial_repeat:
                            last_end = len(lines)
                        else:
                            last_end = j
                    else:
                        last_end = j

                    i = len(lines)
                    modified = True
                else:
                    i += 1

            if modified:
                new_lines.extend(lines[last_end:])
                return "\n".join(new_lines), True

        return txt, False

    def _limit_regex_patterns(txt: str) -> tuple:
        modified = False

        while True:
            changed_this_round = False

            for min_len, max_len in [
                (3, 30),
                (30, 60),
                (60, 120),
                (120, 250),
                (250, 500),
            ]:
                pattern = rf"(.{{{min_len},{max_len}}}?)\1{{{max_reps},}}"

                def replace_func(match):
                    nonlocal changed_this_round, modified
                    unit = match.group(1)
                    full_match = match.group(0)
                    count = len(full_match) // len(unit)
                    if count > max_reps:
                        modified = True
                        changed_this_round = True
                        return unit * max_reps
                    return full_match

                txt = re.sub(pattern, replace_func, txt, flags=re.DOTALL)

            if not changed_this_round:
                break

        return txt, modified

    for _ in range(10):
        changed = False
        text, m1 = _limit_consecutive_lines(text)
        changed = changed or m1
        text, m2 = _limit_block_patterns(text)
        changed = changed or m2
        text, m3 = _limit_regex_patterns(text)
        changed = changed or m3
        if not changed:
            break

    return text


def postprocess_candidate_response(response: str) -> str:
    answer_text = (response or "").strip()
    if answer_text.startswith("<think>"):
        answer_text = answer_text.split("</think>", maxsplit=1)[-1].strip()
    return limit_repetitions(answer_text)
