"""Local HealthBench Easy dataset loading and integrity checks."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from protocol import RubricCriterion


CANONICAL_SHA256 = (
    "0febc8f013eb5d91e3b7e31bbbb0284ae6aa5fbd8cab68b81d63fa252837c0b0"
)
CANONICAL_SAMPLES = 245
CANONICAL_RUBRICS = 1873
CANONICAL_POSITIVE_RUBRICS = 1482
CANONICAL_NEGATIVE_RUBRICS = 391
CANONICAL_MIN_TURNS = 5
CANONICAL_REQUIRED_AXIS = "completeness"
CANONICAL_MAX_NEGATIVE_CRITERIA = 2


class DatasetIntegrityError(RuntimeError):
    """Raised when the frozen HealthBench Easy subset is not canonical."""


@dataclass(frozen=True)
class HealthBenchExample:
    prompt_id: str
    prompt: List[dict]
    rubrics: List[RubricCriterion]
    example_tags: List[str]
    ideal_completions_data: Optional[dict]
    canary: str

    @property
    def theme(self) -> str:
        for tag in self.example_tags:
            if tag.startswith("theme:"):
                return tag.split(":", 1)[1]
        return "unknown"


@dataclass(frozen=True)
class DatasetBundle:
    rows: List[HealthBenchExample]
    sha256: str
    metadata: dict


def _required_axis(rubrics: List[RubricCriterion], axis: str) -> bool:
    return any(rubric.axis == axis for rubric in rubrics)


def parse_row(raw: dict) -> HealthBenchExample:
    return HealthBenchExample(
        prompt_id=raw["prompt_id"],
        prompt=list(raw["prompt"]),
        rubrics=[
            RubricCriterion(
                criterion=rubric["criterion"],
                points=int(rubric["points"]),
                tags=list(rubric["tags"]),
            )
            for rubric in raw["rubrics"]
        ],
        example_tags=list(raw.get("example_tags") or []),
        ideal_completions_data=raw.get("ideal_completions_data"),
        canary=raw.get("canary"),
    )


def validate_dataset(rows: List[HealthBenchExample]) -> dict:
    errors = []
    unique_ids = {row.prompt_id for row in rows}
    rubrics = [rubric for row in rows for rubric in row.rubrics]
    positive = sum(rubric.points > 0 for rubric in rubrics)
    negative = sum(rubric.points < 0 for rubric in rubrics)
    zero = sum(rubric.points == 0 for rubric in rubrics)

    checks = {
        "samples": len(rows),
        "expected_samples": CANONICAL_SAMPLES,
        "rubrics": len(rubrics),
        "expected_rubrics": CANONICAL_RUBRICS,
        "duplicate_ids": len(rows) - len(unique_ids),
        "missing_prompts": sum(not row.prompt for row in rows),
        "missing_rubrics": sum(not row.rubrics for row in rows),
        "positive_rubrics": positive,
        "negative_rubrics": negative,
        "zero_point_rubrics": zero,
        "minimum_turns": min((len(row.prompt) for row in rows), default=0),
    }

    if checks["samples"] != checks["expected_samples"]:
        errors.append(
            f"expected {checks['expected_samples']} samples, found {checks['samples']}"
        )
    if checks["rubrics"] != checks["expected_rubrics"]:
        errors.append(
            f"expected {checks['expected_rubrics']} rubrics, found {checks['rubrics']}"
        )
    if checks["duplicate_ids"]:
        errors.append(f"{checks['duplicate_ids']} duplicate prompt IDs")
    if checks["missing_prompts"] or checks["missing_rubrics"]:
        errors.append("missing prompt or rubric data")
    if positive != CANONICAL_POSITIVE_RUBRICS or negative != CANONICAL_NEGATIVE_RUBRICS:
        errors.append("positive/negative rubric count differs from canonical subset")
    if zero:
        errors.append("zero-point rubric found")

    for row in rows:
        if len(row.prompt) < CANONICAL_MIN_TURNS:
            errors.append(
                f"{row.prompt_id} has fewer than {CANONICAL_MIN_TURNS} turns"
            )
        if not _required_axis(row.rubrics, CANONICAL_REQUIRED_AXIS):
            errors.append(
                f"{row.prompt_id} lacks required {CANONICAL_REQUIRED_AXIS} axis"
            )
        if sum(rubric.points < 0 for rubric in row.rubrics) > CANONICAL_MAX_NEGATIVE_CRITERIA:
            errors.append(f"{row.prompt_id} has more than 2 negative criteria")

    if errors:
        raise DatasetIntegrityError("; ".join(errors))

    return checks


def load_dataset(path: Path, expected_sha256: str) -> DatasetBundle:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"HealthBench Easy dataset not found at {path}")

    data = path.read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()
    if sha256.lower() != expected_sha256.lower():
        raise DatasetIntegrityError(
            f"dataset hash mismatch: expected {expected_sha256}, got {sha256}"
        )

    rows = []
    for line_number, line in enumerate(
        data.decode("utf-8").strip().split("\n"), start=1
    ):
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetIntegrityError(f"invalid JSON at line {line_number}: {exc}") from exc
        rows.append(parse_row(raw))

    checks = validate_dataset(rows)
    return DatasetBundle(
        rows=rows,
        sha256=sha256,
        metadata={
            "sha256": sha256,
            "rows": len(rows),
            "rubrics": checks["rubrics"],
            "positive_rubrics": checks["positive_rubrics"],
            "negative_rubrics": checks["negative_rubrics"],
            "minimum_turns": checks["minimum_turns"],
        },
    )


def select_examples(
    rows: List[HealthBenchExample],
    *,
    seed: int = 42,
    limit: Optional[int] = None,
    sample_indices: Optional[Iterable[int]] = None,
) -> List[HealthBenchExample]:
    """Select examples using PostTrainBench's deterministic seed-42 order."""
    ordered = list(rows)
    random.Random(seed).shuffle(ordered)

    indices = None
    if sample_indices is not None:
        indices = sorted(set(sample_indices))
        if any(index < 0 or index >= len(ordered) for index in indices):
            raise IndexError("sample index outside canonical shuffled dataset")
        return [ordered[index] for index in indices]

    if limit is None or limit < 0:
        return ordered
    return ordered[:limit]
