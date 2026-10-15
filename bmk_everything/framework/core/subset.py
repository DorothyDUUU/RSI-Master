"""Deterministic, representative benchmark subset selection.

``limit`` is the only public control.  Partial runs use a versioned stable hash
order and, when a runner supplies a stratum, proportional stratification.
Explicit ``SAMPLE_INDICES`` continue to take precedence for repair workflows.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, Iterable, Sequence, TypeVar


T = TypeVar("T")
ALGORITHM = "stable-hash-stratified-v1"
_SEED = "bmkeverything-representative-subset-v1"


@dataclass(frozen=True)
class Selection(Generic[T]):
    rows: list[tuple[int, T]]
    manifest: dict


def explicit_sample_indices() -> list[int] | None:
    raw = os.environ.get("SAMPLE_INDICES")
    if raw is None or not raw.strip():
        return None
    values = [int(item) for item in raw.split(",")]
    if any(value < 0 for value in values) or len(values) != len(set(values)):
        raise ValueError("SAMPLE_INDICES must contain unique non-negative integers")
    return sorted(values)


def _digest(fingerprint: str, key: str, index: int) -> str:
    value = f"{_SEED}\0{fingerprint}\0{key}\0{index}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _quotas(group_sizes: dict[str, int], limit: int) -> dict[str, int]:
    total = sum(group_sizes.values())
    exact = {key: limit * size / total for key, size in group_sizes.items()}
    quotas = {key: min(size, math.floor(exact[key])) for key, size in group_sizes.items()}

    # If the budget permits, retain capability coverage for rare strata.
    if limit >= len(group_sizes):
        for key, size in group_sizes.items():
            if size and quotas[key] == 0:
                quotas[key] = 1

    while sum(quotas.values()) > limit:
        candidates = [key for key, count in quotas.items() if count > 1]
        key = min(candidates, key=lambda item: (exact[item] - quotas[item], item))
        quotas[key] -= 1
    while sum(quotas.values()) < limit:
        candidates = [key for key, size in group_sizes.items() if quotas[key] < size]
        key = max(candidates, key=lambda item: (exact[item] - quotas[item], item))
        quotas[key] += 1
    return quotas


def select_subset(
    rows: Sequence[T],
    *,
    limit: int | None,
    fingerprint: str,
    stable_key: Callable[[T], object],
    stratum: Callable[[T], object] | None = None,
    eligible_indices: Iterable[int] | None = None,
) -> Selection[T]:
    eligible = list(range(len(rows))) if eligible_indices is None else list(eligible_indices)
    if len(eligible) != len(set(eligible)) or any(i < 0 or i >= len(rows) for i in eligible):
        raise IndexError("eligible index outside dataset or duplicated")

    explicit = explicit_sample_indices()
    if explicit is not None:
        allowed = set(eligible)
        missing = [index for index in explicit if index not in allowed]
        if missing:
            raise IndexError(f"sample indices do not belong to selected split: {missing}")
        selected = explicit[:limit] if limit is not None and limit >= 0 else explicit
        method = "explicit-indices"
    elif limit is None or limit < 0 or limit >= len(eligible):
        selected = eligible
        method = "full"
    else:
        grouped: dict[str, list[int]] = {}
        for index in eligible:
            group = str(stratum(rows[index])) if stratum else "__all__"
            grouped.setdefault(group, []).append(index)
        quotas = _quotas({key: len(value) for key, value in grouped.items()}, limit)
        selected = []
        for group, indices in grouped.items():
            ordered = sorted(
                indices,
                key=lambda index: (
                    _digest(fingerprint, str(stable_key(rows[index])), index),
                    index,
                ),
            )
            selected.extend(ordered[:quotas[group]])
        selected.sort()
        method = ALGORITHM if stratum else "stable-hash-v1"

    counts: dict[str, int] = {}
    for index in selected:
        group = str(stratum(rows[index])) if stratum else "__all__"
        counts[group] = counts.get(group, 0) + 1
    manifest = {
        "version": 1,
        "algorithm": method,
        "dataset_fingerprint": fingerprint,
        "population": len(eligible),
        "requested_limit": limit,
        "selected_count": len(selected),
        "selected_indices": selected,
        "stratum_counts": dict(sorted(counts.items())),
    }
    canonical = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    manifest["manifest_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return Selection(rows=[(index, rows[index]) for index in selected], manifest=manifest)


def write_subset_manifest(output_dir: Path, manifest: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "subset_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
