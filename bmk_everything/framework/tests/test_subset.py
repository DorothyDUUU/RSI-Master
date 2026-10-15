from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.subset import select_subset


def test_stable_hash_is_repeatable_and_not_head(monkeypatch):
    monkeypatch.delenv("SAMPLE_INDICES", raising=False)
    rows = [{"id": f"row-{index}"} for index in range(100)]
    first = select_subset(rows, limit=20, fingerprint="data-v1", stable_key=lambda row: row["id"])
    second = select_subset(rows, limit=20, fingerprint="data-v1", stable_key=lambda row: row["id"])
    assert first.manifest == second.manifest
    assert len(first.rows) == 20
    assert [index for index, _ in first.rows] != list(range(20))


def test_stratified_selection_covers_rare_groups(monkeypatch):
    monkeypatch.delenv("SAMPLE_INDICES", raising=False)
    rows = [
        {"id": f"major-{index}", "category": "major"} for index in range(98)
    ] + [
        {"id": "rare-a", "category": "rare-a"},
        {"id": "rare-b", "category": "rare-b"},
    ]
    result = select_subset(
        rows,
        limit=10,
        fingerprint="data-v1",
        stable_key=lambda row: row["id"],
        stratum=lambda row: row["category"],
    )
    assert result.manifest["stratum_counts"] == {"major": 8, "rare-a": 1, "rare-b": 1}


def test_explicit_indices_take_precedence(monkeypatch):
    monkeypatch.setenv("SAMPLE_INDICES", "2,7,9")
    rows = [{"id": str(index)} for index in range(10)]
    result = select_subset(rows, limit=2, fingerprint="data-v1", stable_key=lambda row: row["id"])
    assert [index for index, _ in result.rows] == [2, 7]
    assert result.manifest["algorithm"] == "explicit-indices"


def test_cross_process_reproducibility(monkeypatch):
    monkeypatch.delenv("SAMPLE_INDICES", raising=False)
    root = Path(__file__).resolve().parents[1]
    code = """
import json
from core.subset import select_subset
rows = [{\"id\": str(i), \"group\": str(i % 3)} for i in range(100)]
result = select_subset(rows, limit=17, fingerprint=\"same\", stable_key=lambda r: r[\"id\"], stratum=lambda r: r[\"group\"])
print(json.dumps(result.manifest, sort_keys=True))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root)
    one = subprocess.check_output([sys.executable, "-c", code], env=env, text=True)
    two = subprocess.check_output([sys.executable, "-c", code], env=env, text=True)
    assert json.loads(one) == json.loads(two)
