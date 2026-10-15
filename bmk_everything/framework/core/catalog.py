from __future__ import annotations

import json
from pathlib import Path


VALID_STATUSES = {"catalogued", "runnable", "verified"}


def read_manifest(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        manifest = json.load(source)
    benchmark_id = manifest.get("benchmark-id")
    if not benchmark_id:
        raise ValueError(f"missing benchmark-id in {path}")
    if path.parent.name != benchmark_id:
        raise ValueError(f"benchmark-id {benchmark_id!r} does not match directory {path.parent.name!r}")
    status = manifest.get("integration-status", "runnable")
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid integration-status {status!r} in {path}")
    manifest["integration-status"] = status
    manifest["_dir"] = path.parent
    return manifest


def discover(format_dir: Path) -> list[dict]:
    manifests = []
    for config_file in sorted(format_dir.glob("*/config.json")):
        if config_file.parent.name == "benchmark_example":
            continue
        manifests.append(read_manifest(config_file))
    return manifests


def find(format_dir: Path, benchmark_id: str) -> dict:
    path = format_dir / benchmark_id / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"benchmark not found: {benchmark_id}")
    return read_manifest(path)
