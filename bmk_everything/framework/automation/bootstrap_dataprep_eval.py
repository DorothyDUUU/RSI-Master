#!/usr/bin/env python3
"""Download and split the revision-pinned DataPrep-Bench evaluation JSONL files."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path


REPO_ROOT = Path(os.environ.get("BENCHMARK_ROOT") or Path(__file__).resolve().parents[2]).expanduser().resolve()
DATASET_REVISION = "771688905d8c5c7e76acb83d6be615bb1027e066"
BASE_URL = f"https://huggingface.co/datasets/lhpku20010120/Data-Prep-Bench/resolve/{DATASET_REVISION}/eval"
FILES = {
    "business": "7d1d7e24e3ce02cbd284f1b28bd06aa9a4cf07db5e578df65e15ea2e661e6d3e",
    "law": "b5e8475d34f106605897293d40ff957524141dc0f37ab1a5c9a20279fddaf25e",
    "medicine": "e85cc244d25ee344185ca1c5df55be3bf2a0c32807888ad7e7312d33c92a4731",
}
TARGETS = {
    "legalbench": ("law", "legalbench", 413),
    "medcasereasoning": ("medicine", "medcasereasoning", 897),
    "ckqa": ("business", "fincdm", 311),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_source(domain: str, cache: Path) -> Path:
    path = cache / DATASET_REVISION / f"{domain}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and sha256(path) == FILES[domain]:
        return path
    temporary = path.with_suffix(".tmp")
    with urllib.request.urlopen(f"{BASE_URL}/{domain}.jsonl", timeout=120) as response, temporary.open("wb") as sink:
        while chunk := response.read(1024 * 1024):
            sink.write(chunk)
    if sha256(temporary) != FILES[domain]:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {domain}.jsonl")
    temporary.replace(path)
    return path


def materialize(benchmark_id: str, cache: Path, data_root: Path) -> dict:
    domain, source_key, expected = TARGETS[benchmark_id]
    source = ensure_source(domain, cache)
    target = data_root / benchmark_id / "test.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    rows = 0
    with source.open(encoding="utf-8") as stream, temporary.open("w", encoding="utf-8") as sink:
        for line in stream:
            record = json.loads(line)
            if record.get("metadata", {}).get("source") == source_key:
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                rows += 1
    if rows != expected:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"{benchmark_id}: expected {expected} rows, got {rows}")
    temporary.replace(target)
    return {"benchmark": benchmark_id, "rows": rows, "path": str(target), "sha256": sha256(target)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=[*TARGETS, "all"], default="all")
    parser.add_argument("--cache", type=Path, default=REPO_ROOT / "data/cache/dataprep-bench")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/datasets")
    args = parser.parse_args()
    selected = TARGETS if args.benchmark == "all" else [args.benchmark]
    for benchmark_id in selected:
        print(json.dumps(materialize(benchmark_id, args.cache, args.data_root), ensure_ascii=False))


if __name__ == "__main__":
    main()
