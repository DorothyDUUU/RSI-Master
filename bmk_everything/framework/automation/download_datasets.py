#!/usr/bin/env python3
"""Download every generated benchmark's dataset to local JSONL and repoint spec.

Storage layout (one JSONL per split, one HF row dict per line):

    <data-dir>/<benchmark-id>/<split>.jsonl
    <data-dir>/_state.json                  # resumable progress

For each benchmark the spec.json ``dataset`` segment is rewritten in place so the
generic runner can serve rows from disk (local-first) while the original HF
coordinates are preserved under ``dataset.hf``:

    "dataset": {
      "path": "data/datasets/<benchmark-id>",
      "split": "<split>",
      "local_rows": <n>,
      "hf": {"path": ..., "name": ..., "revision": ..., "split": ...}
    }

The three hand-written legacy integrations (gsm8k/gpqa/aime2025) have no spec and
read their own pinned data; a symlink is created in the unified tree when the
upstream PostTrainBench directory exists.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Default to the HF origin. The sandbox sets HF_ENDPOINT=https://hf-mirror.com,
# whose absolute redirect to huggingface.co breaks huggingface_hub's metadata HEAD
# (commit_hash stays None). Callers that need the mirror for flaky repos can preset
# HF_ENDPOINT themselves (e.g. HF_ENDPOINT=https://hf-mirror.com); we only default
# here so a bare invocation reaches the origin.
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")

REPO_DIR = Path(__file__).resolve().parents[2]
# Optional pre-existing dataset tree to copy from instead of downloading.
LEGACY_UPSTREAM = Path(os.environ.get("POSTTRAINBENCH_DATASETS", "")) if os.environ.get("POSTTRAINBENCH_DATASETS") else None

BACKOFF_BASE = 10.0
MAX_ATTEMPTS = 6


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stream_hf_rows(spec: dict, hf_cache: Path):
    """Yield HF row dicts for the benchmark spec, with rate-limit backoff."""
    from datasets import load_dataset  # imported lazily; needs HF_ENDPOINT set above

    source = spec["dataset"]
    kwargs = {
        "path": source["path"],
        "split": source["split"],
        "streaming": True,
        "cache_dir": str(hf_cache),
    }
    if source.get("name"):
        kwargs["name"] = source["name"]
    if source.get("revision"):
        kwargs["revision"] = source["revision"]
    attempt = 0
    while True:
        try:
            dataset = load_dataset(**kwargs)
            for row in dataset:
                yield dict(row)
            return
        except Exception as exc:  # noqa: BLE001 - retryable network/HF errors
            attempt += 1
            if attempt >= MAX_ATTEMPTS:
                raise RuntimeError(f"load_dataset failed after {MAX_ATTEMPTS} attempts: {exc}") from exc
            delay = min(BACKOFF_BASE * (2 ** (attempt - 1)), 120.0)
            print(f"    [retry {attempt}/{MAX_ATTEMPTS}] {source['path']} "
                  f"name={source.get('name')} in {delay:.0f}s: {type(exc).__name__}: {exc}",
                  flush=True)
            time.sleep(delay)


def download_one(benchmark_id: str, spec: dict, data_dir: Path, hf_cache: Path,
                 state: dict, replace: bool) -> dict:
    entry = state.get(benchmark_id) or {}
    if entry.get("status") == "done" and not replace:
        return {"benchmark-id": benchmark_id, "status": "skipped",
                "rows": entry.get("rows"), "split": entry.get("split")}
    source = spec["dataset"]
    split = source["split"]
    target = data_dir / benchmark_id / f"{split}.jsonl"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.parent / f".tmp-{os.getpid()}-{int(time.time())}"
        rows = 0
        with tmp.open("w", encoding="utf-8") as out:
            for row in stream_hf_rows(spec, hf_cache):
                # default=str keeps image/bytes columns serializable (they become
                # placeholder repr strings); text columns are preserved verbatim.
                out.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                rows += 1
                if rows % 10_000 == 0:
                    print(f"  {benchmark_id}: {rows} rows...", flush=True)
        tmp.replace(target)
    except Exception as exc:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        state[benchmark_id] = {"status": "error", "split": split,
                               "error": f"{type(exc).__name__}: {exc}",
                               "updated": time.strftime("%Y%m%dT%H%M%S")}
        return {"benchmark-id": benchmark_id, "status": "error", "split": split,
                "error": str(exc)}
    if rows == 0:
        state[benchmark_id] = {"status": "error", "split": split,
                               "error": "dataset returned 0 rows",
                               "updated": time.strftime("%Y%m%dT%H%M%S")}
        return {"benchmark-id": benchmark_id, "status": "error", "split": split,
                "error": "dataset returned 0 rows"}
    # Repoint spec so the generic runner serves local rows.
    spec["dataset"] = {
        "path": f"data/datasets/{benchmark_id}",
        "split": split,
        "local_rows": rows,
        "hf": {
            "path": source["path"],
            "name": source.get("name"),
            "revision": source.get("revision"),
            "split": source.get("split"),
        },
    }
    write_json(REPO_DIR / "format" / benchmark_id / "spec.json", spec)
    state[benchmark_id] = {"status": "done", "split": split, "rows": rows,
                           "updated": time.strftime("%Y%m%dT%H%M%S")}
    return {"benchmark-id": benchmark_id, "status": "done", "split": split, "rows": rows}


def link_legacy(data_dir: Path) -> list[dict]:
    results = []
    if LEGACY_UPSTREAM is None:
        return [{"benchmark-id": benchmark_id, "status": "skip",
                 "error": "no upstream tree configured (set $POSTTRAINBENCH_DATASETS)"}
                for benchmark_id in ("gsm8k", "gpqa", "aime2025")]
    for benchmark_id in ("gsm8k", "gpqa", "aime2025"):
        upstream = LEGACY_UPSTREAM / benchmark_id
        link = data_dir / benchmark_id
        if not upstream.is_dir():
            results.append({"benchmark-id": benchmark_id, "status": "skip",
                            "error": f"upstream missing: {upstream}"})
            continue
        if link.exists():
            results.append({"benchmark-id": benchmark_id, "status": "exists"})
            continue
        link.symlink_to(upstream, target_is_directory=True)
        results.append({"benchmark-id": benchmark_id, "status": "linked"})
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format-dir", type=Path, default=REPO_DIR / "format")
    parser.add_argument("--data-dir", type=Path, default=REPO_DIR / "data/datasets")
    parser.add_argument("--hf-cache", type=Path, default=REPO_DIR / "data/cache")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--state", type=Path, default=REPO_DIR / "data/datasets/_state.json")
    parser.add_argument("--ids", help="Comma-separated benchmark ids; default: all generated")
    parser.add_argument("--replace", action="store_true",
                        help="Re-download and rewrite specs even if already done")
    args = parser.parse_args()
    format_dir = args.format_dir.resolve()
    data_dir = args.data_dir.resolve()
    hf_cache = args.hf_cache.resolve()
    hf_cache.mkdir(parents=True, exist_ok=True)

    markers = sorted(format_dir.glob("*/.automated-integration"))
    jobs = []
    for marker in markers:
        benchmark_id = marker.parent.name
        if args.ids and benchmark_id not in {i.strip() for i in args.ids.split(",")}:
            continue
        spec = read_json(marker.parent / "spec.json")
        jobs.append((benchmark_id, spec))
    state = json.loads(args.state.read_text()) if args.state.exists() else {}
    print(f"downloading {len(jobs)} generated datasets into {data_dir} "
          f"(workers={args.workers}, cache={hf_cache})", flush=True)

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, bid, spec, data_dir, hf_cache,
                               state, args.replace): bid for bid, spec in jobs}
        for future in as_completed(futures):
            bid = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                result = {"benchmark-id": bid, "status": "error", "error": repr(exc)}
            results.append(result)
            print(f"[{result['status'].upper()}] {bid}"
                  f"{' ' + str(result.get('rows')) + ' rows' if result.get('rows') else ''}", flush=True)
            write_json(args.state, state)

    legacy = link_legacy(data_dir)
    results.extend(legacy)
    for row in legacy:
        print(f"[{row['status'].upper()}] {row['benchmark-id']} (legacy symlink)", flush=True)
    results.sort(key=lambda item: item["benchmark-id"])
    done = sum(1 for r in results if r["status"] == "done")
    failed = [r for r in results if r["status"] == "error"]
    skipped = sum(1 for r in results if r["status"] == "skipped")
    print(json.dumps({"processed": len(results), "done": done, "skipped": skipped,
                      "failed": len(failed), "failures": failed,
                      "state": str(args.state)}, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
