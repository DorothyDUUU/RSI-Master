#!/usr/bin/env python3
"""Bundle per-benchmark context for the BENCHMARK.md authoring agents.

Reads only local files (spec, manifest, config, downloaded JSONL rows, registry
row, bigbench official metrics) and writes one context JSON per benchmark into
<repo>/data/md-context/<benchmark-id>.json.  The authoring agent reads that
bundle and never invents facts beyond it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sample_rows(spec: dict, data_dir: Path, n: int = 3) -> list[dict]:
    source = spec.get("dataset") or {}
    if not str(source.get("path", "")).startswith("data/datasets"):
        return []
    jsonl_path = data_dir / spec["benchmark-id"] / f"{source.get('split', 'default')}.jsonl"
    if not jsonl_path.is_file():
        return []
    rows = []
    with jsonl_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) >= n:
                break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format-dir", type=Path, default=REPO_DIR / "format")
    parser.add_argument("--data-dir", type=Path, default=REPO_DIR / "data/datasets")
    parser.add_argument("--registry", type=Path,
                        default=REPO_DIR / "data/benchmark-survey/registry/selected_200.jsonl")
    parser.add_argument("--bigbench-metrics", type=Path,
                        default=REPO_DIR / "data/output/kimi27-answer-audit10-20260807/bigbench_upstream_metrics.json")
    parser.add_argument("--out-dir", type=Path, default=REPO_DIR / "data/md-context")
    parser.add_argument("--ids", help="Comma-separated benchmark ids; default: all generated")
    args = parser.parse_args()
    format_dir = args.format_dir.resolve()
    data_dir = args.data_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = {i.strip() for i in args.ids.split(",")} if args.ids else None

    # Registry ids are a re-verified scheme that does NOT equal the format-dir
    # benchmark-id, so build a multi-key index: exact id, legacy id, hf repo id.
    registry = {}
    registry_by_hf = {}
    if args.registry.is_file():
        for line in args.registry.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            registry[row.get("id")] = row
            legacy = row.get("legacy_benchmark_id")
            if legacy:
                registry[legacy] = row
            if row.get("hf_id"):
                registry_by_hf.setdefault(row["hf_id"], []).append(row)
    bigbench = {}
    if args.bigbench_metrics.is_file():
        for row in read_json(args.bigbench_metrics):
            bigbench[row.get("benchmark_id")] = row

    def lookup_registry(spec: dict, benchmark_id: str) -> dict:
        row = registry.get(benchmark_id)
        if row:
            return row
        hf = (spec.get("dataset") or {}).get("hf") or {}
        candidates = registry_by_hf.get(hf.get("path") or "", [])
        if not candidates:
            return {}
        # Prefer a row whose id/name hints at the same config, else the first.
        for candidate in candidates:
            hint = f"{candidate.get('id')} {candidate.get('name')}".lower()
            if (hf.get("name") or "").lower() in hint:
                return candidate
        return candidates[0]

    written = 0
    for marker in sorted(format_dir.glob("*/.automated-integration")):
        benchmark_id = marker.parent.name
        if selected and benchmark_id not in selected:
            continue
        spec = read_json(marker.parent / "spec.json")
        bundle = {
            "benchmark-id": benchmark_id,
            "spec": spec,
            "manifest": read_json(marker.parent / "config.json"),
            "config_yaml": (marker.parent / "config/config.yaml").read_text(encoding="utf-8"),
            "local_sample_rows": sample_rows(spec, data_dir),
            "registry": lookup_registry(spec, benchmark_id),
            "bigbench_official": bigbench.get(benchmark_id, {}),
        }
        (out_dir / f"{benchmark_id}.json").write_text(
            json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written += 1
    print(json.dumps({"written": written, "out_dir": str(out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
