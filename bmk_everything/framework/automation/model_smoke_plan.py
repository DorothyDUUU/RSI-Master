#!/usr/bin/env python3
"""Print the exact paid smoke-test plan. This script never calls a model."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def main() -> None:
    format_dir = Path(__file__).resolve().parents[2] / "format"
    ids = [item["benchmark-id"] for item in json.loads((format_dir / "benchmark.json").read_text())]
    adapters = Counter()
    for benchmark_id in ids:
        manifest = json.loads((format_dir / benchmark_id / "config.json").read_text())
        adapters[manifest.get("adapter", "legacy_runner")] += 1
    base_calls = len(ids) * 3
    family_calls = len(adapters) * 20
    calls = base_calls + family_calls
    # Budget estimate, not metering: 1.5k input and 512 output tokens per call.
    assumed_input = calls * 1500
    assumed_output = calls * 512
    estimated_usd = assumed_input / 1_000_000 * 0.95 + assumed_output / 1_000_000 * 4.00
    worst_input = calls * 3000
    worst_output = calls * 2048
    worst_usd = worst_input / 1_000_000 * 0.95 + worst_output / 1_000_000 * 4.00
    print(json.dumps({
        "model": "kimi-k2.7-code",
        "benchmarks": len(ids),
        "adapter_families": adapters,
        "smoke_3_calls": base_calls,
        "representative_20_per_adapter_calls": family_calls,
        "total_calls": calls,
        "estimate": {
            "assumption": {"input_tokens_per_call": 1500, "output_tokens_per_call": 512},
            "tokens": {"input": assumed_input, "output": assumed_output},
            "usd_at_0.95_input_4.00_output_per_million": round(estimated_usd, 2),
        },
        "conservative_upper_bound": {
            "assumption": {"input_tokens_per_call": 3000, "output_tokens_per_call": 2048},
            "tokens": {"input": worst_input, "output": worst_output},
            "usd_at_0.95_input_4.00_output_per_million": round(worst_usd, 2),
        },
        "note": "Ksyun fallback traffic is excluded because its price was not independently verified.",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
