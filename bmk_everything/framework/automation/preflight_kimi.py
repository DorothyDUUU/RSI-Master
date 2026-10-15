#!/usr/bin/env python3
"""Credential-safe Kimi endpoint/model preflight; does not submit an evaluation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

FORMAT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORMAT_DIR))

from core.provider import kimi_endpoints, list_models


def main() -> None:
    wanted = "kimi-k2.7-code"
    results = []
    for endpoint in kimi_endpoints():
        models = list_models(endpoint)
        results.append({"alias": endpoint.alias, "base_url": endpoint.base_url,
                        "model_present": wanted in models, "model_count": len(models)})
    print(json.dumps({"model": wanted, "endpoints": results}, ensure_ascii=False, indent=2))
    if not all(item["model_present"] for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
