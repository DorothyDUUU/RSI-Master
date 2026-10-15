#!/usr/bin/env python3
"""Fill the BENCHMARK.md agent prompt template per benchmark.

Reads the shared template (benchmark_md_agent_prompt.txt), substitutes the
per-benchmark placeholders (context path, output path, eval-method, splits,
verify command), and writes one ready-to-use prompt file under
<repo>/data/md-prompts/<benchmark-id>.txt.  The authoring agent is told to read
that single file, which keeps every launch identical.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
TEMPLATE_PATH = Path(__file__).resolve().parent / "benchmark_md_agent_prompt.txt"
OUT_DIR = REPO_DIR / "data/md-prompts"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", help="Comma-separated ids; default: all generated")
    parser.add_argument("--format-dir", type=Path, default=REPO_DIR / "format")
    parser.add_argument("--context-dir", type=Path, default=REPO_DIR / "data/md-context")
    args = parser.parse_args()
    format_dir = args.format_dir.resolve()
    selected = {i.strip() for i in args.ids.split(",")} if args.ids else None
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    written = []
    for marker in sorted(format_dir.glob("*/.automated-integration")):
        benchmark_id = marker.parent.name
        if selected and benchmark_id not in selected:
            continue
        manifest = json.loads((marker.parent / "config.json").read_text(encoding="utf-8"))
        context_path = (args.context_dir.resolve() / f"{benchmark_id}.json")
        output_path = marker.parent / "BENCHMARK.md"
        verify_cmd = (f"{REPO_DIR}/framework/.venv/bin/python "
                      f"{REPO_DIR}/framework/automation/verify_benchmark_md.py "
                      f"--ids {benchmark_id}")
        prompt = (template
                  .replace("{context_path}", str(context_path))
                  .replace("{output_path}", str(output_path))
                  .replace("{eval_method}", str(manifest.get("eval-method") or "unscored"))
                  .replace("{splits}", ", ".join(str(s) for s in manifest.get("split") or ["default"]))
                  .replace("{verify_cmd}", verify_cmd))
        (OUT_DIR / f"{benchmark_id}.txt").write_text(prompt, encoding="utf-8")
        written.append(benchmark_id)
    print(json.dumps({"written": len(written), "out_dir": str(OUT_DIR)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
