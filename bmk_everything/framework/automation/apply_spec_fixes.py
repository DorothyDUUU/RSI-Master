#!/usr/bin/env python3
"""Materialize reviewed spec fixes into generated benchmark directories."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FRAMEWORK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRAMEWORK_DIR))
from automation.spec_fixes_data import apply_spec_fixes


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    changes = []
    for spec_path in sorted(args.format_dir.glob("*/spec.json")):
        spec = json.loads(spec_path.read_text())
        fixed = apply_spec_fixes(spec)
        if fixed == spec:
            continue
        changes.append(str(spec_path))
        if not args.check:
            write_json(spec_path, fixed)
            manifest_path = spec_path.parent / "config.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["eval-method"] = fixed["scorer"]
            manifest["scoring-status"] = (
                "unsupported" if fixed["scorer"] == "unscored" else "supported"
            )
            if fixed.get("required-metric"):
                manifest["required-metric"] = fixed["required-metric"]
            write_json(manifest_path, manifest)
    config_changes = []
    for marker in sorted(args.format_dir.glob("*/.automated-integration")):
        config_path = marker.parent / "config/config.yaml"
        if not config_path.exists():
            continue
        text = config_path.read_text()
        fixed_text = text.replace("max_tokens: 2048", "max_tokens: 4096")
        if fixed_text != text:
            config_changes.append(str(config_path))
            if not args.check:
                config_path.write_text(fixed_text)
    if args.registry and args.registry.exists():
        rows = [apply_spec_fixes(json.loads(line))
                for line in args.registry.read_text().split("\n") if line]
        if not args.check:
            args.registry.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
    print(json.dumps({"changed_specs": len(changes), "changed_configs": len(config_changes),
                      "check": args.check, "files": changes + config_changes},
                     ensure_ascii=False, indent=2))
    if args.check and (changes or config_changes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
