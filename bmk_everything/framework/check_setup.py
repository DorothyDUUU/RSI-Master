"""Read-only benchmark asset check; uses only the Python standard library."""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=os.environ.get('BENCHMARK_ROOT') or str(Path(__file__).resolve().parent.parent))
    parser.add_argument('--bench-id', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.bench_id):
        parser.error('invalid benchmark id')
    root = Path(args.root).expanduser().resolve()
    bench = root / 'format' / args.bench_id
    required = [root / 'framework/run_eval.py', root / 'framework/.venv/bin/python',
                bench / 'config.json', bench / 'run_benchmark.py', bench / 'BENCHMARK.md',
                root / 'data/datasets']
    missing = [str(p) for p in required if not p.exists()]
    errors = []
    manifest_path = bench / 'config.json'
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text())
            if not isinstance(manifest, dict) or manifest.get('benchmark-id') != args.bench_id:
                errors.append('manifest benchmark-id does not match directory')
            elif manifest.get('integration-status', 'runnable') not in {'runnable', 'verified'}:
                errors.append('benchmark is not marked runnable or verified')
        except (OSError, ValueError) as exc:
            errors.append(f'invalid benchmark manifest: {exc}')
    print(json.dumps({'root': str(root), 'bench_id': args.bench_id,
                      'assets_present': not missing and not errors,
                      'missing': missing, 'errors': errors,
                      'next': 'Validate runner-specific data and dependencies, then run a real evaluation. See bmk_everything/SETUP.md.'}, indent=2))
    return int(bool(missing or errors))


if __name__ == '__main__':
    raise SystemExit(main())
