"""Skip this suite when the ``format/`` tree is not present.

Every module here loads a per-benchmark runner at import time
(``format/<bench>/run_benchmark.py``), so without that tree pytest fails at
collection rather than reporting a skip. ``format/`` is not vendored alongside
``framework/`` in this repo — add it (or point ``$BENCHMARK_ROOT`` at a full
checkout and run these tests from there) to exercise them.
"""

from __future__ import annotations

import pathlib

_FORMAT_DIR = pathlib.Path(__file__).resolve().parents[2] / "format"

if not _FORMAT_DIR.is_dir():
    collect_ignore = sorted(p.name for p in pathlib.Path(__file__).parent.glob("test_*.py"))
