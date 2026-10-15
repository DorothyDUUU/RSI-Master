#!/usr/bin/env python3
"""search_cli CLI — one-file entry point.

Usage::

    python search_cli/main.py search "math word problems" --limit 20
    python search_cli/main.py inspect openai/gsm8k
    python search_cli/main.py materialize openai/gsm8k --config main --split train
    python search_cli/main.py web-search "LLM post-training"
    python search_cli/main.py github-repos "post-training"
    python search_cli/main.py scholar-arxiv "RLHF" --by content
    python search_cli/main.py health
    python search_cli/main.py server start --port 8991

All configuration is read from ``config.yaml`` in the same directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    _this_dir = Path(__file__).resolve().parent      # .../backend/backend/search_cli
    _package_dir = _this_dir.parent                  # .../backend/backend
    if str(_package_dir) not in sys.path:
        sys.path.insert(0, str(_package_dir))

    from search_cli.cli import main
    sys.exit(main())
