#!/usr/bin/env python3
"""GPU Backend CLI — one-file entry point.

Usage::

    python backend/gpu_backend/main.py exec -c "nvidia-smi"
    python backend/gpu_backend/main.py list-sessions
    python backend/gpu_backend/main.py submit -p ./train.sh --background
    python backend/gpu_backend/main.py info

All configuration is read from ``config.yaml`` in the same directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    _this_dir = Path(__file__).resolve().parent  # .../backend/gpu_backend
    _project_root = _this_dir.parent.parent       # .../ (<repo>)
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

    from backend.gpu_backend.cli import main
    sys.exit(main())
