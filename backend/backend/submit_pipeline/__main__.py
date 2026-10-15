"""``python -m backend.backend.submit_pipeline <submit|status|list|health>``."""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
