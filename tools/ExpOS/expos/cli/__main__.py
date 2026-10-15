"""Enable ``python -m expos.cli`` invocation."""

from .main import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
