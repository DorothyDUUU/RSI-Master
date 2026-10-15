"""``python -m cc_rollout_expos`` — workspace/expos launcher 的模块入口."""

from .expos_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
