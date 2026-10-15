"""``python -m expos`` —— 与 ``python -m expos.cli`` 完全相同的 CLI 入口。

agents 常直接写 ``python -m expos <verb>`` (expos 在 PYTHONPATH 时), 之前缺
这个 __main__ 会让 ``python -m expos`` 报 "No module named expos.__main__",
只能绕道 ``python -m expos.cli`` (任务 #82)。
"""

from __future__ import annotations

import sys

from expos.cli import main

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
