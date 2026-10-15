"""``python -m backend.gpu_backend`` —— gpu-backend CLI 的模块入口。

与 console script ``gpu-backend`` (= ``backend.gpu_backend.cli:main``) 行为一致,
但用 ``python -m`` 跑, 不依赖 ``pip install -e .`` / PATH —— 见
``playground/rsi_master/agent_loop/playground.py::_seed_gpu_backend_launcher``
(任务 #80: agent workspace 里 seed 的 ``public/gpu-backend`` 就是包了一层这个)。

之所以不用 ``python -m backend.gpu_backend.cli``: ``__init__.py`` 会提前 import
``.cli``, 触发 runpy 的 ``found in sys.modules`` RuntimeWarning; ``__main__``
模块不会被 ``__init__`` 提前加载, 入口干净。
"""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
