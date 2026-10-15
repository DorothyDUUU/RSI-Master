"""共享 bash 安全拦截模块(纯 stdlib)。

由 bash_backend 过滤器与 evomaster BashTool 兜底路径共用。不依赖 fastapi /
pydantic —— 放在 tools/bash_security/ 而非 tools/bash_backend/ 下,正是为了
避免触发 tools/bash_backend/__init__.py(它 import fastapi/pydantic)。

用法:
    from tools.bash_security import check_command, check_paths, SecurityResult
    r = check_command("rm -rf /")
    r = check_paths("cat /etc/passwd", cwd="/workspace", allowed_dirs=["/workspace"])
"""

from __future__ import annotations

from .result import SecurityResult
from .interceptor import check_command
from .pathguard import check_paths, PATH_COMMANDS, SKIP_PATH_CHECK_COMMANDS
from .python_scan import check_python_code

# 拦截命中时追加给 LLM 的英文引导 —— 明确禁止尝试逃逸出允许目录。
# 所有拦截入口(后端过滤器 + evomaster 兜底)共用同一份文案。
ESCAPE_GUIDANCE = "Do not attempt to escape-run commands outside the allowed folders at any time."

__all__ = [
    "SecurityResult",
    "check_command",
    "check_paths",
    "check_python_code",
    "ESCAPE_GUIDANCE",
    "PATH_COMMANDS",
    "SKIP_PATH_CHECK_COMMANDS",
]
