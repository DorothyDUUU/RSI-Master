"""Bash Backend Server — 安全的 bash 命令执行服务。

核心功能：
1. Pre-filtering: 可插拔的命令过滤链，限制文件系统访问等
2. Post-checking: 输出长度检查，防止超长输出
3. 进程监控: 跟踪运行中的进程，支持 kill 操作

自定义过滤器：
    继承 BashFilter 并用 @register_filter 装饰即可添加新规则。
    参见 filters/sandbox.py 的实现。
"""

from .app import create_app
from .command_parser import (
    extract_paths_from_segment,
    is_path_in_allowed_dirs,
    parse_command_segments,
    resolve_path,
)
from .constants import DEFAULT_PORT, DEFAULT_TIMEOUT, DEFAULT_WHITELIST, MAX_OUTPUT_LENGTH
from .executor import CommandExecutor
from .filter_base import (
    BashFilter,
    FilterAction,
    FilterContext,
    FilterResult,
    get_all_filters,
    get_post_checks,
    get_pre_filters,
    register_filter,
)
from .filters import *  # noqa: F401,F403 — 触发过滤器注册
from .models import (
    ExecuteRequest,
    ExecuteResponse,
    FilterInfoResponse,
    KillAllResponse,
    KillResponse,
    ProcessInfoResponse,
    SystemStatusResponse,
    ToggleFilterRequest,
)
from .monitor import SystemMonitor
from .tracker import ProcessInfo, ProcessStatus, ProcessTracker

__all__ = [
    # 过滤器系统
    "BashFilter",
    "FilterAction",
    "FilterContext",
    "FilterResult",
    "register_filter",
    "get_all_filters",
    "get_pre_filters",
    "get_post_checks",
    # 命令解析
    "parse_command_segments",
    "extract_paths_from_segment",
    "resolve_path",
    "is_path_in_allowed_dirs",
    # 执行器
    "CommandExecutor",
    # 进程跟踪
    "ProcessTracker",
    "ProcessInfo",
    "ProcessStatus",
    # 系统监控
    "SystemMonitor",
    # API 模型
    "ExecuteRequest",
    "ExecuteResponse",
    "ProcessInfoResponse",
    "KillResponse",
    "KillAllResponse",
    "FilterInfoResponse",
    "SystemStatusResponse",
    "ToggleFilterRequest",
    # 应用工厂
    "create_app",
    # 常量
    "DEFAULT_PORT",
    "DEFAULT_TIMEOUT",
    "DEFAULT_WHITELIST",
    "MAX_OUTPUT_LENGTH",
]
