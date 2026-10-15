"""文件系统沙箱过滤器。

委托 tools/bash_security.check_paths —— 移植自 Claude Code 泄露源码
pathValidation.ts / filesystem.ts 的路径边界检查(重定向目标 / -- 结束选项 /
flag 值 / mv-cp flag 拒绝 / cd 后写路径 fail-closed / 危险删除路径 /
shell 展开字符 / 符号链接感知 realpath)。

python/python3 命令跳过路径检查(脚本内部文件访问无法静态分析)。
"""

from __future__ import annotations

from ..filter_base import BashFilter, FilterAction, FilterContext, FilterResult, register_filter

try:
    from tools.bash_security import ESCAPE_GUIDANCE, check_paths
    _SECURITY_AVAILABLE = True
except ImportError:  # repo root 不在 sys.path 时降级为放行
    _SECURITY_AVAILABLE = False
    ESCAPE_GUIDANCE = None


@register_filter
class FileSystemSandboxFilter(BashFilter):
    """文件系统沙箱过滤器。

    检查命令中引用的路径是否在 agent_workspace 或白名单目录内。
    拒绝访问沙箱外的文件系统。
    python/python3 命令跳过路径检查（脚本内部文件访问无法静态分析）。
    """
    name = "filesystem_sandbox"
    description = "限制文件系统访问范围，只允许访问 workspace 和白名单路径（python 命令跳过）"
    is_pre_filter = True

    def check(self, context: FilterContext, output: str = "") -> FilterResult:
        # 从 meta_data 获取白名单(服务端已用 DEFAULT_WHITELIST + --whitelist 覆盖)
        whitelist: list[str] = []
        if context.meta_data:
            whitelist = context.meta_data.get("whitelist", [])
        allowed_dirs = [context.resolved_workspace] + whitelist
        allowed_dirs = [d for d in allowed_dirs if d]

        if _SECURITY_AVAILABLE:
            r = check_paths(
                context.command,
                cwd=context.resolved_workspace or ".",
                allowed_dirs=allowed_dirs,
            )
            if not r.allowed:
                guidance = f"\n{ESCAPE_GUIDANCE}" if ESCAPE_GUIDANCE else ""
                return FilterResult(
                    action=FilterAction.REJECT,
                    reason=f"[{r.check_id}] {r.reason}{guidance}",
                    filter_name=self.name,
                )
        return FilterResult(action=FilterAction.ALLOW, filter_name=self.name)
