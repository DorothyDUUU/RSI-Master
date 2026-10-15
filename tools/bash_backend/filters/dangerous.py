"""危险命令过滤器。

委托 tools/bash_security.check_command —— 移植自 Claude Code 泄露源码
bashSecurity.ts 的校验器电池(命令替换递归分析 / 混淆引号 / 反斜杠转义 /
Zsh 混淆 / eval 类内建 / 危险系统命令等)。相比旧的 ~9 条正则,现在能拦截
`rm -rf $(pwd)`、反引号命令替换、ANSI-C 引号混淆等已知逃逸。
"""

from __future__ import annotations

from ..filter_base import BashFilter, FilterAction, FilterContext, FilterResult, register_filter

try:
    from tools.bash_security import ESCAPE_GUIDANCE, check_command
    _SECURITY_AVAILABLE = True
except ImportError:  # repo root 不在 sys.path 时降级为放行(与历史行为一致)
    _SECURITY_AVAILABLE = False
    ESCAPE_GUIDANCE = None


@register_filter
class DangerousCommandFilter(BashFilter):
    """危险命令过滤器。

    拦截可能导致严重后果的命令：危险系统命令、命令替换/混淆、eval 类内建等。
    """
    name = "dangerous_command"
    description = "拦截危险的系统命令与混淆命令(命令替换/反引号/eval 类内建)"
    is_pre_filter = True

    def check(self, context: FilterContext, output: str = "") -> FilterResult:
        if _SECURITY_AVAILABLE:
            r = check_command(context.command)
            if not r.allowed:
                guidance = f"\n{ESCAPE_GUIDANCE}" if ESCAPE_GUIDANCE else ""
                return FilterResult(
                    action=FilterAction.REJECT,
                    reason=f"[{r.check_id}] {r.reason}{guidance}",
                    filter_name=self.name,
                )
        return FilterResult(action=FilterAction.ALLOW, filter_name=self.name)
