"""网络访问过滤器（默认禁用）。

可选：限制网络访问，只允许白名单域名。
默认不启用，需要手动开启。
"""

from __future__ import annotations

from ..filter_base import BashFilter, FilterAction, FilterContext, FilterResult, register_filter


@register_filter
class NetworkAccessFilter(BashFilter):
    """网络访问过滤器（默认禁用）。

    可选：限制网络访问，只允许白名单域名。
    默认不启用，需要手动开启。
    """
    name = "network_access"
    description = "限制网络访问（默认禁用，需手动开启）"
    is_pre_filter = True
    enabled = False  # 默认禁用

    def check(self, context: FilterContext, output: str = "") -> FilterResult:
        # 此过滤器默认禁用，启用后可扩展实现
        return FilterResult(action=FilterAction.ALLOW, filter_name=self.name)
