"""输出长度检查。

如果命令输出超过阈值，拒绝返回完整输出，提示使用 head/tail/wc。
"""

from __future__ import annotations

from ..constants import MAX_OUTPUT_LENGTH
from ..filter_base import BashFilter, FilterAction, FilterContext, FilterResult, register_filter


@register_filter
class OutputLengthCheck(BashFilter):
    """输出长度检查。

    如果命令输出超过阈值，拒绝返回完整输出，提示使用 head/tail/wc。
    """
    name = "output_length"
    description = "检查输出长度，超长时拒绝返回并提示截断"
    is_pre_filter = False
    max_length: int = MAX_OUTPUT_LENGTH

    def check(self, context: FilterContext, output: str = "") -> FilterResult:
        if len(output) > self.max_length:
            return FilterResult(
                action=FilterAction.REJECT,
                reason=(
                    f"输出长度 {len(output)} 字符超过限制 {self.max_length}。"
                    f"请使用 head/tail/wc 等命令限制输出，例如: "
                    f"your_command | head -n 100"
                ),
                filter_name=self.name,
            )
        return FilterResult(action=FilterAction.ALLOW, filter_name=self.name)
