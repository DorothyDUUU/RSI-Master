"""长时间 sleep 命令过滤器。

拦截 `sleep <N>` 中 N 超过上限(默认 40min=2400s, 服务端启动可用
`--max-sleep-seconds` 覆盖)的命令。动机: agent 用超长阻塞式 sleep 轮询
训练/评测/蒸馏进度, 白白占用 agent 时间 —— 2026-08-27 对 4 个 0826 run 实测
68 次 sleep>=1800s、最长 7200s=2h、合计阻塞 45.4h。

策略: 按 shell 分隔符(&& / || / ; / |)把命令拆段(引号安全), 若某段以
`sleep` 命令开头, 把该段全部时间操作数求和(支持小数与 s/m/h/d 后缀, 同 GNU
sleep)。超过上限 → REJECT。等长(=上限)放行, 严格「超过」才拦。

已知局限: 只识别段首为 sleep 的命令(覆盖 0826 全部实测 pattern); `for ...;
do sleep N; done` 这类把 sleep 放在段中间的写法不会命中。
"""

from __future__ import annotations

import re

from ..command_parser import parse_command_segments
from ..constants import DEFAULT_MAX_SLEEP_SECONDS
from ..filter_base import BashFilter, FilterAction, FilterContext, FilterResult, register_filter

# 单个 sleep 时间操作数: 数字 + 可选单位(s/m/h/d), 支持小数 (GNU sleep)
_SLEEP_OPERAND = re.compile(r"^\d+(?:\.\d+)?[smhdSMHD]?$")
_SLEEP_OPERAND_NUM = re.compile(r"^(\d+(?:\.\d+)?)([smhdSMHD]?)$")

_UNIT_SECONDS = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}

# sleep 二进制名(含常见绝对路径; 不带路径前缀时 sh 也从 PATH 找)
_SLEEP_BINS = {"sleep", "/usr/bin/sleep", "/bin/sleep"}


def _operand_seconds(token: str) -> float | None:
    """把单个 sleep 时间操作数转成秒数; 非操作数返回 None。

    容忍引号包裹(如 `sleep "45m"`), 以便 parse_command_segments 还原后的段
    也能解析。
    """
    tok = token.strip().strip('"').strip("'")
    if not _SLEEP_OPERAND.match(tok):
        return None
    m = _SLEEP_OPERAND_NUM.match(tok)
    num = float(m.group(1))
    unit = m.group(2).lower()
    return num * _UNIT_SECONDS.get(unit, 1.0)


def _sleep_total_seconds(segment: str) -> float | None:
    """若段以 sleep 命令开头, 返回该段全部时间操作数的总秒数; 否则 None。

    遇到第一个非操作数 token 就停(超过操作数即视为非 sleep 用法)。
    """
    tokens = segment.split()
    if not tokens or tokens[0] not in _SLEEP_BINS:
        return None
    total = 0.0
    for tok in tokens[1:]:
        sec = _operand_seconds(tok)
        if sec is None:
            break
        total += sec
    return total if total > 0 else None


def _command_sleep_seconds(command: str) -> float | None:
    """扫描整条命令, 返回出现的 sleep 时长最大值(秒); 无 sleep 返回 None。

    引号内文本(echo/注释里的 "sleep")不会误命中。
    """
    best = 0.0
    for seg in parse_command_segments(command):
        sec = _sleep_total_seconds(seg)
        if sec is not None and sec > best:
            best = sec
    return best if best > 0 else None


@register_filter
class LongSleepFilter(BashFilter):
    """长时间 sleep 过滤器。

    拦截 `sleep <N>` 超过上限的命令, 引导 agent 改用短轮询/低频查询。
    """
    name = "long_sleep"
    description = "拦截 sleep 超过上限(默认 40 分钟)的命令, 防超长阻塞式轮询"
    is_pre_filter = True

    def check(self, context: FilterContext, output: str = "") -> FilterResult:
        if not context.command:
            return FilterResult(action=FilterAction.ALLOW, filter_name=self.name)

        max_sec = DEFAULT_MAX_SLEEP_SECONDS
        if context.meta_data:
            max_sec = int(context.meta_data.get("max_sleep_seconds", DEFAULT_MAX_SLEEP_SECONDS))

        dur = _command_sleep_seconds(context.command)
        if dur is not None and dur > max_sec:
            return FilterResult(
                action=FilterAction.REJECT,
                reason=(
                    f"命令包含 sleep {dur:g}s, 超过上限 {max_sec}s({max_sec // 60}min)。"
                    f"超长阻塞式 sleep 会长时间占用 agent 轮次 —— 请改成短轮询"
                    f"(sleep ≤ {max_sec}s 多次轮询), 或用 gpu-backend "
                    f"script-status/script-log 做低频查询。"
                ),
                filter_name=self.name,
            )
        return FilterResult(action=FilterAction.ALLOW, filter_name=self.name)
