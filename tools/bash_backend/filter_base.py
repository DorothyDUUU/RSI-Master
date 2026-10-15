"""过滤器基类和注册系统。

所有自定义过滤器应继承 BashFilter 并用 @register_filter 装饰。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger("bash_backend")


# ---------------------------------------------------------------------------
# 过滤器动作和数据结构
# ---------------------------------------------------------------------------


class FilterAction(str, Enum):
    """过滤器对命令的处理动作。"""
    ALLOW = "allow"
    REJECT = "reject"
    WARN = "warn"  # 允许执行但记录警告


@dataclass
class FilterContext:
    """过滤器执行上下文。"""
    command: str
    agent_workspace: str
    timeout: float
    meta_data: dict[str, Any] | None = None
    # 由 server 追加的运行时信息
    resolved_workspace: str = ""


@dataclass
class FilterResult:
    """过滤器检查结果。"""
    action: FilterAction
    reason: str = ""
    filter_name: str = ""


# ---------------------------------------------------------------------------
# 过滤器基类
# ---------------------------------------------------------------------------


class BashFilter(ABC):
    """过滤器基类。

    所有自定义过滤器应继承此类并实现 check 方法。
    过滤器分为 pre-filter（执行前）和 post-check（执行后）。
    """
    name: str = "base_filter"
    description: str = ""
    is_pre_filter: bool = True  # True = pre-filter, False = post-check
    enabled: bool = True

    @abstractmethod
    def check(self, context: FilterContext, output: str = "") -> FilterResult:
        """检查命令或输出。

        Args:
            context: 过滤上下文
            output: 仅 post-check 时有值，为命令的 stdout+stderr

        Returns:
            FilterResult 包含动作和原因
        """
        ...


# ---------------------------------------------------------------------------
# 全局过滤器注册表
# ---------------------------------------------------------------------------

_FILTER_REGISTRY: dict[str, BashFilter] = {}


def register_filter(filter_cls: type[BashFilter]) -> type[BashFilter]:
    """装饰器：注册一个过滤器类。

    用法：
        @register_filter
        class MyFilter(BashFilter):
            name = "my_filter"
            ...
    """
    instance = filter_cls()
    _FILTER_REGISTRY[instance.name] = instance
    logger.info(f"Registered filter: {instance.name} ({'pre' if instance.is_pre_filter else 'post'})")
    return filter_cls


def get_all_filters() -> dict[str, BashFilter]:
    """返回所有已注册的过滤器。"""
    return dict(_FILTER_REGISTRY)


def get_pre_filters() -> list[BashFilter]:
    """返回所有启用的 pre-filter。"""
    return [f for f in _FILTER_REGISTRY.values() if f.is_pre_filter and f.enabled]


def get_post_checks() -> list[BashFilter]:
    """返回所有启用的 post-check。"""
    return [f for f in _FILTER_REGISTRY.values() if not f.is_pre_filter and f.enabled]
