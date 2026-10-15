"""过滤器基类和注册系统测试。"""

from __future__ import annotations

import pytest

from tools.bash_backend.filter_base import (
    BashFilter,
    FilterAction,
    FilterContext,
    FilterResult,
    _FILTER_REGISTRY,
    get_all_filters,
    get_post_checks,
    get_pre_filters,
    register_filter,
)


class TestFilterRegistry:
    """过滤器注册表测试。"""

    def test_builtin_filters_registered(self):
        """内置过滤器应在导入时自动注册。"""
        # 导入 filters 包触发注册
        import tools.bash_backend.filters  # noqa: F401

        all_filters = get_all_filters()
        assert "filesystem_sandbox" in all_filters
        assert "dangerous_command" in all_filters
        assert "network_access" in all_filters
        assert "output_length" in all_filters

    def test_pre_filters_only_enabled(self):
        """get_pre_filters 只返回启用的 pre-filter。"""
        import tools.bash_backend.filters  # noqa: F401

        pre = get_pre_filters()
        for f in pre:
            assert f.is_pre_filter is True
            assert f.enabled is True

    def test_post_checks_only_enabled(self):
        """get_post_checks 只返回启用的 post-check。"""
        import tools.bash_backend.filters  # noqa: F401

        post = get_post_checks()
        for f in post:
            assert f.is_pre_filter is False
            assert f.enabled is True

    def test_register_custom_filter(self):
        """自定义过滤器可以被注册。"""
        @register_filter
        class TestFilter(BashFilter):
            name = "_test_custom_filter"
            description = "测试用过滤器"
            is_pre_filter = True

            def check(self, context, output=""):
                return FilterResult(action=FilterAction.ALLOW, filter_name=self.name)

        try:
            assert "_test_custom_filter" in get_all_filters()
        finally:
            # 清理
            _FILTER_REGISTRY.pop("_test_custom_filter", None)


class TestFilterContext:
    """过滤器上下文测试。"""

    def test_creation(self):
        ctx = FilterContext(
            command="echo hello",
            agent_workspace="/workspace",
            timeout=30.0,
        )
        assert ctx.command == "echo hello"
        assert ctx.agent_workspace == "/workspace"
        assert ctx.timeout == 30.0
        assert ctx.meta_data is None
        assert ctx.resolved_workspace == ""

    def test_with_meta_data(self):
        ctx = FilterContext(
            command="ls",
            agent_workspace="/workspace",
            timeout=30.0,
            meta_data={"whitelist": ["/tmp"]},
            resolved_workspace="/workspace",
        )
        assert ctx.meta_data["whitelist"] == ["/tmp"]


class TestFilterResult:
    """过滤器结果测试。"""

    def test_allow(self):
        r = FilterResult(action=FilterAction.ALLOW, filter_name="test")
        assert r.action == FilterAction.ALLOW
        assert r.reason == ""

    def test_reject(self):
        r = FilterResult(
            action=FilterAction.REJECT,
            reason="not allowed",
            filter_name="test",
        )
        assert r.action == FilterAction.REJECT
        assert r.reason == "not allowed"
