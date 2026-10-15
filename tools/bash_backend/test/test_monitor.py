"""系统监控器测试。"""

from __future__ import annotations

import pytest

from tools.bash_backend.monitor import SystemMonitor


class TestSystemMonitor:
    """系统监控器测试。"""

    def test_get_cpu_usage(self):
        cpu = SystemMonitor.get_cpu_usage()
        # 应该返回一个合理的百分比值，或 -1.0（/proc 不可用时）
        assert cpu == -1.0 or (0.0 <= cpu <= 100.0)

    def test_get_memory_usage(self):
        mem = SystemMonitor.get_memory_usage()
        assert "total_kb" in mem
        assert "used_kb" in mem
        assert "available_kb" in mem
        assert "usage_percent" in mem

        # Linux 系统上应该有正的内存值
        if mem["total_kb"] > 0:
            assert mem["used_kb"] >= 0
            assert mem["available_kb"] >= 0
            assert 0.0 <= mem["usage_percent"] <= 100.0

    def test_memory_consistency(self):
        mem = SystemMonitor.get_memory_usage()
        if mem["total_kb"] > 0:
            # used + available 应该约等于 total（有少量 reserved memory）
            assert mem["used_kb"] + mem["available_kb"] <= mem["total_kb"] * 1.01
