"""系统资源监控器。

读取 /proc 获取 CPU 和内存使用情况。
"""

from __future__ import annotations

from typing import Any


class SystemMonitor:
    """系统资源监控器。"""

    @staticmethod
    def get_cpu_usage() -> float:
        """获取系统 CPU 使用率（百分比）。"""
        try:
            with open("/proc/stat", "r") as f:
                line = f.readline()
            parts = line.split()
            # user, nice, system, idle, iowait, irq, softirq, steal
            times = [int(x) for x in parts[1:9]]
            idle = times[3] + times[4]  # idle + iowait
            total = sum(times)
            return round((1 - idle / total) * 100, 2) if total > 0 else 0.0
        except (FileNotFoundError, IndexError, ValueError, ZeroDivisionError):
            return -1.0

    @staticmethod
    def get_memory_usage() -> dict[str, Any]:
        """获取系统内存使用情况。"""
        try:
            with open("/proc/meminfo", "r") as f:
                lines = f.readlines()
            info: dict[str, int] = {}
            for line in lines:
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = parts[1].strip().split()[0]
                    info[key] = int(val)  # kB

            total = info.get("MemTotal", 0)
            available = info.get("MemAvailable", 0)
            used = total - available
            return {
                "total_kb": total,
                "used_kb": used,
                "available_kb": available,
                "usage_percent": round(used / total * 100, 2) if total > 0 else 0.0,
            }
        except (FileNotFoundError, IndexError, ValueError):
            return {"total_kb": 0, "used_kb": 0, "available_kb": 0, "usage_percent": -1.0}
