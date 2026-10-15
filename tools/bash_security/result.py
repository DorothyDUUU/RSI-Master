"""拦截结果类型。

本模块纯 stdlib,不依赖 fastapi / pydantic / bash_backend —— 供 bash_backend 过滤器
和 evomaster 的 BashTool 兜底路径共用。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SecurityResult:
    """一次命令/路径安全检查的结果。

    `allowed=False` 表示命令必须被拒绝执行。本框架无人在环(没有交互式批准),所以
    源码 Claude Code 里的 `ask`/`deny` 在这里统一收敛为 reject —— 消费端(过滤器、
    兜底 pre-filter)把 `allowed=False` 一律当作"不执行 + 返回过滤原因"。
    """

    allowed: bool
    reason: str
    check_id: str

    @classmethod
    def allow(cls, reason: str = "pass") -> "SecurityResult":
        return cls(True, reason, "allow")

    @classmethod
    def reject(cls, check_id: str, reason: str) -> "SecurityResult":
        return cls(False, reason, check_id)

    def __bool__(self) -> bool:
        return self.allowed
