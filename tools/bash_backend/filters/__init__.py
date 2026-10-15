"""内置过滤器子包。

导入此包会自动注册所有内置过滤器。
"""

# 导入所有内置过滤器以触发 @register_filter 装饰器
from .dangerous import DangerousCommandFilter
from .long_sleep import LongSleepFilter
from .network import NetworkAccessFilter
from .output_length import OutputLengthCheck
from .sandbox import FileSystemSandboxFilter

__all__ = [
    "FileSystemSandboxFilter",
    "DangerousCommandFilter",
    "NetworkAccessFilter",
    "OutputLengthCheck",
    "LongSleepFilter",
]
