"""全局日志 (迁移自 thirdparty/agent-code-base codebase/_utils/_logging.py)。

提供 ``get_logger(name)`` 统一日志入口: 默认同时输出到 stdout (带时间戳/级别/
文件名行号)。如需落盘, 传 ``log_file=``。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


class GlobalLogger:
    """全局日志管理器 (单例), 按 name 缓存 logger 以避免重复加 handler。"""

    _instance: Optional["GlobalLogger"] = None
    _loggers: dict = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

    def setup_logger(
        self,
        name: str,
        level: int = logging.INFO,
        log_file: Optional[str] = None,
        format_string: Optional[str] = None,
        reset: bool = False,
    ) -> logging.Logger:
        """创建/获取并配置一个 logger。"""
        if name in self._loggers and not reset:
            return self._loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(level)

        if reset:
            logger.handlers.clear()
        elif logger.handlers:
            return logger

        if format_string is None:
            format_string = (
                "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s"
            )

        formatter = logging.Formatter(format_string)

        # 控制台输出
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 可选文件输出
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        self._loggers[name] = logger
        return logger

    def get_logger(self, name: str) -> logging.Logger:
        """获取已有 logger, 不存在则创建默认 logger。"""
        if name in self._loggers:
            return self._loggers[name]
        return self.setup_logger(name)


# 全局实例
_global_logger = GlobalLogger()


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    获取或创建一个 logger。

    Example:
        >>> from qa_distill.utils.logging import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("start")
    """
    return _global_logger.setup_logger(name, level, log_file, format_string)
