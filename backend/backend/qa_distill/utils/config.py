"""全局 YAML 配置管理 (迁移自 thirdparty/agent-code-base codebase/_utils/_config.py)。

提供 dot-notation / 属性访问的配置单例, 是 qa_distill 各模块共享配置的入口。

约定:
    - 未指定 config_path 时, 依次尝试环境变量 ``QA_DISTILL_CONFIG`` →
      ``<包根>/config/config.yaml``。
    - 单例一旦初始化, 后续 ``get_config(path)`` 传入不同路径会触发
      ``reload_from`` 重新加载, 避免跨模块拿到过期配置。
"""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Any, Optional

import yaml


#: 包根目录 (qa_distill/), 用于定位包内默认配置。
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
#: 默认配置文件路径: qa_distill/config/config.yaml
DEFAULT_CONFIG_PATH = _PACKAGE_ROOT / "config" / "config.yaml"


class Config:
    """
    全局配置管理器 (单例)。

    从 YAML 文件加载配置, 支持 dot-notation 与属性两种访问方式,
    在应用的各模块之间共享同一份配置。
    """

    _instance: Optional["Config"] = None
    _lock = Lock()
    _config_data: dict = {}
    _config_path: Optional[Path] = None

    def __new__(cls, config_path: Optional[str] = None):
        """返回全局唯一实例。"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path: Optional[str] = None):
        """从 YAML 文件加载配置。config_path 为空时使用默认路径。"""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            if config_path is None:
                config_path = os.environ.get("QA_DISTILL_CONFIG", DEFAULT_CONFIG_PATH)
            config_path = Path(config_path).expanduser()

            # 默认路径不存在时回退到模板 (config.example.yaml), 方便"只配 env 就跑";
            # 显式传入的 config_path 不存在则照常报错。
            if not config_path.exists() and config_path == DEFAULT_CONFIG_PATH:
                example_path = _PACKAGE_ROOT / "config" / "config.example.yaml"
                if example_path.exists():
                    config_path = example_path

            self._config_path = config_path
            self._load_config()
            self._initialized = True

    def _load_config(self) -> None:
        """读取 YAML 文件到内部 dict。"""
        if not self._config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self._config_path}"
            )

        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                self._config_data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise yaml.YAMLError(
                f"Failed to parse configuration file: {self._config_path}\nError: {e}"
            )

    def reload(self) -> None:
        """按当前路径重新加载配置。"""
        with self._lock:
            self._load_config()

    def reload_from(self, config_path: Optional[str]) -> "Config":
        """
        切换并重新加载配置; 若路径与当前一致则跳过, 避免无谓重读。

        返回 self 以便链式调用。
        """
        if not config_path:
            return self
        new_path = Path(config_path).expanduser().resolve()
        if new_path != self._config_path:
            with self._lock:
                self._config_path = new_path
                self._load_config()
        return self

    @property
    def path(self) -> Optional[Path]:
        """当前配置文件路径。"""
        return self._config_path

    def get(self, key: str, default: Any = None) -> Any:
        """按 dot-notation 取配置值 (如 'qa_distill.model.model')。"""
        keys = key.split(".")
        value = self._config_data

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_all(self) -> dict:
        """返回全部配置的浅拷贝。"""
        return self._config_data.copy()

    def __getattr__(self, name: str) -> Any:
        """属性访问顶层配置节 (返回 ConfigDict 以支持嵌套属性)。"""
        if name in self._config_data:
            value = self._config_data[name]
            return ConfigDict(value) if isinstance(value, dict) else value
        raise AttributeError(
            f"Configuration has no attribute '{name}'. "
            f"Available keys: {list(self._config_data.keys())}"
        )

    def __contains__(self, key: str) -> bool:
        """判断配置键是否存在 (dot-notation)。"""
        return self.get(key) is not None

    def __repr__(self) -> str:
        return (
            f"Config(path='{self._config_path}', "
            f"keys={list(self._config_data.keys())})"
        )


class ConfigDict:
    """嵌套配置节的 dict 式 + 属性式访问。"""

    def __init__(self, data: dict):
        self._data = data

    def get(self, key: str, default: Any = None) -> Any:
        value = self._data.get(key, default)
        return ConfigDict(value) if isinstance(value, dict) else value

    def __getattr__(self, name: str) -> Any:
        if name in self._data:
            value = self._data[name]
            return ConfigDict(value) if isinstance(value, dict) else value
        raise AttributeError(
            f"Configuration section has no attribute '{name}'. "
            f"Available keys: {list(self._data.keys())}"
        )

    def __getitem__(self, key: str) -> Any:
        value = self._data[key]
        return ConfigDict(value) if isinstance(value, dict) else value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"ConfigDict(keys={list(self._data.keys())})"

    def to_dict(self) -> dict:
        return self._data.copy()


# 全局配置单例
_global_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    获取全局配置实例 (主入口)。

    - 首次调用时创建单例; 若显式给出 config_path, 会保证该单例加载的是该路径。
    - 之后若传入不同路径, 会自动 reload 到新路径。

    Example:
        >>> from qa_distill.utils.config import get_config
        >>> config = get_config("config.yaml")
        >>> config.get("qa_distill.model.model")
        >>> config.qa_distill.output_data.experiment_name
    """
    global _global_config

    if _global_config is None:
        _global_config = Config(config_path)
    elif config_path is not None:
        _global_config.reload_from(config_path)

    return _global_config
