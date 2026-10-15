"""API 可用性预检工具 (迁移自 thirdparty/agent-code-base codebase/test/test_api.py)。

大批量蒸馏前先用它对目标模型/endpoint 做连通性检查 (含流式/非流式、定时心跳、
飞书告警), 避免跑到一半才发现 API 不可用。属于可选组件, 不影响蒸馏主链路。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, Union

from dotenv import load_dotenv
from openai import OpenAI

from .utils.config import get_config
from .utils.feishu import send_feishu_message
from .utils.logging import get_logger


class StreamType(Enum):
    """流式检查类型枚举。"""

    NON_STREAM_ONLY = "non_stream_only"
    STREAM_ONLY = "stream_only"
    BOTH = "both"

    @classmethod
    def from_value(cls, value):
        """字符串/枚举 → StreamType。支持别名: base/non_stream → 非流式,
        stream → 流式, all/both → 两者。"""
        if value is None:
            return cls.NON_STREAM_ONLY

        if isinstance(value, cls):
            return value

        if isinstance(value, str):
            value_lower = value.lower().strip()

            mapping = {
                "base": cls.NON_STREAM_ONLY,
                "non_stream": cls.NON_STREAM_ONLY,
                "non_stream_only": cls.NON_STREAM_ONLY,
                "stream": cls.STREAM_ONLY,
                "stream_only": cls.STREAM_ONLY,
                "all": cls.BOTH,
                "both": cls.BOTH,
            }

            if value_lower in mapping:
                return mapping[value_lower]

            raise ValueError(
                f"Invalid stream_type value: '{value}'. "
                f"Supported values: {list(mapping.keys())}"
            )

        raise ValueError(f"Expected string or StreamType, got {type(value).__name__}")


@dataclass
class APIConfig:
    """API endpoint 配置。"""

    api_key: str
    base_url: str

    def __post_init__(self):
        if not self.api_key:
            raise ValueError("API key cannot be empty")
        if not self.base_url:
            raise ValueError("Base URL cannot be empty")


class APITester:
    """API 可用性测试器: 单次/定时检查、流式测试、飞书告警。"""

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        api_configs: Optional[List[Tuple[str, str]]] = None,
        logger=None,
        config_path: Optional[str] = None,
    ):
        self.logger = logger or get_logger(__name__)
        self.api_configs = self._load_api_configs(
            api_key, base_url, api_configs, config_path
        )
        self._is_running = False

    def _load_api_configs(
        self,
        api_key: Optional[str],
        base_url: Optional[str],
        api_configs: Optional[List[Tuple[str, str]]],
        config_path: Optional[str] = None,
    ) -> List[APIConfig]:
        """
        加载 API 配置, 优先级:

        1. api_key + base_url 参数 (覆盖一切)
        2. api_configs 参数
        3. config 文件 (qa_distill.model 节)
        4. .env (OPENAI_API_KEY / OPENAI_BASE_URL)
        5. 环境变量
        """
        # 1. 显式 api_key + base_url
        if api_key and base_url:
            self.logger.info("Using provided api_key and base_url (overrides api_configs)")
            return [APIConfig(api_key, base_url)]

        # 2. api_configs
        if api_configs:
            self.logger.info(f"Using provided api_configs (count: {len(api_configs)})")
            return [APIConfig(key, url) for key, url in api_configs]

        # 3. config 文件
        try:
            config = get_config(config_path)
            cfg_api_key = config.get("qa_distill.model.api_key")
            cfg_base_url = config.get("qa_distill.model.base_url")
            if cfg_api_key and cfg_base_url:
                self.logger.info("Loaded configuration from config.yaml")
                return [APIConfig(cfg_api_key, cfg_base_url)]
        except Exception as e:
            self.logger.debug(f"Could not load from config file: {e}")

        # 4. .env
        env_path = Path.cwd() / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=True)
            self.logger.info(f"Loaded configuration from .env file: {env_path}")

        env_api_key = os.getenv("OPENAI_API_KEY")
        env_base_url = os.getenv("OPENAI_BASE_URL")

        if env_api_key and env_base_url:
            self.logger.info("Using configuration from environment variables")
            return [APIConfig(env_api_key, env_base_url)]

        raise ValueError(
            "No API configuration found. Please provide one of the following:\n"
            "1. Both api_key and base_url parameters\n"
            "2. api_configs parameter (list of tuples)\n"
            "3. Config file (qa_distill.model section)\n"
            "4. .env file with OPENAI_API_KEY and OPENAI_BASE_URL\n"
            "5. Environment variables OPENAI_API_KEY and OPENAI_BASE_URL"
        )

    def test_single_api(
        self,
        config: APIConfig,
        model: str,
        message: str,
        stream_type: StreamType = StreamType.NON_STREAM_ONLY,
    ) -> Tuple[bool, Optional[str]]:
        """测试单个 endpoint; 返回 (是否通过, 错误信息)。"""
        client = OpenAI(api_key=config.api_key, base_url=config.base_url)

        try:
            if stream_type in (StreamType.NON_STREAM_ONLY, StreamType.BOTH):
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": message}],
                    stream=False,
                )
                self.logger.info(
                    f"Non-stream test passed - Model: {model}, "
                    f"Base URL: {config.base_url}, "
                    f"Response: {response.choices[0].message.content}"
                )

            if stream_type in (StreamType.STREAM_ONLY, StreamType.BOTH):
                stream = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": message}],
                    stream=True,
                )
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        self.logger.debug(
                            f"Stream chunk: {chunk.choices[0].delta.content}"
                        )
                self.logger.info(
                    f"Stream test passed - Model: {model}, Base URL: {config.base_url}"
                )

            return True, None

        except Exception as e:
            error_msg = str(e)
            self.logger.error(
                f"API test failed - Model: {model}, "
                f"Base URL: {config.base_url}, Error: {error_msg}"
            )
            return False, error_msg

    def run_test(
        self,
        models: Optional[Union[List[str], str]] = None,
        messages: Optional[Union[List[str], str]] = None,
        stream_type: Union[StreamType, str] = StreamType.NON_STREAM_ONLY,
        enable_feishu_alert: bool = False,
    ) -> dict:
        """
        对全部 config × model 组合做连通性测试。

        Returns:
            {config_i: {model: {passed, api_key, base_url, error}}}。
        """
        stream_type = StreamType.from_value(stream_type)

        if models is None:
            self.logger.warning(
                "Models are not set, using default model 'gpt-3.5-turbo' instead."
            )
            models = ["gpt-3.5-turbo"]
        if isinstance(models, str):
            models = [models]

        if messages is None:
            messages = "Hello! Introduce who are you in English with approximately 50 words."
        if isinstance(messages, str):
            messages = [messages] * (len(self.api_configs) * len(models))

        self.logger.info(f"Starting API availability test with models: {models}")
        results = {}

        for config_idx, config in enumerate(self.api_configs):
            config_key = f"config_{config_idx + 1}"
            results[config_key] = {}

            for model_idx, model in enumerate(models):
                msg_idx = config_idx * len(models) + model_idx
                message = messages[msg_idx] if msg_idx < len(messages) else messages[0]

                self.logger.info(
                    f"Testing - Config: {config.base_url}, Model: {model}, "
                    f"Message: {message}"
                )

                passed, error_msg = self.test_single_api(config, model, message, stream_type)
                results[config_key][model] = {
                    "passed": passed,
                    "api_key": config.api_key[:10] + "...",
                    "base_url": config.base_url,
                    "error": error_msg,
                }

                if not passed and enable_feishu_alert:
                    alert_msg = (
                        f"API Test Failed\n"
                        f"Base URL: {config.base_url}\n"
                        f"Model: {model}\n"
                        f"Message: {message}\n"
                        f"Error: {error_msg}"
                    )
                    send_feishu_message(alert_msg)

                    if self._is_running:
                        self._is_running = False

        success_count = sum(
            1
            for config_results in results.values()
            for model_result in config_results.values()
            if model_result["passed"]
        )
        total_count = sum(len(config_results) for config_results in results.values())

        self.logger.info(f"Test completed: {success_count}/{total_count} tests passed")

        return results

    def start_periodic_test(
        self,
        interval_seconds: int = 60,
        models: Optional[Union[List[str], str]] = None,
        messages: Optional[Union[List[str], str]] = None,
        stream_type: Union[StreamType, str] = StreamType.NON_STREAM_ONLY,
        enable_feishu_alert: bool = False,
    ):
        """定时心跳测试 (直到 stop_periodic_test 或 Ctrl-C)。"""
        stream_type = StreamType.from_value(stream_type)
        if self._is_running:
            self.logger.warning("Periodic test is already running")
            return

        self._is_running = True
        self.logger.info(f"Starting periodic test with interval: {interval_seconds}s")

        try:
            while self._is_running:
                self.run_test(models, messages, stream_type, enable_feishu_alert)
                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            self.logger.info("Periodic test stopped by user")
            self._is_running = False

    def stop_periodic_test(self):
        """停止定时测试。"""
        self._is_running = False
        self.logger.info("Stopping periodic test...")
