"""异步 OpenAI 兼容 API 客户端 (迁移自 thirdparty/agent-code-base codebase/llm/llm_client.py)。

特性:
    - 自动限速 (令牌桶, 按每分钟请求数)
    - 智能重试 (指数退避, 超时/限速/5xx)
    - 请求计时 + token 用量等元数据
    - **kwargs 透传任意 OpenAI API 参数
    - 异步并发安全

使用方式与 config 配合: api_key / base_url 缺省时自动回退到环境变量
``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` (与仓库 ``llm.openai`` 约定一致)。
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from .utils.config import get_config
from .utils.logging import get_logger


@dataclass
class RetryConfig:
    """
    重试机制配置。

    Args:
        max_retries: 最大重试次数。
        retry_delay: 首次重试前的等待秒数。
        backoff_factor: 指数退避倍数 (每次重试 delay *= factor)。
        retry_on_timeout: 是否对超时重试。
        retry_on_rate_limit: 是否对限速重试。
        retry_on_server_error: 是否对 5xx 服务端错误重试。
    """

    max_retries: int = 3
    retry_delay: float = 1.0
    backoff_factor: float = 2.0
    retry_on_timeout: bool = True
    retry_on_rate_limit: bool = True
    retry_on_server_error: bool = True

    def should_retry(self, error: Exception, attempt: int) -> bool:
        """根据错误类型与已重试次数判断是否继续重试。"""
        if attempt >= self.max_retries:
            return False

        error_msg = str(error).lower()

        # 超时
        if self.retry_on_timeout and "timeout" in error_msg:
            return True

        # 限速
        if self.retry_on_rate_limit and "rate_limit" in error_msg:
            return True

        # 5xx 服务端错误
        if self.retry_on_server_error:
            if "5" in error_msg or "server error" in error_msg:
                return True

        return False

    def get_retry_delay(self, attempt: int) -> float:
        """计算第 attempt 次重试前的等待秒数 (指数退避)。"""
        return self.retry_delay * (self.backoff_factor ** attempt)


@dataclass
class ModelConfig:
    """模型/API 配置。"""

    model_name: str
    api_key: str
    base_url: str
    max_tokens: int = 5042
    temperature: float = 1.0
    rate_limit: int = 200
    retry_config: RetryConfig = field(default_factory=RetryConfig)


class RateLimiter:
    """令牌桶限速器, 控制每分钟最大请求数。"""

    def __init__(self, max_per_minute: int):
        self.interval = 60.0 / max_per_minute
        self.lock = asyncio.Lock()
        self.last = 0

    async def acquire(self):
        """获取一个令牌; 必要时等待以遵守限速。"""
        async with self.lock:
            now = time.monotonic()
            wait_time = self.interval - (now - self.last)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self.last = time.monotonic()


class LLMClient:
    """异步 OpenAI 兼容客户端: 限速 + 重试 + 计时 + 元数据。"""

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        config_path: Optional[str] = None,
    ):
        """config 缺省时从配置文件加载 (见 _load_config_from_file)。"""
        self.logger = get_logger(__name__)

        if config is None:
            config = self._load_config_from_file(config_path)

        self._validate_config(config)
        self.config = config

        self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        self.rate_limiter = RateLimiter(max_per_minute=config.rate_limit)
        self.logger.info(
            f"LLMClient initialized: model={config.model_name}, "
            f"base_url={config.base_url}, "
            f"max_retries={config.retry_config.max_retries}"
        )

    def _load_config_from_file(self, config_path: Optional[str]) -> ModelConfig:
        """
        从配置文件加载模型配置。

        读 ``qa_distill.model`` 节; api_key/base_url 缺省时回退到环境变量
        ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL``。
        """
        try:
            config = get_config(config_path)

            api_key = config.get("qa_distill.model.api_key")
            base_url = config.get("qa_distill.model.base_url")
            model_name = config.get("qa_distill.model.model", "gpt-4o-mini")
            max_tokens = config.get("qa_distill.model.max_tokens", 5042)
            temperature = config.get("qa_distill.model.temperature", 1.0)
            rate_limit = config.get("qa_distill.model.rate_limit", 200)

            # 环境变量兜底 (与仓库 llm.openai 约定一致)
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            base_url = base_url or os.getenv("OPENAI_BASE_URL")

            return ModelConfig(
                model_name=model_name,
                api_key=api_key,
                base_url=base_url,
                max_tokens=max_tokens,
                temperature=temperature,
                rate_limit=rate_limit,
            )

        except Exception as e:
            raise ValueError(
                f"Failed to load configuration: {e}. "
                "Please ensure config.yaml contains a valid 'qa_distill.model' section."
            )

    def _validate_config(self, config: ModelConfig):
        """校验必需字段。"""
        if not config.api_key:
            raise ValueError("API key is required")
        if not config.base_url:
            raise ValueError("Base URL is required")
        if not config.model_name:
            raise ValueError("Model name is required")

    async def chat_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        调用 chat completion API, 带重试与计时。

        Args:
            prompt: 用户提示词。
            system_prompt: 可选 system 提示词。
            **kwargs: 透传给 OpenAI API 的额外参数 (temperature / max_tokens /
                top_p / n / stream / stop / presence_penalty / frequency_penalty 等)。

        Returns:
            成功返回 (content, metadata); 重试耗尽返回 None。metadata 含:
            duration_ms / retry_count / timestamp / api_response / usage / model /
            finish_reason。
        """
        retry_config = self.config.retry_config
        last_error = None

        for attempt in range(retry_config.max_retries + 1):
            await self.rate_limiter.acquire()

            start_time = time.monotonic()
            timestamp = datetime.now().isoformat()

            try:
                messages: List[Dict[str, str]] = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                self.logger.debug(
                    f"Sending request to {self.config.model_name} "
                    f"(attempt {attempt + 1}/{retry_config.max_retries + 1}): "
                    f"prompt_length={len(prompt)}"
                )

                api_params = {
                    "model": kwargs.get("model", self.config.model_name),
                    "temperature": kwargs.get("temperature", self.config.temperature),
                    "messages": messages,
                    "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                }
                for key, value in kwargs.items():
                    if key not in ("model", "temperature", "max_tokens"):
                        api_params[key] = value

                response = await self.client.chat.completions.create(**api_params)

                content = response.choices[0].message.content
                api_response = response.model_dump()

                duration_ms = (time.monotonic() - start_time) * 1000

                metadata = {
                    "duration_ms": round(duration_ms, 2),
                    "retry_count": attempt,
                    "timestamp": timestamp,
                    "api_response": api_response,
                    "usage": {
                        "prompt_tokens": api_response.get("usage", {}).get("prompt_tokens", 0),
                        "completion_tokens": api_response.get("usage", {}).get("completion_tokens", 0),
                        "total_tokens": api_response.get("usage", {}).get("total_tokens", 0),
                    },
                    "model": api_response.get("model", self.config.model_name),
                    "finish_reason": api_response.get("choices", [{}])[0].get("finish_reason", "unknown"),
                }

                self.logger.info(
                    f"Request completed successfully: "
                    f"duration={duration_ms:.2f}ms, "
                    f"tokens={metadata['usage']['total_tokens']}, "
                    f"attempts={attempt + 1}"
                )

                return content, metadata

            except Exception as e:
                last_error = e
                duration_ms = (time.monotonic() - start_time) * 1000

                self.logger.warning(
                    f"API call failed (attempt {attempt + 1}): {type(e).__name__}: {str(e)}"
                )

                if retry_config.should_retry(e, attempt):
                    delay = retry_config.get_retry_delay(attempt)
                    self.logger.info(
                        f"Retrying in {delay:.2f}s... "
                        f"(attempt {attempt + 1}/{retry_config.max_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    break

        self.logger.error(
            f"API call failed after {retry_config.max_retries + 1} attempts: {last_error}"
        )
        return None

    async def safe_chat_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        timeout: int = 3600,
        **kwargs,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        带超时保护的 chat completion (避免 API 挂死拖住整个并发批次)。

        Returns:
            成功返回 (content, metadata); 失败或超时返回 None。
        """
        try:
            return await asyncio.wait_for(
                self.chat_completion(prompt, system_prompt, **kwargs),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            self.logger.warning(f"Request timeout after {timeout}s: {prompt[:50]}...")
            return None
        except Exception as e:
            self.logger.error(f"Safe API call failed: {e}")
            return None
