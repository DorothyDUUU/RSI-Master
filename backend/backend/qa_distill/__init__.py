"""qa_distill — 大批量 QA 数据蒸馏工具包。

迁移自 ``thirdparty/agent-code-base`` 的核心蒸馏链路 (LLMClient + LLMPipeline)。

最小使用闭环 (配置 config && 写 prompt && 写解析器 → 批量并发生成):

    from qa_distill import LLMPipeline

    pipeline = LLMPipeline(config_path="config.yaml")
    data_pool = [{"input": {...}, "system_prompt_kwargs": {...}, "user_prompt_kwargs": {...}}, ...]

    def extract_qa(response: str):
        q = pipeline.extract_all("question", response)
        a = pipeline.extract_all("answer", response)
        return [{"question": x, "answer": y} for x, y in zip(q, a)]

    results = pipeline.run(data_pool, concurrency_limit=5, extract_function=extract_qa)
    # 结果实时追加写入 <output_dir>/<experiment_name>/result.jsonl

详细文档见本目录 README.md。
"""

from .apitest import APIConfig, APITester, StreamType
from .llm_client import LLMClient, ModelConfig, RetryConfig
from .pipeline import LLMPipeline
from .utils.config import Config, get_config
from .utils.feishu import FeishuBot, get_feishu_bot, send_feishu_message
from .utils.logging import get_logger

__version__ = "0.2.0"

__all__ = [
    "LLMPipeline",
    "LLMClient",
    "ModelConfig",
    "RetryConfig",
    "APITester",
    "StreamType",
    "APIConfig",
    "Config",
    "get_config",
    "get_logger",
    "FeishuBot",
    "get_feishu_bot",
    "send_feishu_message",
]
