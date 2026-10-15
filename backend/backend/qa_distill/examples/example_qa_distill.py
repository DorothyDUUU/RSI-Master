#!/usr/bin/env python3
"""QA 蒸馏最小可跑示例: 配置 config && 写 prompt && 写解析器 → 批量并发生成。

运行前:
    cd backend/backend/qa_distill
    cp config/config.example.yaml config/config.yaml   # 填入 api_key / base_url
    # 或者不填, 直接设置环境变量 / .env:
    #   export OPENAI_API_KEY=sk-xxx
    #   export OPENAI_BASE_URL=https://api.openai.com/v1

运行:
    python examples/example_qa_distill.py                # 全部条目
    python examples/example_qa_distill.py --resume       # 跳过已完成的条目 (断点续跑)
    python examples/example_qa_distill.py --dry-run      # 不调 API, 用假响应演示全流程

输出: config.output_data.output_dir 下的 result.jsonl (默认 ../output/...)。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 让 qa_distill 包可导入: 同时兼容
#   (a) 任意目录运行 → from qa_distill import ... (把 backend/backend 加进 path)
#   (b) 在仓库根运行 → from backend.backend.qa_distill import ...
_THIS_DIR = Path(__file__).resolve().parent
_PKG_DIR = _THIS_DIR.parent  # backend/backend/qa_distill
_BACKEND_ROOT = _PKG_DIR.parent  # backend/backend
_REPO_ROOT = _BACKEND_ROOT.parent.parent  # 仓库根
for _p in (str(_BACKEND_ROOT), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qa_distill import LLMPipeline, ModelConfig  # noqa: E402
from qa_distill.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

CONFIG_PATH = _PKG_DIR / "config" / ("config.yaml" if (_PKG_DIR / "config" / "config.yaml").exists() else "config.example.yaml")
DATA_POOL_SAMPLE = _PKG_DIR / "examples" / "data_pool_sample.jsonl"

# 内置兜底 data_pool (data_pool_sample.jsonl 不存在或 --inline 时使用)
_INLINE_DATA_POOL = [
    {
        "input": {"source": "线性代数"},
        "system_prompt_kwargs": {"language": "中文"},
        "user_prompt_kwargs": {"topic": "矩阵乘法的计算复杂度", "difficulty": "基础", "num": 1},
    },
    {
        "input": {"source": "机器学习"},
        "system_prompt_kwargs": {"language": "中文"},
        "user_prompt_kwargs": {"topic": "过拟合的成因与缓解手段", "difficulty": "中等", "num": 1},
    },
]


def load_data_pool(inline: bool = False) -> list[dict]:
    """加载 data_pool: 优先读 sample jsonl, 否则用内置列表。"""
    if not inline and DATA_POOL_SAMPLE.exists():
        pool = []
        with open(DATA_POOL_SAMPLE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    pool.append(json.loads(line))
        logger.info(f"Loaded {len(pool)} items from {DATA_POOL_SAMPLE}")
        return pool
    logger.info(f"Using inline data pool ({len(_INLINE_DATA_POOL)} items)")
    return _INLINE_DATA_POOL


def make_extract_qa(pipeline: LLMPipeline):
    """
    自定义输出解析器: 从模型输出里提取 <question> / <answer> 标签。

    返回 list[dict]: [{"question": ..., "answer": ...}, ...]; 解析失败返回 None
    (该条结果仍会落盘, 但 extracted=None, 便于事后过滤)。
    """
    def extract_qa(response: str):
        questions = pipeline.extract_all("question", response)
        answers = pipeline.extract_all("answer", response)
        if not questions or not answers:
            logger.warning("No <question>/<answer> found, extracted=None")
            return None
        return [
            {"question": q, "answer": a}
            for q, a in zip(questions, answers)
        ]

    return extract_qa


class _FakeClient:
    """dry-run 用假客户端: 返回固定格式响应, 演示整条流水线不依赖真实 API。"""

    def __init__(self, config=None, config_path=None):
        # run_all_tasks 会以 LLMClient(config=model_config) 构造; 这里直接忽略
        self.config = config

    async def safe_chat_completion(self, prompt, system_prompt=None, timeout=3600, **kwargs):
        return (
            "<question>占位问题?</question>\n\n<answer>占位答案。</answer>",
            {"duration_ms": 1.0, "retry_count": 0, "timestamp": "2026-01-01T00:00:00",
             "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
             "model": "fake", "finish_reason": "stop"},
        )


def main():
    parser = argparse.ArgumentParser(description="QA 蒸馏示例")
    parser.add_argument("--config", default=str(CONFIG_PATH), help="配置文件路径")
    parser.add_argument("--inline", action="store_true", help="用内置 data_pool, 不读 sample jsonl")
    parser.add_argument("--resume", action="store_true", help="跳过 result.jsonl 里已完成的条目")
    parser.add_argument("--dry-run", action="store_true", help="不调真实 API, 用假响应演示")
    parser.add_argument("--concurrency", type=int, default=3, help="并发数")
    args = parser.parse_args()

    pipeline = LLMPipeline(config_path=args.config)

    if args.dry_run:
        # 不联网: 把 pipeline 内部构造的 LLMClient 换成本地假实现 (只覆盖本次运行)。
        # 同时让 _get_model_config 返回占位配置 —— 假客户端不真发请求, 无需 api_key。
        import qa_distill.pipeline as _pipeline_mod

        _orig_client = _pipeline_mod.LLMClient
        _orig_get_cfg = _pipeline_mod.LLMPipeline._get_model_config
        _pipeline_mod.LLMClient = _FakeClient
        _pipeline_mod.LLMPipeline._get_model_config = lambda self: ModelConfig(
            model_name="fake",
            api_key="sk-dry-run",
            base_url="https://api.dry-run.local/v1",
        )
        try:
            run_pipeline(pipeline, args, load_data_pool(args.inline))
        finally:
            _pipeline_mod.LLMClient = _orig_client
            _pipeline_mod.LLMPipeline._get_model_config = _orig_get_cfg
    else:
        run_pipeline(pipeline, args, load_data_pool(args.inline))


def run_pipeline(pipeline: LLMPipeline, args, data_pool: list[dict]):
    """跑蒸馏: config → prompt → 解析器 → 并发生成。"""
    results = pipeline.run(
        data_pool=data_pool,
        concurrency_limit=args.concurrency,
        extract_function=make_extract_qa(pipeline),
        resume=args.resume,
    )

    ok = sum(1 for r in results if r.get("extracted") is not None)
    failed = sum(1 for r in results if r.get("error"))
    print(f"\n完成: 共 {len(results)} 条 | 解析成功 {ok} 条 | 失败 {failed} 条")
    print(f"结果文件: {pipeline.experiment_path}")

    # 打印前 2 条解析结果预览
    for r in results[:2]:
        if r.get("extracted"):
            for qa in r["extracted"]:
                print(f"  Q: {qa['question'][:60]}...")
                print(f"  A: {qa['answer'][:60]}...")


if __name__ == "__main__":
    main()
