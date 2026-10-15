"""LLM 单价表 —— token 用量 → 金额成本 (traj_analysis README TODO T6).

统计侧 (utils_tools/traj_analysis) 用 ``price_per_1k`` 把每 agent 的
``prompt_tokens`` / ``completion_tokens`` 乘算成估算成本。这是**常量表**:
- key 是模型名 (config ``llm.openai.model``, 也是 llm_calls.jsonl 的 ``model``)。
- value 是 ``{"input": 每1k prompt token 美元, "output": 每1k completion token 美元}``。
- 未知模型返回 ``None`` → 统计侧标 ``N/A`` (不猜价)。

价格是**估算**用的 (2026-09 定格), 新网关/新模型出现时在此更新即可, 不必改分析代码。
config 里也可以用 ``llm.<name>.prices: {input_per_1k: ..., output_per_1k: ...}`` 覆盖
(见下文 ``resolve_price``), 但默认直接读本表。
"""

from __future__ import annotations

from typing import Any

# model → {input: $/1k prompt, output: $/1k completion} (美元)
LLM_PRICES: dict[str, dict[str, float]] = {
    # kimi-k3 (Moonshot Kimi K3) — 2026-09-01 录价
    "kimi-k3": {"input": 0.60, "output": 2.50},
    "kimi-k3-4k": {"input": 0.60, "output": 2.50},
    # deepseek-v4-flash — 低价档
    "deepseek-v4-flash": {"input": 0.27, "output": 1.10},
    # glm-4.5-flash / glm-53-flash (zhipu 系)
    "glm-53-flash": {"input": 0.10, "output": 0.40},
    "glm-4.5-flash": {"input": 0.10, "output": 0.40},
    # qwen 系 (llm-judge / 本地 vllm 通常免费或按实例)
    "qwen3-8b": {"input": 0.0, "output": 0.0},
    "qwen3.5-35b-a3b": {"input": 0.0, "output": 0.0},
    # fallback 低/未知 —— 显式 None 由调用方处理
}

# 宽松匹配: 去掉版本号后缀等 (model 串可能带 deployment name)
_NORMALIZED_KEYS: dict[str, str] = {}
for _model in LLM_PRICES:
    _NORMALIZED_KEYS[_model.lower().replace("_", "-").replace(":", "")] = _model


def price_per_1k(model: str | None) -> dict[str, float] | None:
    """返回 model 的 {input, output} 每1k token 美元价; 未知模型返回 None。

    宽松匹配: 忽略大小写 / 下划线 vs 连字符 / 版本后缀 (如 ``kimi-k3@v2``)。
    """
    if not model:
        return None
    exact = LLM_PRICES.get(model)
    if exact is not None:
        return exact
    key = model.lower().replace("_", "-").replace(":", "")
    # 去掉 @ 后的 deployment 后缀再试一次
    if "@" in key:
        key = key.split("@")[0]
    base = _NORMALIZED_KEYS.get(key)
    if base is not None:
        return LLM_PRICES[base]
    # 前缀匹配: kimi-k3-* -> kimi-k3 档
    for base_model, price in LLM_PRICES.items():
        if key.startswith(base_model.lower()):
            return price
    return None


def estimate_cost(
    model: str | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    price: dict[str, float] | None = None,
) -> float | None:
    """按单价表估算一次/一组调用的美元成本。

    ``price`` 可显式传入 (config 覆盖); 否则查常量表。任一 token 数或单价缺失
    返回 None (表示 N/A, 不计入汇总)。
    """
    if prompt_tokens is None or completion_tokens is None:
        return None
    p = price or price_per_1k(model)
    if not p:
        return None
    return (prompt_tokens / 1000.0) * p["input"] + (completion_tokens / 1000.0) * p["output"]


def resolve_price(config_llm: dict[str, Any] | None = None, model: str | None = None) -> dict[str, float] | None:
    """从 config 或常量表解析单价。

    ``config_llm`` = config 里某 provider 的块, 支持 ``prices: {input_per_1k, output_per_1k}``
    覆盖 (键名风格与 config 其他字段一致)。缺省回退常量表。
    """
    if config_llm and isinstance(config_llm.get("prices"), dict):
        prices = config_llm["prices"]
        if "input_per_1k" in prices and "output_per_1k" in prices:
            return {"input": float(prices["input_per_1k"]), "output": float(prices["output_per_1k"])}
    return price_per_1k(model)
