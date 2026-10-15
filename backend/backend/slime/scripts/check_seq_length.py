#!/usr/bin/env python3
"""Check token-length distribution of training data and recommend seq_length / CP.

Usage (standalone, on any node with the data + tokenizer):

    python check_seq_length.py \\
        --data-file /path/to/train.parquet \\
        --model-dir /path/to/Qwen3-8B-Base

    # Output: human report + YAML snippet for slime_config.

Purpose (pre-check before submit):
    slime/Megatron-LM 训练时, ``seq_length`` 决定每条样本的最大 token 数。超长样本
    被截断 -> 答案尾部丢失 -> 精确答案 (GSM8K/AIME) 质量直降。本脚本统计真实 token
    长度分布, 推荐安全的 seq_length 和 (若需要) CP 并行度。

    Agent 应在每次 ``worker_submit`` 前先跑本脚本, 把推荐值写入 slime_config YAML。

Design:
    - 读 parquet 的 ``messages`` 列 (OpenAI chat 格式, [{role, content}, ...]).
    - 用 Qwen3 tokenizer 的 chat_template 渲染 + tokenize (与训练侧一致).
    - 报告: len, max, mean, p50, p90, p95, p99, p999.
    - 推荐 seq_length = ceil(p99 / 128) * 128 (128 对齐, 过 4096 后 CP 建议).
    - 若推荐 seq_length > 4096, 给出 CP 建议。

CP 决策逻辑:
    Megatron 契约: world_size = TP × EP × CP = NUM_GPUS (PP=1).
    seq_length <= 4096 时, CP=1 足够 (激活内存不大)。
    seq_length > 4096 时, 推荐 CP = ceil(seq_length / 4096), 但 CP 必须整除
    NUM_GPUS / (TP × EP)。脚本不知道实际 TP/EP/NUM_GPUS, 所以只输出推荐值
    + 警告 "CP 会占用并行度预算, 请确认 TP×EP×CP = NUM_GPUS"。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


#: 128 对齐推荐值 (Megatron 的 seq_length 通常对齐到 128 / 256 的整数倍).
ROUND_BASE = 128

#: 默认 seq_length: 4B/8B 档位 = 32768 (Qwen3 32K 上下文); render 后备默认按
#: 尺寸给 (小模型 32768 / 大模型 2048)。
DEFAULT_SEQ_LENGTH = 32768

#: CP 推荐的阈值: 超过此值建议开 CP.
CP_THRESHOLD = 4096


def _round_up(x: int, base: int = ROUND_BASE) -> int:
    """向上取整到 base 的整数倍."""
    return ((x + base - 1) // base) * base


def _load_data(path: str) -> list[dict[str, Any]]:
    """加载 parquet 的 messages 列. 处理 numpy 类型."""
    import pandas as pd  # slow import

    df = pd.read_parquet(path)
    col = "messages"
    if col not in df.columns:
        raise ValueError(
            f"Parquet 缺少 '{col}' 列. 列: {list(df.columns)}"
        )
    msgs = df[col].tolist()

    # 转换 numpy 类型为原生 Python
    def _to_native(o):
        import numpy as np  # slow import

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, dict):
            return {k: _to_native(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_to_native(x) for x in o]
        return o

    return [_to_native(m) for m in msgs]


def _load_tokenizer(model_dir: str):
    """加载 Qwen3 tokenizer."""
    from transformers import AutoTokenizer  # slow import

    return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)


def _tokenize_lengths(
    messages: list[dict[str, Any]], model_dir: str
) -> tuple[list[int], Any]:
    """Tokenize 每条消息, 返回 (长度列表, tokenizer)."""
    tok = _load_tokenizer(model_dir)
    lengths = []
    for m in messages:
        rendered = tok.apply_chat_template(m, tokenize=False)
        ids = tok(rendered, add_special_tokens=False)["input_ids"]
        lengths.append(len(ids))
    return lengths, tok


def _compute_stats(lengths: list[int]) -> dict[str, float]:
    """计算统计量."""
    s = sorted(lengths)
    n = len(s)
    return {
        "n": n,
        "max": max(s),
        "mean": statistics.mean(s),
        "p50": s[int(n * 0.50)],
        "p90": s[int(n * 0.90)],
        "p95": s[int(n * 0.95)],
        "p99": s[int(n * 0.99)],
        "p999": s[int(n * 0.999)],
        "p100": max(s),
    }


def _recommend_seq_length(stats: dict) -> int:
    """推荐 seq_length: 向上取整到 128 的倍数, 至少 p99.

    策略:
    - 如果 p99 <= 4096: 推荐 = ceil(p99 / 128) * 128.
    - 如果 p99 > 4096: 推荐 = ceil(p99 / 128) * 128, 推荐 CP.
    """
    raw = stats["p99"]
    recommended = _round_up(raw)
    # 至少 128
    if recommended < 128:
        recommended = 128
    return recommended


def _recommend_cp(seq_length: int) -> int:
    """推荐 CP: seq_length > 4096 时 CP² (向上取整, 但不小于 2)."""
    if seq_length <= CP_THRESHOLD:
        return 1
    # 建议 CP = ceil(seq_length / 4096), 但至少 2
    cp = max(2, _round_up(seq_length, 4096) // 4096)
    return cp


def _print_report(
    stats: dict,
    recommended_seq: int,
    recommended_cp: int,
    model_dir: str,
    data_file: str,
) -> None:
    """打印人类可读报告."""
    print(f"=== seq_length checker: {data_file} ===")
    print(f"  tokenizer: {model_dir}")
    print(f"  rows:      {stats['n']}")
    print(f"  max:       {stats['max']}")
    print(f"  mean:      {stats['mean']:.1f}")
    print(f"  p50:       {stats['p50']}")
    print(f"  p90:       {stats['p90']}")
    print(f"  p95:       {stats['p95']}")
    print(f"  p99:       {stats['p99']}")
    print(f"  p999:      {stats['p999']}")
    print(f"  p100:      {stats['p100']}")
    print()
    print(f"  training_mode: slime_sft")
    print(f"  recommended seq_length: {recommended_seq}")
    if recommended_seq > CP_THRESHOLD:
        print(f"  ⚠️  recommended seq_length > {CP_THRESHOLD}, "
              f"consider CP (context parallelism).")
        print(f"  recommended CP_SIZE: {recommended_cp}")
        print(f"  ⚠️  CP 占用并行度预算: 请确保 TP×EP×CP = NUM_GPUS。")
        print(f"      例如: 当前 TP=2 EP=2 NUM_GPUS=4, 开 CP={recommended_cp} "
              f"需要 TP×EP = {4 // recommended_cp}, 即调整 EP 或 TP。")
    else:
        print(f"  CP_SIZE: 1 (seq_length <= {CP_THRESHOLD}, CP 无收益)")
    print()

    # 截断检查
    if recommended_seq < stats["max"]:
        truncated = sum(1 for x in [stats["p100"]] if x > recommended_seq)
        print(f"  ⚠️  WARNING: 推荐 seq_length={recommended_seq} < "
              f"单条最长样本 {stats['max']} token!")
        print(f"      最长样本会被截断 {stats['max'] - recommended_seq} token。")
        print(f"      若需无截断, 请将 seq_length 设为至少 {stats['max']}。")
        print()
    else:
        print(f"  ✅ 推荐 seq_length={recommended_seq} >= 样本最大长度 "
              f"{stats['max']}, 零截断。")
        print()


def _print_yaml_snippet(
    recommended_seq: int,
    recommended_cp: int,
    stats: dict,
) -> None:
    """打印 YAML 配置片段."""
    print(f"# === YAML snippet for slime_config ===")
    print(f"# (paste into your slime config YAML)")
    print(f"#")
    print(f"# 数据: {stats['n']} 条, "
          f"max={stats['max']}, p99={stats['p99']}")
    print(f"# 推荐: seq_length={recommended_seq}, "
          f"cp_size={recommended_cp}")
    print(f"#")
    print(f"training_mode: slime_sft")
    print(f"slime_config:")
    print(f"  seq_length: {recommended_seq}")
    print(f"  max_tokens_per_gpu: {recommended_seq}")
    if recommended_cp > 1:
        print(f"  cp_size: {recommended_cp}")
        print(f"  # ⚠️  开 CP={recommended_cp} 后, 必须调整 TP/EP 使 "
              f"TP×EP×CP = NUM_GPUS")
    else:
        print(f"  # cp_size: 1 (default, CP not needed)")
    print(f"# =================================")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Check token-length distribution and recommend seq_length / CP.",
    )
    ap.add_argument(
        "--data-file",
        required=True,
        help="Path to training parquet with 'messages' column.",
    )
    ap.add_argument(
        "--model-dir",
        required=True,
        help="Path to HF model dir (for tokenizer).",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Output JSON only (machine-readable).",
    )
    args = ap.parse_args()

    data_file = args.data_file
    if not Path(data_file).is_file():
        print(f"ERROR: 数据文件不存在: {data_file}", file=sys.stderr)
        sys.exit(1)
    model_dir = args.model_dir
    if not Path(model_dir).is_dir():
        print(f"ERROR: 模型目录不存在: {model_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"正在加载数据: {data_file} ...", file=sys.stderr)
    messages = _load_data(data_file)
    print(f"加载 {len(messages)} 条, 正在 tokenize ...", file=sys.stderr)

    lengths, tok = _tokenize_lengths(messages, model_dir)
    print(f"tokenize 完成, 共 {len(lengths)} 条, "
          f"vocab_size={tok.vocab_size}", file=sys.stderr)

    stats = _compute_stats(lengths)
    recommended_seq = _recommend_seq_length(stats)
    recommended_cp = _recommend_cp(recommended_seq)

    if args.json:
        print(json.dumps({
            "stats": {k: int(v) if isinstance(v, (int, float)) else v
                      for k, v in stats.items()},
            "recommended": {
                "seq_length": recommended_seq,
                "cp_size": recommended_cp,
                "cp_recommended": recommended_cp > 1,
            },
            "truncation_risk": {
                "max_sample": stats["max"],
                "recommended_seq": recommended_seq,
                "truncated": stats["max"] > recommended_seq,
                "truncated_tokens": max(0, stats["max"] - recommended_seq),
            },
        }, ensure_ascii=False, indent=2))
    else:
        _print_report(
            stats, recommended_seq, recommended_cp,
            model_dir=model_dir, data_file=data_file,
        )
        _print_yaml_snippet(recommended_seq, recommended_cp, stats)


if __name__ == "__main__":
    main()