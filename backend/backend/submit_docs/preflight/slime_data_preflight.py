#!/usr/bin/env python3
"""slime SFT 训练数据预检 (数据格式校验, GPU 入队前).

在占住 GPU 前用 pyarrow 校验 parquet 训练数据格式, 快速失败不烧 GPU:
  1. 文件存在且是 pyarrow 可读的 parquet;
  2. 有 ``messages`` 列 (slime 训练数据 = OpenAI chat 格式);
  3. 每行 ``messages`` 是非空 chat 数组 (元素是含 role/content 的 dict);
  4. 行数 > 0。

**污染/leakage 检查不在这里** —— WorkerService.submit 已先做 decontaminate_parquet
(benchmark query 去污染), 本脚本只兜底数据格式。

用法 (slime conda env):
  python slime_data_preflight.py --data-file /path/to/train.parquet

退出码: 0 = 通过; 1 = 格式校验失败 (pipeline stage-0 据此快速失败)。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow.parquet as pq


def _validate_messages(value: object) -> list[str]:
    """Return a list of problems for one row's ``messages`` value ([] = valid)."""
    if not isinstance(value, list):
        return [f"messages 必须是 list (OpenAI chat 数组), got {type(value).__name__}"]
    if not value:
        return ["messages 为空 list (至少一个 turn)"]
    problems: list[str] = []
    for i, turn in enumerate(value):
        if not isinstance(turn, dict):
            problems.append(f"messages[{i}] 不是 dict (got {type(turn).__name__})")
            continue
        if not str(turn.get("role") or "").strip():
            problems.append(f"messages[{i}] 缺非空 role 字段")
        if not str(turn.get("content") or "").strip():
            problems.append(f"messages[{i}] 缺非空 content 字段")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="slime SFT 训练数据预检 (数据格式校验)")
    parser.add_argument("--data-file", required=True, help="训练 parquet 路径")
    args = parser.parse_args()

    data_file = Path(args.data_file)
    if not data_file.is_file():
        print(f"[FAIL] 训练数据不存在: {data_file}", file=sys.stderr)
        return 1

    try:
        table = pq.read_table(data_file, columns=["messages"])
    except Exception as exc:  # noqa: BLE001 — 无法解析的 parquet 直接判失败
        print(f"[FAIL] 不是 pyarrow 可读的 parquet: {exc}", file=sys.stderr)
        return 1

    n_rows = table.num_rows
    if n_rows <= 0:
        print("[FAIL] parquet 0 行 —— 没有可训练数据", file=sys.stderr)
        return 1
    if "messages" not in table.column_names:
        cols = ", ".join(table.column_names)
        print(f"[FAIL] 缺 messages 列; 实际列: {cols}", file=sys.stderr)
        return 1

    sample = table.column("messages").to_pylist()
    bad_rows = 0
    first_problem = ""
    for i, value in enumerate(sample):
        problems = _validate_messages(value)
        if problems:
            bad_rows += 1
            if not first_problem:
                first_problem = f"row {i}: {problems[0]}"
    if bad_rows:
        print(
            f"[FAIL] {bad_rows}/{n_rows} 行 messages 格式非法; 首个问题: {first_problem}",
            file=sys.stderr,
        )
        return 1

    print(f"[OK] slime 数据预检通过: {n_rows} 行, messages 列格式合法 ({data_file})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
