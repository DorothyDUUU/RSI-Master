#!/usr/bin/env python3
"""Regenerate BENCHMARK_ANALYSIS.md for the 244-benchmark cohort.

Sections 1-6 are derived deterministically from ``format/*/spec.json`` (+ the 3
legacy benchmarks).  Section 7.2-7.6 come from the ``summarize_batch.py`` JSON
output (``--out analysis.json``); 7.1 and 7.7 are carried over with cohort-wide
numbers patched in.

Usage::

    framework/.venv/bin/python framework/automation/generate_analysis_md.py \
      --analysis data/output/qwen36-batch/qwen36-20260808/analysis.json \
      --out BENCHMARK_ANALYSIS.md
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FORMAT_DIR = REPO / "format"

# legacy: id -> (domain, scorer, rows, adapter, spdx)
LEGACY = {
    "gsm8k": ("math_reasoning", "exact_match", 2638, "legacy_runner", "mit"),
    "gpqa": ("domain_knowledge", "multiple_choice", 448, "legacy_runner", "mit"),
    "aime2025": ("math_reasoning", "numeric", 90, "legacy_runner", "other"),
}

SCORER_DESC = {
    "multiple_choice": "多选字母匹配",
    "exact_match": "精确/归一化文本匹配",
    "numeric": "数值容差匹配",
    "boolean": "布尔判答",
    "label": "标签匹配",
    "squad": "SQuAD span/F1 匹配",
    "json": "JSON 规范匹配",
    "unscored": "无确定性打分（需 required-metric，只记答案不记 reward）",
}


def load_specs() -> list[dict]:
    specs = []
    for path in sorted(FORMAT_DIR.glob("*/spec.json")):
        try:
            spec = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if spec.get("benchmark-id"):
            specs.append(spec)
    return specs


def local_rows(spec: dict) -> int:
    return int(spec.get("dataset", {}).get("local_rows") or 0)


def fmt_int(n: int) -> str:
    return f"{n:,}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path,
                        default=REPO / "data/output/qwen36-batch/qwen36-20260808/analysis.json")
    parser.add_argument("--out", type=Path, default=REPO / "BENCHMARK_ANALYSIS.md")
    args = parser.parse_args()

    specs = load_specs()
    all_rows = [local_rows(s) for s in specs] + [v[2] for v in LEGACY.values()]
    total_rows = sum(all_rows)
    total = len(specs) + len(LEGACY)

    bigbench = sum(1 for s in specs if s.get("adapter") == "bigbench")
    independent = total - bigbench - len(LEGACY)

    # ---------------- Section 1 ----------------
    s1 = [
        "# Benchmark 目录数值分析",
        "",
        f"> 数据来源：`format/` 下 {len(specs)} 个 benchmark 的 `spec.json` + `config.json`（含 3 个手写 legacy：gsm8k / gpqa / aime2025）。",
        "> 统计口径：`local_rows` 为数据集下载落盘后的真实行数（`data/datasets/<id>/<split>.jsonl`）。",
        "> 统计日期：2026-08-10。",
        "",
        "## 1. 总览",
        "",
        "| 指标 | 数值 |",
        "|---|---|",
        f"| benchmark 总数 | {total} |",
        f"| 题目总数 | {fmt_int(total_rows)} |",
        f"| 最小数据量 | {min(all_rows)} 题 |",
        f"| 最大数据量 | {max(all_rows)} 题 |",
        f"| 数据量中位数 | {statistics.median(all_rows)} 题 |",
        f"| 数据量均值 | {statistics.mean(all_rows):,.0f} 题 |",
        "",
        "### 来源构成",
        "",
        "| 来源 | 数量 | 说明 |",
        "|---|---|---|",
        f"| BIG-bench（tasksource/bigbench） | {bigbench} | 同一仓库 {bigbench} 个 config |",
        f"| 独立 HuggingFace 数据集 | {independent} | 数学 / 知识 / 代码 / 阅读理解 / 领域学科等 |",
        "| 手写 legacy | 3 | gsm8k / gpqa / aime2025 |",
    ]

    # ---------------- Section 2 ----------------
    dom = Counter()
    dom_rows = Counter()
    for s in specs:
        d = s.get("domain") or "-"
        dom[d] += 1
        dom_rows[d] += local_rows(s)
    for key, (d, *_rest) in LEGACY.items():
        dom[d] += 1
        dom_rows[d] += LEGACY[key][2]

    s2 = ["", "## 2. 题目分布", "", "### 2.1 按领域分布", "",
          "| 领域 | benchmark 数 | 题目数 | 占比（题目） |", "|---|---|---|---|"]
    total_dom_rows = sum(dom_rows.values())
    for d, cnt in sorted(dom.items(), key=lambda kv: -kv[1]):
        s2.append(f"| {d} | {cnt} | {fmt_int(dom_rows[d])} | {100*dom_rows[d]/total_dom_rows:.1f}% |")
    s2.append(f"| **合计** | **{total}** | **{fmt_int(total_rows)}** | 100% |")

    scorer = Counter()
    scorer_rows = Counter()
    for s in specs:
        scorer[s.get("scorer")] += 1
        scorer_rows[s.get("scorer")] += local_rows(s)
    for key, (_d, sc, _r, *_rest) in LEGACY.items():
        scorer[sc] += 1
        scorer_rows[sc] += LEGACY[key][2]
    s2 += ["", "### 2.2 按打分器分布", "",
           "| 打分器（scorer） | benchmark 数 | 题目数 | 说明 |", "|---|---|---|---|"]
    for sc, cnt in sorted(scorer.items(), key=lambda kv: -kv[1]):
        s2.append(f"| {sc} | {cnt} | {fmt_int(scorer_rows[sc])} | {SCORER_DESC.get(sc, sc)} |")
    unscored_rows = scorer_rows.get("unscored", 0)
    s2 += ["",
           f"> 注意：{scorer.get('unscored', 0)} 个 `unscored` benchmark 的 {fmt_int(unscored_rows)} 题只产出答案轨迹，不产出标量 reward，不进入分数统计。",
           "", "### 2.3 按适配器分布", "", "| 适配器 | benchmark 数 |", "|---|---|"]
    adapter = Counter(s.get("adapter") for s in specs)
    adapter["legacy_runner"] = len(LEGACY)
    for a, cnt in sorted(adapter.items(), key=lambda kv: -kv[1]):
        s2.append(f"| {a} | {cnt} |")

    # ---------------- Section 3 ----------------
    rows_sorted = sorted(all_rows)
    n = len(rows_sorted)

    def quantile(pct: float) -> int:
        return rows_sorted[min(n - 1, int(pct / 100 * n))]

    s3 = ["", "## 3. 数据规模分布", "", "### 3.1 分位数", "",
          "| 分位 | 题目数 |", "|---|---|", f"| min | {rows_sorted[0]} |",
          f"| q25 | {quantile(25)} |", f"| 中位数 q50 | {quantile(50)} |",
          f"| 均值 | {statistics.mean(all_rows):,.0f} |", f"| q75 | {quantile(75)} |",
          f"| q90 | {quantile(90)} |", f"| q95 | {quantile(95)} |",
          f"| q99 | {quantile(99)} |", f"| max | {rows_sorted[-1]} |",
          "", "### 3.2 规模桶分布", "", "| 规模桶 | benchmark 数 | 题目数 |", "|---|---|---|"]
    bucket_counts = []
    for lo, hi in ((1, 100), (101, 300), (301, 1000), (1001, 3000), (3001, 10000), (10001, 100000)):
        cnt = sum(1 for r in all_rows if lo <= r <= hi)
        tot = sum(r for r in all_rows if lo <= r <= hi)
        bucket_counts.append((lo, hi, cnt, tot))
        s3.append(f"| {lo} – {hi} | {cnt} | {fmt_int(tot)} |")
    small_n = bucket_counts[0][2]
    large_n = bucket_counts[-1][2]
    large_rows = bucket_counts[-1][3]
    s3 += ["",
           f"分布呈「两极」：**{small_n} 个 ≤100 题的小 benchmark**（适合全量快速评测）与 **{large_n} 个 >10,000 题的大 benchmark**（占据 {100*large_rows/total_rows:.1f}% 题目量，评测需采样）。",
           "", "### 3.3 最大的 15 个 benchmark", "",
           "| benchmark | 题目数 | 打分器 | 领域 |", "|---|---|---|---|"]
    all_entries = [(s.get("benchmark-id"), local_rows(s), s.get("scorer"), s.get("domain") or "-") for s in specs]
    all_entries += [(k, v[2], v[1], v[0]) for k, v in LEGACY.items()]
    for bid, rows, sc, d in sorted(all_entries, key=lambda x: -x[1])[:15]:
        s3.append(f"| {bid} | {fmt_int(rows)} | {sc} | {d} |")

    s3 += ["", f"### 3.4 独立 HF 数据集（{independent} 个，非 BIG-bench）", "",
           "按本地行数排序的完整清单：", "", "| benchmark | 题目数 | 来源 |", "|---|---|---|"]
    indep = sorted([(s.get("benchmark-id"), local_rows(s),
                     s.get("source") or s.get("dataset", {}).get("hf", {}).get("path", ""))
                    for s in specs if s.get("adapter") != "bigbench"], key=lambda x: -x[1])
    for bid, rows, src in indep:
        s3.append(f"| {bid} | {fmt_int(rows)} | {src} |")

    # ---------------- Section 4 ----------------
    splits = Counter(s.get("dataset", {}).get("split") for s in specs)
    s4 = ["", "## 4. 本地数据分片", "", "| 分片（split） | benchmark 数 |", "|---|---|"]
    for sp, cnt in sorted(splits.items(), key=lambda kv: -kv[1]):
        s4.append(f"| {sp} | {cnt} |")
    s4 += ["",
           "评测入口 `run_eval.py` 统一以 `default` split 名义运行（`config.json` 声明 `[\"default\"]`），运行时按 `spec.json` 的 `dataset.split` 读取本地对应分片文件。"]

    # ---------------- Section 5 ----------------
    lic = Counter(s.get("license", {}).get("spdx", "unknown") for s in specs)
    s5 = ["", "## 5. 许可协议", "", "| SPDX | benchmark 数 |", "|---|---|"]
    for l, cnt in sorted(lic.items(), key=lambda kv: -kv[1]):
        s5.append(f"| {l} | {cnt} |")
    s5.append("| mit（gsm8k / gpqa） + aime2025 逐题原始版权 | 3（legacy） |")

    # ---------------- Section 6 ----------------
    full = sum(1 for r in all_rows if r <= 500)
    sampled = total - full - len(LEGACY)
    s6 = [
        "", "## 6. 评测覆盖策略（Qwen3.6-35B-A3B）", "",
        f"全量 {total} 个 benchmark 共 {fmt_int(total_rows)} 题，直接全量评测不现实。本批评测采用 **每 benchmark 上限 500 题** 的采样策略：",
        "", f"- `local_rows ≤ 500`：全量跑（{full} 个，含 3 legacy 及新接入的小型域数据集）",
        f"- `local_rows > 500`：抽前 500 题（{sampled} 个采样）",
        "- 含 `qwen36-new46-20260810` 批：46 个新接入 domain_knowledge 数据集",
        "- 空答在 32768/65536 预算下多轮修复后计入失败（见 7.7）",
        "- `unscored` 的 benchmark 只记答案轨迹（samples.jsonl），不产生 reward",
        "",
        "每个 benchmark 的产物：",
        "",
        "```",
        "data/output/<benchmark-id>/<trial>/",
        "├── samples.jsonl   # 逐题轨迹（response / reasoning_content / finish_reason / usage / correct）",
        "├── reward.json     # 标量分数 + meta_data（completed / incomplete / observed_reward）",
        "└── config.yaml     # 本次实际生效配置（已脱敏）",
        "```",
    ]

    # ---------------- Section 7 ----------------
    analysis = {}
    if args.analysis.exists():
        analysis = json.loads(args.analysis.read_text())

    def pct(a: int, b: int) -> str:
        return f"{100*a/max(b,1):.2f}%"

    if analysis:
        scored = [r for r in analysis["results"] if r["scorer"] != "unscored" and r["score"] is not None]
        s7 = [
            "", "## 7. 模型评测结果（Qwen3.6-35B-A3B）", "",
            "> 统计基于 `data/output/qwen36-20260808`（主批）+ `data/output/qwen36-new46-20260810`（新接入 46）+ `data/output/qwen36-legacy-20260808`（3 legacy），",
            "> 汇总脚本 `framework/automation/summarize_batch.py`，产物 `analysis.json`。",
            "", "### 7.1 运行信息", "", "| 项 | 值 |", "|---|---|",
            "| 模型 | Qwen3.6-35B-A3B（vllm，thinking 开启） |",
            "| 端点 | self-hosted router (no auth) |",
            "| 温度 / max_tokens | 0.0 / 16384（新接入批空答经 32768 修复后并入） |",
            "| 空答计分 | `empty_content` 判 0 分计入分母（模型失败）；`request_error` 不计分不进分母 |",
            "| 主批并发 | 150（15 进程 × 10 workers） |",
            "| 修复预算 | 32768（仅重跑空答 index；新接入批 3 组分片并行） |",
            "| 批次 | qwen36-20260808 + qwen36-new46-20260810 + qwen36-legacy-20260808 |",
            "| 覆盖 | 每 benchmark 上限 500 题；`local_rows ≤ 500` 全量，其余抽前 500 |",
            "", "### 7.2 总体", "", "| 项 | 值 |", "|---|---|",
            f"| 有产出 benchmark | {analysis['benchmarks_with_output']} |",
            f"| 有分数 benchmark（排除 unscored） | {analysis['scored_benchmarks']} |",
            f"| 部分完成（空答已判 0 分；仅剩传输故障） | {analysis['partial_completion']} |",
            f"| unscored（只记轨迹） | {analysis['unscored']} |",
            f"| 空答判失败样本（计入分母，0 分） | {analysis['overall_failures']} |",
            f"| 总体完成率 | {analysis['overall_completed']}/{analysis['overall_attempted']} = {pct(analysis['overall_completed'], analysis['overall_attempted'])} |",
            f"| 平均分（各 benchmark 等权） | {analysis['score_mean']} |",
            f"| 中位分 | {analysis['score_median']} |",
            f"| 最低 / 最高 | {analysis['score_min']} / {analysis['score_max']} |",
            "", "### 7.3 按领域聚合（等权平均分）", "",
            "| 领域 | n | mean | min | max |", "|---|---|---|---|---|",
        ]
        for k, v in sorted(analysis["by_domain"].items()):
            s7.append(f"| {k} | {v['n']} | {v['mean']} | {v['min']} | {v['max']} |")
        s7 += ["", "### 7.4 按打分器聚合", "", "| 打分器 | n | mean | min | max |", "|---|---|---|---|---|"]
        for k, v in sorted(analysis["by_scorer"].items()):
            s7.append(f"| {k} | {v['n']} | {v['mean']} | {v['min']} | {v['max']} |")
        s7 += ["", "### 7.5 Top 10", "", "| benchmark | 领域 | 打分器 | 分数 | 完成 |", "|---|---|---|---|---|"]
        for r in analysis["top10"]:
            s7.append(f"| {r['benchmark']} | {r['domain']} | {r['scorer']} | {round(r['score'],4) if r['score'] is not None else '—'} | {r['completed']}/{r['attempted']} |")
        s7 += ["", "### 7.6 Bottom 10", "", "| benchmark | 领域 | 打分器 | 分数 | 完成 |", "|---|---|---|---|---|"]
        for r in analysis["bottom10"]:
            s7.append(f"| {r['benchmark']} | {r['domain']} | {r['scorer']} | {round(r['score'],4) if r['score'] is not None else '—'} | {r['completed']}/{r['attempted']} |")
    else:
        s7 = ["", "## 7. 模型评测结果（Qwen3.6-35B-A3B）", "",
              "> 运行 `summarize_batch.py` 生成 analysis.json 后重跑本脚本填充本节。"]

    lines = s1 + s2 + s3 + s4 + s5 + s6 + s7 + [""]
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
