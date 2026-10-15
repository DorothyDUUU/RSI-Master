#!/usr/bin/env python3
"""Aggregate one benchmark batch into numeric scores for BENCHMARK_ANALYSIS.md.

Reads every ``data/output/<id>/<trial>/reward.json`` (+ ``samples.jsonl`` when present) and
emits a JSON summary plus a markdown fragment covering:

- overall completion rate and scored/partial/unscored counts
- aggregate mean reward by domain and by scorer (observed_reward is used when
  ``reward`` is null because of incomplete samples; ``unscored`` benchmarks are excluded)
- Top-10 / Bottom-10 scored benchmarks

Usage::

    framework/.venv/bin/python framework/automation/summarize_batch.py \
      --trial qwen36-20260808 --format-dir format \
      --out data/output/qwen36-batch/qwen36-20260808/analysis.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


LEGACY_TRIAL_DEFAULT = "qwen36-legacy-20260808"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", required=True)
    parser.add_argument("--legacy-trial", type=Path,
                        help="trial dir holding the 3 hand-written legacy rewards "
                             f"(default: {LEGACY_TRIAL_DEFAULT})")
    parser.add_argument("--extra-trial", type=Path,
                        help="trial dir holding an additional cohort of rewards (e.g. the "
                             "46 new domain-knowledge benchmarks); the cohort ids are read "
                             "from --extra-ids")
    parser.add_argument("--extra-ids", type=Path,
                        help="file whose lines are benchmark ids belonging to --extra-trial")
    parser.add_argument("--format-dir", type=Path, default=REPO / "format")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    extra_ids = set()
    if args.extra_ids and args.extra_ids.exists():
        extra_ids = {line.strip() for line in args.extra_ids.read_text().splitlines() if line.strip()}

    def trial_for(bench_id: str) -> str:
        # The 3 legacy benchmarks run under their own trial (run_legacy_batch / repair_legacy_batch).
        if bench_id in ("gsm8k", "gpqa", "aime2025"):
            return str(args.legacy_trial) if args.legacy_trial else LEGACY_TRIAL_DEFAULT
        if args.extra_trial and bench_id in extra_ids:
            return str(args.extra_trial)
        return args.trial

    # benchmark id -> (domain, scorer)
    meta = {}
    for spec_path in sorted((args.format_dir).glob("*/spec.json")):
        try:
            spec = json.loads(spec_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        try:
            cfg = json.loads((spec_path.parent / "config.json").read_text())
        except (OSError, json.JSONDecodeError):
            cfg = {}
        meta[spec.get("benchmark-id")] = (cfg.get("domain") or spec.get("domain") or "-",
                                          spec.get("scorer"))
    for _id, dom, scorer in (("gsm8k", "math_reasoning", "exact_match"),
                             ("gpqa", "domain_knowledge", "multiple_choice"),
                             ("aime2025", "math_reasoning", "numeric")):
        meta.setdefault(_id, (dom, scorer))

    results = []
    for bench_id, (domain, scorer) in sorted(meta.items()):
        trial = trial_for(bench_id)
        reward_path = REPO / f"data/output/{bench_id}/{trial}/reward.json"
        samples_path = REPO / f"data/output/{bench_id}/{trial}/samples.jsonl"
        if not reward_path.exists():
            continue
        reward = json.loads(reward_path.read_text())
        meta_data = reward.get("meta_data") or {}
        attempted = meta_data.get("attempted")
        completed = meta_data.get("completed")
        incomplete = meta_data.get("incomplete")
        observed = meta_data.get("observed_reward")
        if observed is None:
            # Legacy benches (gsm8k/gpqa/aime2025) record accuracy instead of observed_reward.
            observed = meta_data.get("accuracy")
        # Sanity re-count of empty-content samples from the trajectory when available.
        if samples_path.exists() and incomplete is not None:
            empty = 0
            for line in samples_path.read_text().split("\n"):
                if not line.strip():
                    continue
                sample = json.loads(line)
                if not sample.get("complete"):
                    empty += 1
            incomplete = incomplete if incomplete is not None else empty
        # reward is null when there were incomplete samples; observed_reward is the
        # accuracy over completed ones — use it so partial runs still yield a score.
        scored = reward.get("reward") if reward.get("reward") is not None else observed
        results.append({
            "benchmark": bench_id, "domain": domain, "scorer": scorer,
            "attempted": attempted, "completed": completed, "incomplete": incomplete,
            "failures": meta_data.get("failures"),
            "reward": reward.get("reward"), "observed_reward": observed, "score": scored,
        })

    scored_results = [r for r in results if r["scorer"] != "unscored"]
    partial = [r for r in scored_results if r["incomplete"]]
    scored = [r for r in scored_results if r["score"] is not None]

    summary = {
        "trial": args.trial,
        "legacy_trial": trial_for("gsm8k"),
        "extra_trial": str(args.extra_trial) if args.extra_trial else None,
        "results": [{k: r[k] for k in ("benchmark", "domain", "scorer", "attempted",
                                       "completed", "incomplete", "failures", "reward",
                                       "observed_reward", "score")} for r in results],
        "benchmarks_with_output": len(results),
        "scored_benchmarks": len(scored),
        "partial_completion": len(partial),
        "unscored": len([r for r in results if r["scorer"] == "unscored"]),
        "overall_completed": sum(r["completed"] or 0 for r in results),
        "overall_attempted": sum(r["attempted"] or 0 for r in results),
        "overall_failures": sum(r["failures"] or 0 for r in results),
        "score_mean": round(statistics.mean(r["score"] for r in scored), 4) if scored else None,
        "score_median": round(statistics.median(r["score"] for r in scored), 4) if scored else None,
        "score_min": round(min(r["score"] for r in scored), 4) if scored else None,
        "score_max": round(max(r["score"] for r in scored), 4) if scored else None,
        "by_domain": {},
        "by_scorer": {},
        "top10": sorted(scored, key=lambda r: -r["score"])[:10],
        "bottom10": sorted(scored, key=lambda r: r["score"])[:10],
    }
    for key, label in (("domain", "by_domain"), ("scorer", "by_scorer")):
        grouped = defaultdict(list)
        for r in scored:
            grouped[r[key]].append(r["score"])
        summary[label] = {k: {"n": len(v), "mean": round(statistics.mean(v), 4),
                              "min": round(min(v), 4), "max": round(max(v), 4)}
                          for k, v in sorted(grouped.items())}

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    # Markdown fragment for BENCHMARK_ANALYSIS.md section 7.
    def fmt(row: dict) -> str:
        score = row["score"]
        fails = f"（{row['failures']} 空答判 0 分）" if row.get("failures") else ""
        return (f"| {row['benchmark']} | {row['domain']} | {row['scorer']} "
                f"| {score if score is None else round(score,4)} "
                f"| {row['completed']}/{row['attempted']}{fails} |")

    lines = [
        f"### 7.2 总体",
        "",
        f"| 项 | 值 |",
        f"|---|---|",
        f"| 有产出 benchmark | {summary['benchmarks_with_output']} |",
        f"| 有分数 benchmark（排除 unscored） | {summary['scored_benchmarks']} |",
        f"| 部分完成（空答已判 0 分；仅剩传输故障） | {summary['partial_completion']} |",
        f"| unscored（只记轨迹） | {summary['unscored']} |",
        f"| 空答判失败样本（计入分母，0 分） | {summary['overall_failures']} |",
        f"| 总体完成率 | {summary['overall_completed']}/{summary['overall_attempted']} = {100*summary['overall_completed']/max(summary['overall_attempted'],1):.2f}% |",
        f"| 平均分（各 benchmark 等权） | {summary['score_mean']} |",
        f"| 中位分 | {summary['score_median']} |",
        f"| 最低 / 最高 | {summary['score_min']} / {summary['score_max']} |",
        "",
        "### 7.3 按领域聚合（等权平均分）",
        "",
        "| 领域 | n | mean | min | max |",
        "|---|---|---|---|---|",
    ]
    for k, v in summary["by_domain"].items():
        lines.append(f"| {k} | {v['n']} | {v['mean']} | {v['min']} | {v['max']} |")
    lines += ["", "### 7.4 按打分器聚合", "", "| 打分器 | n | mean | min | max |", "|---|---|---|---|---|"]
    for k, v in summary["by_scorer"].items():
        lines.append(f"| {k} | {v['n']} | {v['mean']} | {v['min']} | {v['max']} |")
    lines += ["", "### 7.5 Top 10", "", "| benchmark | 领域 | 打分器 | 分数 | 完成 |", "|---|---|---|---|---|"]
    lines += [fmt(r) for r in summary["top10"]]
    lines += ["", "### 7.6 Bottom 10", "", "| benchmark | 领域 | 打分器 | 分数 | 完成 |", "|---|---|---|---|---|"]
    lines += [fmt(r) for r in summary["bottom10"]]

    print("\n".join(lines))
    print(f"\n[json summary] {args.out}")


if __name__ == "__main__":
    main()
