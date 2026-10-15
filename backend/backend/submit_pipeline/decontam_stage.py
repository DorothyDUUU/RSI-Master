"""Runtime stage-0 decontamination entry — runs INSIDE pipeline.sh on the GPU node.

设计: 去污 (decontamination) 是管线自洽的一环, 逻辑**单一来源**于
``tools/ExpOS/expos/service/decontamination.py`` (controller 侧 WorkerService.submit
与 serverless GPU stage-0 共用同一份; 本文件的兄弟 ``decontaminate.py`` 只是把它按
路径独立加载的 loader, 非逻辑 fork)。controller 已把原始数据先净化为
``decontaminated_train.*``, stage-0 的复清 (通常 0 命中) 保底防漏。

launch 环境 (.venv) 保持轻量 (无 pyarrow), 因此 stage-0 用 heavy python
(llama_factory_env / slime env, 有 pyarrow) 执行:

- ``llama_factory`` 路径: 读原始 TRAIN_DATA_PATH (Alpaca JSONL) -> 剔除命中评测
  query 的行 -> 写 ``<workspace>/llama_factory_data/decontaminated_train.jsonl``,
  再按训练 config 的 ``dataset`` 名自动生成 ``dataset_info.json`` 注册表
  (单命名 dataset, ``file_name`` 指向净化副本) —— 替代 ExpOS 里
  ``_stage_decontaminated_dataset`` 的渲染期 staging。
- ``slime`` 路径: 读原始 TRAIN_DATA_PATH (messages 列 parquet) -> 剔除命中行 ->
  写 ``<workspace>/artifacts/decontaminated_train.parquet``。

污染比对是精确匹配 (NFKC + 空白折叠)。benchmark query 源不可用 (该 bench 的
format/<id>/ 没有 query jsonl) 时**降级为 pass-through**: 数据原样拷成净化副本,
report 里记录 skipped 原因, 不卡死整轮 (评测分如实反映泄漏, 不静默掩盖)。

用法 (pipeline.sh stage-0 内调用)::

    python decontam_stage.py --mode llama_factory \\
        --data <raw> --train-config <yaml> --bench-id <id> --bmk-root <root> \\
        --benchmark <label> --out-jsonl <clean> --registry-dir <lf_data_dir> \\
        --report <artifacts/decontamination_report.json>
    python decontam_stage.py --mode slime \\
        --data <raw> --bench-id <id> --bmk-root <root> \\
        --benchmark <label> --out-parquet <clean> --report <...>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# 独立脚本: 通过 __file__ 目录加载兄弟模块 decontaminate.py (heavy python 直接跑
# 文件路径, 不依赖 repo 在 PYTHONPATH 上)。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from decontaminate import (  # noqa: E402
    DecontaminationReport,
    decontaminate_jsonl,
    decontaminate_parquet,
    load_benchmark_queries,
)

# benchmark suite 根目录: $BENCHMARK_ROOT, 否则用仓库内自带的 bmk_everything/。
# (本脚本按文件路径执行, 不能用包的相对 import, 所以在这里内联同一套解析。)
BENCHMARK_ROOT = os.environ.get("BENCHMARK_ROOT", "").strip() or str(
    Path(__file__).resolve().parents[3] / "bmk_everything"
)


def _load_queries(bench_id: str | None, benchmark: str, bmk_root: str) -> tuple[list, str]:
    """Load benchmark queries. Returns (queries, note). 不可用 -> ([], reason)."""
    if not bench_id:
        return [], "no bench_id supplied; decontamination skipped (pass-through)"
    try:
        queries = load_benchmark_queries(
            benchmark,
            bench_id=bench_id,
            eval_all={"bmk_root": bmk_root},
        )
    except Exception as exc:  # noqa: BLE001 — 降级不卡轮
        return [], f"benchmark queries unavailable ({exc}); pass-through"
    if not queries:
        return [], f"benchmark {bench_id!r} yielded no queries; pass-through"
    return queries, ""


def _write_report(report_path: Path, report: DecontaminationReport | None, note: str) -> None:
    payload = {
        "decontamination": "done" if report is not None else "skipped",
        "benchmark": report.benchmark if report else None,
        "original_rows": report.original_rows if report else None,
        "removed_rows": report.removed_rows if report else None,
        "retained_rows": report.retained_rows if report else None,
        "contamination_rate": report.contamination_rate if report else None,
        "output_path": report.output_path if report else None,
        "removed_query_hashes": list(report.removed_query_hashes) if report else [],
        "note": note,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _detect_format(row: dict) -> tuple[str, dict]:
    """Auto-detect LlamaFactory dataset format from the first data row.

    ``columns`` maps LLaMA-Factory *internal* column names -> the row's actual
    field names (see ``llamafactory/data/parser.py::DatasetAttr``: alpaca defaults
    are ``prompt=instruction`` / ``query=input`` / ``response=output``). Writing the
    row field names as keys (instruction/input/output) makes the preflight reject
    the registry with "columns keys are not valid internal names".
    """
    if isinstance(row.get("messages"), list):
        return "sharegpt", {"messages": "messages"}
    if "instruction" in row:
        return "alpaca", {"prompt": "instruction", "query": "input", "response": "output"}
    raise ValueError(
        "cannot auto-detect training data format: expected Alpaca "
        "(`instruction`/`input`/`output`) or sharegpt (`messages`) columns"
    )


def _stage_llama_factory(args: argparse.Namespace) -> int:
    raw = Path(args.data)
    if not raw.is_file():
        print(f"ERROR: training data not found: {raw}", file=sys.stderr)
        return 2
    import yaml

    try:
        train_cfg = yaml.safe_load(Path(args.train_config).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: cannot read training config {args.train_config}: {exc}", file=sys.stderr)
        return 2
    dataset_names = (train_cfg or {}).get("dataset") if isinstance(train_cfg, dict) else None
    if isinstance(dataset_names, str):
        dataset_names = [n.strip() for n in dataset_names.split(",") if n.strip()]
    if not isinstance(dataset_names, list) or len(dataset_names) != 1:
        print(
            "ERROR: llama_factory decontamination requires exactly one named training "
            f"dataset (got {dataset_names!r})",
            file=sys.stderr,
        )
        return 2

    queries, note = _load_queries(args.bench_id, args.benchmark, args.bmk_root)
    registry_dir = Path(args.registry_dir)
    registry_dir.mkdir(parents=True, exist_ok=True)
    clean_jsonl = Path(args.out_jsonl) if args.out_jsonl else registry_dir / "decontaminated_train.jsonl"

    report: DecontaminationReport | None = None
    if queries:
        report = decontaminate_jsonl(raw, clean_jsonl, queries, benchmark=args.benchmark)
    else:
        clean_jsonl.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(raw, clean_jsonl)

    # 自动注册表: 让 stage-1 的 dataset_dir 只认本次净化副本, 不依赖 run 副本注册。
    dataset_name = str(dataset_names[0])
    sample: dict = {}
    with raw.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                sample = json.loads(line)
                break
    try:
        formatting, columns = _detect_format(sample)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    entry = {
        "file_name": clean_jsonl.name,
        "formatting": formatting,
        "columns": columns,
    }
    (registry_dir / "dataset_info.json").write_text(
        json.dumps({dataset_name: entry}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_report(Path(args.report), report, note)
    print(
        f"stage-0 decontam: bench={args.bench_id} dataset={dataset_name} "
        f"format={formatting} clean={clean_jsonl} report={args.report}"
    )
    return 0


def _stage_slime(args: argparse.Namespace) -> int:
    raw = Path(args.data)
    if not raw.is_file():
        print(f"ERROR: training data not found: {raw}", file=sys.stderr)
        return 2
    queries, note = _load_queries(args.bench_id, args.benchmark, args.bmk_root)
    clean_parquet = Path(args.out_parquet)
    report: DecontaminationReport | None = None
    if queries:
        report = decontaminate_parquet(raw, clean_parquet, queries, benchmark=args.benchmark)
    else:
        clean_parquet.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(raw, clean_parquet)
    _write_report(Path(args.report), report, note)
    print(
        f"stage-0 decontam: bench={args.bench_id} clean={clean_parquet} report={args.report}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="pipeline stage-0 decontamination")
    parser.add_argument("--mode", choices=["llama_factory", "slime"], required=True)
    parser.add_argument("--data", required=True, help="raw training data path")
    parser.add_argument("--train-config", default="", help="llama_factory training YAML")
    parser.add_argument("--bench-id", default="", help="benchmark suite format/<id>/ name")
    parser.add_argument("--bmk-root", default=os.environ.get("BENCHMARK_ROOT", ""))
    parser.add_argument("--benchmark", default="", help="report label (benchmark name)")
    parser.add_argument("--out-jsonl", default="", help="clean JSONL destination (LF)")
    parser.add_argument("--out-parquet", default="", help="clean parquet destination (slime)")
    parser.add_argument("--registry-dir", default="", help="llama_factory_data dir (LF)")
    parser.add_argument("--report", required=True, help="decontamination_report.json path")
    args = parser.parse_args(argv)

    if args.mode == "llama_factory":
        return _stage_llama_factory(args)
    return _stage_slime(args)


if __name__ == "__main__":
    sys.exit(main())
