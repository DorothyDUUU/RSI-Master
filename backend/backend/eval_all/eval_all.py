#!/usr/bin/env python3
"""Unified ``eval_all`` backend — wraps benchmark suite ``framework/run_eval.py``.

与 ``backend/backend/eval_*`` 分立评测模块的区别: 这里**不实现**任何评分逻辑,
而是把评测委托给统一评测框架 benchmark suite:

    framework/run_eval.py <bench_id> --limit N --trial-name X \
        --model M --openai-api-base <url> --openai-api-token <key> \
        --config-path <override>

这样一个 ``bench_id`` (benchmark suite ``format/<id>/`` 目录名) 就能评测任意
bench, 不再需要为每个新 bench 新写一个 ``eval_<bench>/`` 模块。

CLI 契约与其余 ``eval_*`` 脚本对齐 (pipeline stage-3 用同一套参数调用):

    python eval_all.py \
        --model <serve_dir> --base-url <vllm_url> --api-key sk-123 \
        --max-tokens N --max-workers N --timeout 600 --limit N \
        --run-dir <eval_dir> \
        --bench-id <bmk_bench_id> [--trial-name X] [--split S] [--bmk-root <root>]

输出: ``--run-dir/metrics.json`` (pipeline 用 ``metric_key`` = ``accuracy``
提取 score)。另外把 benchmark suite 产出的 ``reward.json`` / ``samples.jsonl`` /
``config.yaml`` / ``run_eval.log`` 一起拷进 ``--run-dir`` 供 agent 检查。

关键语义:
    - **完整性优先**: 只要 ``incomplete > 0`` 就写 ``status="failed"`` 且
      exit 1，pipeline 不得把局部结果升级成 ``completed``。诊断用 accuracy 在
      ``correct`` 可用时统一按 ``attempted`` 作分母，避免遗漏样本抬高分数。
    - **limit 必须是正数**: ``--limit <= 0`` 会回退到 ``_DEFAULT_LIMIT`` (500),
      绝不传 ``-1`` (generic runner 的 ``_load_rows`` 会把 ``len(rows) >= -1``
      当成 `>= limit`, 无法正常工作)。
    - **纯函数可测**: ``resolve_bmk_root`` / ``resolve_bench_dir`` /
      ``build_override_yaml`` / ``build_run_eval_command`` / ``map_reward_to_metrics``
      都在模块顶层、不依赖子进程, ``main()`` 在 ``__main__`` guard 下。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml


#: benchmark suite 仓库根目录: ``$BENCHMARK_ROOT``, 否则用仓库内自带的
#: ``bmk_everything/`` (本文件位置上溯 3 层 = 仓库根)。
DEFAULT_BMK_ROOT = os.environ.get("BENCHMARK_ROOT", "").strip() or str(
    Path(__file__).resolve().parents[3] / "bmk_everything"
)
#: 与 run_eval.py 一致的 bench_id / trial_name 合法性约束。
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
#: 不传 -1; 任何 <= 0 的 limit 都落到这个默认值。
_DEFAULT_LIMIT = 500
_RUN_EVAL_PY = "framework/run_eval.py"
_VENV_PYTHON = "framework/.venv/bin/python"
_OUTPUT_DIR_RE = re.compile(r"\[run_eval\] output_dir=(\S+)")


# ── 纯函数 (可单测, 不碰子进程 / 文件系统之外的东西) ──────────────────────


def resolve_bmk_root(cli: str | None, env: Mapping[str, str] | None = None) -> Path:
    """解析 benchmark suite 根目录: ``--bmk-root`` > ``$BENCHMARK_ROOT`` > 默认值。

    校验 ``framework/run_eval.py`` 存在, 否则大声失败 (尽早暴露配置错误)。
    """
    env = env if env is not None else os.environ
    root = Path(cli or env.get("BENCHMARK_ROOT") or DEFAULT_BMK_ROOT).resolve()
    if not (root / _RUN_EVAL_PY).exists():
        raise FileNotFoundError(
            f"benchmark suite framework not found under {root} "
            f"(missing {_RUN_EVAL_PY}); set --bmk-root or $BENCHMARK_ROOT."
        )
    return root


def resolve_bench_dir(bmk_root: Path, bench_id: str) -> Path:
    """校验 ``bench_id`` 并返回 ``format/<bench_id>/``, 镜像 run_eval.py 的报错。

    只校验 runnable 所必需的文件 (``run_benchmark.py``), 尽早失败而不是等到
    stage-3 的 child 进程里才报。
    """
    if not SAFE_NAME.fullmatch(bench_id):
        raise ValueError(f"unsafe bench-id {bench_id!r} (must match {SAFE_NAME.pattern})")
    bench_dir = bmk_root / "format" / bench_id
    if not (bench_dir / "run_benchmark.py").exists():
        raise FileNotFoundError(
            f"benchmark {bench_id!r} not runnable in benchmark suite "
            f"(looked at {bench_dir}/run_benchmark.py)"
        )
    return bench_dir


def framework_python(bmk_root: Path, fallback: str | None = None) -> str:
    """benchmark suite venv python; 缺省 fallback 到当前解释器 (镜像 run_eval.py)."""
    candidate = bmk_root / _VENV_PYTHON
    if candidate.exists():
        return str(candidate)
    return fallback or sys.executable


def build_override_yaml(
    *,
    max_tokens: int = 0,
    max_workers: int = 0,
    temperature: float | None = None,
    timeout: int = 0,
) -> dict[str, Any]:
    """只把**正值**的评测 knob 放进 ``custom``, 交给 benchmark suite deep_merge。

    0 / None 表示「用 benchmark suite 该 bench 自身 config.yaml 里的值」。
    温度单独处理: ``0.0`` 是合法的 greedy 值, 只要显式传入就保留。
    """
    custom: dict[str, Any] = {}
    if max_tokens and max_tokens > 0:
        custom["max_tokens"] = int(max_tokens)
    if max_workers and max_workers > 0:
        custom["max_workers"] = int(max_workers)
    if temperature is not None:
        custom["temperature"] = float(temperature)
    if timeout and timeout > 0:
        custom["timeout"] = int(timeout)
    # 通用附加 custom 通道: 调用方可用 EVAL_EXTRA_CUSTOM 注入任意键 (deep_merge 覆盖
    # bench 自身 config.yaml 同键), 默认关闭。2026-09-02 用于把部分 bench 的
    # dataset_path 从构建机/相对路径覆盖为显式绝对路径 (如 aime2026 的
    # "data/datasets/aime2026" 是 repo-root 相对, run_eval 以 cwd=bench_dir 起 runner
    # 会解析错位), 不污染 eval_all 全局逻辑。
    extra_raw = os.environ.get("EVAL_EXTRA_CUSTOM", "").strip()
    if extra_raw:
        try:
            extra = json.loads(extra_raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"EVAL_EXTRA_CUSTOM 不是合法 JSON 对象: {exc}") from exc
        if not isinstance(extra, dict):
            raise ValueError("EVAL_EXTRA_CUSTOM 必须是 JSON object")
        for key, value in extra.items():
            custom[str(key)] = value
    return {"custom": custom} if custom else {}


def build_run_eval_command(
    *,
    bmk_root: Path,
    bench_id: str,
    limit: int,
    trial_name: str,
    model: str,
    base_url: str,
    api_key: str,
    override_yaml: Path,
    split: str | None = None,
) -> list[str]:
    """组 ``run_eval.py`` 命令行 (参数顺序固定, 便于测试断言)."""
    cmd = [
        framework_python(bmk_root),
        str(bmk_root / _RUN_EVAL_PY),
        bench_id,
        "--limit",
        str(limit),
        "--trial-name",
        trial_name,
        "--model",
        model,
        "--openai-api-base",
        base_url,
        "--openai-api-token",
        api_key,
        "--config-path",
        str(override_yaml),
    ]
    if split:
        cmd += ["--split", split]
    return cmd


def map_reward_to_metrics(
    reward_json: dict[str, Any],
    *,
    bench_id: str,
    metric_key: str = "accuracy",
    trial_name: str = "",
) -> dict[str, Any]:
    """把 benchmark suite 的 ``reward.json`` 映射成 pipeline 的 metrics dict.

    两种 reward 结构:
      - legacy runner (gsm8k):      ``meta_data.accuracy``
      - generic runner (其余 bench): ``meta_data.observed_reward``
    优先读取 runner 指标；只要有 ``correct/attempted`` 就使用完整
    attempted 分母重算 accuracy。部分请求失败会标记 degraded，
    但不会丢掉整场已获得的分数；只有零条有效请求才判 failed。
    ``raw_reward`` 保留 ``reward`` 原值 (incomplete / unscored 时 benchmark suite
    会写 None)。
    """
    meta = reward_json.get("meta_data") or {}
    attempted = int(meta.get("attempted") or 0)
    completed = int(meta.get("completed") or 0)
    incomplete = int(meta.get("incomplete") or 0)
    correct = int(meta.get("correct") or 0)
    scoring_supported = bool(meta.get("scoring_supported", True))

    metric = meta.get(metric_key)
    if metric is None:
        metric = meta.get("observed_reward")
    if metric is None:
        metric = meta.get("accuracy")
    if isinstance(metric, bool) or not isinstance(metric, (int, float)):
        metric = None
    if attempted > 0 and "correct" in meta:
        metric = correct / attempted

    raw_reward = reward_json.get("reward")
    if isinstance(raw_reward, bool) or not isinstance(raw_reward, (int, float)):
        raw_reward = None

    if metric is None or attempted <= 0 or completed <= 0:
        status = "failed"
    else:
        status = "completed"
    return {
        "benchmark": bench_id,
        "trial_name": trial_name,
        "metric_key": metric_key,
        "accuracy": metric,
        "raw_reward": raw_reward,
        "attempted": attempted,
        "completed": completed,
        "incomplete": incomplete,
        "correct": correct,
        "scoring_supported": scoring_supported,
        "status": status,
        "evaluation_status": "degraded" if status == "completed" and incomplete > 0 else status,
        "infra_failures": incomplete,
    }


def find_output_dir(log_text: str, bmk_root: Path, bench_id: str, trial_name: str) -> Path:
    """定位 ``run_eval.py`` 的产出目录。

    优先解析 stdout 里的 ``[run_eval] output_dir=<dir>`` 行 (run_eval.py:214),
    这是真实落盘位置; 解析不到再回退到模板推导路径 ``data/output/<id>/<trial>``。
    """
    match = _OUTPUT_DIR_RE.search(log_text)
    if match:
        return Path(match.group(1))
    return bmk_root / "data" / "output" / bench_id / trial_name


# ── CLI ────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="eval_all: evaluate a trained model on a benchmark suite bench."
    )
    # 与 pipeline stage-3 契约对齐的公共参数
    parser.add_argument("--model", required=True, help="Model name as registered in the vLLM endpoint")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default="sk-123")
    parser.add_argument("--max-tokens", type=int, default=0, help="0 = use the bench's own config")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-workers", type=int, default=0, help="0 = use the bench's own config")
    parser.add_argument("--timeout", type=int, default=0, help="0 = use the bench's own config")
    parser.add_argument("--limit", type=int, default=0, help="<=0 -> default 500; never -1")
    parser.add_argument("--run-dir", default=".", help="Output dir; writes metrics.json inside")
    # eval_all 专属
    parser.add_argument("--bench-id", required=True, help="benchmark suite format/<id> directory name")
    parser.add_argument("--metric-key", default="accuracy", help="metrics.json key for the final score")
    parser.add_argument("--trial-name", default="", help="benchmark suite trial name (default: <bench_id>-<ts>)")
    parser.add_argument("--split", default=None, help="Bench split (default: resolved by run_eval)")
    parser.add_argument("--bmk-root", default=None, help="benchmark suite repo root (default: env/default path)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # judge (grader) 凭据不再在这里硬编码: 自动 pipeline 由 serve_eval._judge_exports
    # 把 config ``evaluation.eval_all.judge`` 的 base_url/api_key export 进 pipeline.sh,
    # 手动重评时由调用方 export OPENAI_BASE_URL / OPENAI_API_KEY. 缺少凭据时
    # run_benchmark 会打印 "grader_api_token_env is not configured", 评测以
    # completed=0 / status=failed 大声失败 —— 绝不静默记假 0.0.

    try:
        bmk_root = resolve_bmk_root(args.bmk_root)
        bench_dir = resolve_bench_dir(bmk_root, args.bench_id)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[eval_all] fatal: {exc}", file=sys.stderr)
        return 1

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # limit 必须是正数; 0 / 负数回退到默认, 绝不传 -1.
    limit = int(args.limit or 0)
    if limit <= 0:
        limit = _DEFAULT_LIMIT

    trial_name = args.trial_name or f"{args.bench_id}-eval"
    if not SAFE_NAME.fullmatch(trial_name):
        print(
            f"[eval_all] fatal: unsafe trial-name {trial_name!r} "
            f"(must match {SAFE_NAME.pattern})",
            file=sys.stderr,
        )
        return 1

    # 组装 override config -> --config-path (只带正值的 custom.*)
    override = build_override_yaml(
        max_tokens=args.max_tokens,
        max_workers=args.max_workers,
        temperature=args.temperature,
        timeout=args.timeout,
    )
    # The pipeline passes the candidate endpoint through --base-url while
    # OPENAI_BASE_URL remains the external judge endpoint. benchmark suite
    # rewrites OPENAI_API_BASE for its candidate subprocess, so a placeholder
    # judge_api_base would otherwise resolve back to the local vLLM server.
    judge_base = os.environ.get("OPENAI_BASE_URL", "").strip()
    judge_token = os.environ.get("OPENAI_API_KEY", "").strip()
    if judge_base and judge_base.rstrip("/") != args.base_url.rstrip("/"):
        custom = override.setdefault("custom", {})
        custom["judge_api_base"] = judge_base
        if judge_token:
            custom["judge_api_token"] = judge_token
    override_yaml = run_dir / "bmk_override.yaml"
    override_yaml.write_text(
        yaml.safe_dump(override, allow_unicode=True, sort_keys=False) if override else "# empty override\n",
        encoding="utf-8",
    )

    cmd = build_run_eval_command(
        bmk_root=bmk_root,
        bench_id=args.bench_id,
        limit=limit,
        trial_name=trial_name,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        override_yaml=override_yaml,
        split=args.split,
    )

    # Arena's official candidate generator reads the endpoint credential from
    # VLLM_API_KEY rather than from benchmark suite's resolved config.  Preserve
    # the CLI contract by forwarding the same candidate key into the child
    # environment; judge credentials remain the separate OPENAI_* variables.
    child_env = dict(os.environ)
    child_env["VLLM_API_KEY"] = args.api_key

    # 子进程: stdout+stderr 都进 run_eval.log, 供 agent 与解析器使用.
    log_path = run_dir / "run_eval.log"
    with log_path.open("wb") as log_fh:
        proc = subprocess.run(
            cmd,
            cwd=str(bmk_root),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=child_env,
        )

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    output_dir = find_output_dir(log_text, bmk_root, args.bench_id, trial_name)
    reward_path = output_dir / "reward.json"

    metrics: dict[str, Any] | None = None
    if reward_path.exists():
        try:
            with reward_path.open(encoding="utf-8") as fh:
                reward = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[eval_all] warning: unreadable reward.json: {exc}", file=sys.stderr)
            reward = None
        if reward is not None:
            metrics = map_reward_to_metrics(
                reward,
                bench_id=args.bench_id,
                metric_key=args.metric_key,
                trial_name=trial_name,
            )
            # 把 benchmark suite 产出拷进 run-dir, agent 不用翻 data/output.
            for name in ("reward.json", "samples.jsonl", "config.yaml"):
                src = output_dir / name
                if src.exists():
                    try:
                        shutil.copy2(src, run_dir / name)
                    except OSError as exc:
                        print(f"[eval_all] warning: copy {src}: {exc}", file=sys.stderr)

    if metrics is None:
        metrics = {
            "benchmark": args.bench_id,
            "trial_name": trial_name,
            "metric_key": args.metric_key,
            "accuracy": None,
            "raw_reward": None,
            "attempted": 0,
            "completed": 0,
            "incomplete": 0,
            "correct": 0,
            "scoring_supported": False,
            "status": "failed",
            "failure_reason": (
                f"run_eval produced no reward.json (rc={proc.returncode}); "
                f"see {log_path}"
            ),
        }
        exit_code = 1
    elif metrics["status"] == "failed":
        metrics["failure_reason"] = (
            f"incomplete evaluation for {args.bench_id} "
            f"(completed={metrics['completed']}, incomplete={metrics['incomplete']}, "
            f"attempted={metrics['attempted']}); see {log_path}"
        )
        exit_code = 1
    elif metrics["accuracy"] is None:
        metrics["failure_reason"] = (
            f"no numeric score for {args.bench_id} "
            f"(scoring_supported={metrics['scoring_supported']}, "
            f"completed={metrics['completed']}, incomplete={metrics['incomplete']}); "
            f"see {log_path}"
        )
        exit_code = 1
    else:
        exit_code = 0

    metrics["output_dir"] = str(output_dir)
    metrics["log"] = str(log_path)
    metrics["limit"] = limit
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)
    print(
        f"[eval_all] {args.bench_id} status={metrics['status']} "
        f"accuracy={metrics['accuracy']} (completed={metrics['completed']}/"
        f"attempted={metrics['attempted']}) rc={proc.returncode} exit={exit_code}"
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
