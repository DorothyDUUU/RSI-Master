#!/usr/bin/env python3
"""validate_on_host — 在 GPU 主机上验证 submit_docs 三份 Qwen3-4B-Base 配置。

目标主机: 一台 GPU 机器 (通过 ``--host`` 指定 SSH 别名; 无默认值)。
该主机与渲染机共享同一个 NFS volume —— 所以本地渲染的
pipeline.sh 路径在主机上直接可执行, 无需拷文件。

两份配置 (llama_factory 已整体下线 2026-08-27, 本集群只跑 slime_sft):
  1. ``slime_sft_qwen3_4b_single_gpu.yaml``      -> launch_sft_smoke.sh, NUM_GPUS=1 TP=1 EP=1
  2. ``slime_sft_qwen3_4b_multi_gpu.yaml``       -> launch_sft_smoke.sh, NUM_GPUS=2 TP=2 EP=1

llama_factory 整体下线回归: ``--dry-run`` 额外断言真实提交路径 (WorkerService.submit
+ render-only backend) 对 llama_factory_sft 请求**必须拒绝** (守卫 (c), 消息含
「已整体下线」)—— 不给 worker 任何 '试一把' 的机会。

slime 两个配置: 4B 是**稠密**模型, 框架 slime_sft 委托的 ``launch_sft.sh`` 是 35B
MoE demo (架构参数硬编码) 不适用; 4B 稠密的正确入口是 ``launch_sft_smoke.sh``
(source 官方 ``qwen3-4B.sh`` spec)。示例配置里的键对两个脚本是同一套 env 契约,
harness 用 smoke 入口 + 配置键值驱动训练, 断言 SMOKE_PASSED 标记。

前置/注意:
  - slime 需先有一次 torch_dist 转换 (约 10-30 分钟): ``--convert``。
  - slime 冒烟只用 ``ray stop --force`` 清理自己的 ray 集群, **禁止 pkill 无差别杀
    python** (见 prompts_0812 公共 prompt 的「进程管理纪律」); harness 在 slime 案例
    前打印节点进程/GPU 作只读观察。
  - GPU 由 harness 显式钉住 (CUDA_VISIBLE_DEVICES): 这是运维工具, 与
    "worker 不设 CUDA_VISIBLE_DEVICES" 的 agent 规则不冲突。

用法:
  # 只渲染, 不 SSH / 不烧 GPU (本地自检)
  python backend/backend/submit_docs/validate_on_host.py --dry-run

  # 只跑某一个配置
  python backend/backend/submit_docs/validate_on_host.py --only slime-single

  # 先做 torch_dist 转换 (slime 前置), 再全量验证
  python backend/backend/submit_docs/validate_on_host.py --convert

  # 全量 (默认): 渲染 + 主机执行 slime 配置
  python backend/backend/submit_docs/validate_on_host.py
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── 路径: submit_docs 在 <repo>/backend/backend/submit_docs ────────────────
REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (REPO_ROOT, REPO_ROOT / "tools" / "ExpOS", REPO_ROOT / "backend"):
    _p_s = _p.as_posix()
    if _p_s not in sys.path:
        sys.path.insert(0, _p_s)

from expos.service.experiment_os import ExperimentOS  # noqa: E402
from expos.service.worker_service import (  # noqa: E402
    WorkerService,
    load_slime_config_file,
)

BASE_MODEL = REPO_ROOT / "public" / "models" / "Qwen" / "Qwen3-4B-Base"
TORCH_DIST_DIR = REPO_ROOT / "public" / "models" / "Qwen" / "Qwen3-4B-Base_torch_dist"

SLIME_WORKSPACE = os.environ.get("SLIME_WORKSPACE", "")  # slime 源码 / Megatron-LM 所在
SLIME_SCRIPTS = REPO_ROOT / "backend" / "backend" / "slime" / "scripts"
SLIME_SMOKE = SLIME_SCRIPTS / "launch_sft_smoke.sh"
SLIME_CONVERT = SLIME_SCRIPTS / "convert_hf_to_torch_dist.sh"
SLIME_MODEL_SPEC = f"{SLIME_WORKSPACE}/slime/scripts/models/qwen3-4B.sh"

# llama_factory 训练时把 dataset 名注册进 run 副本的 registry (见 _setup_lf_run_copy)。
LF_DATASET = "identity"
LF_DATA_REL = Path("public/backend/llama_factory/llama_factory/data")

CASES = {
    "slime-single": {
        "training_mode": "slime_sft",
        "slime_yaml": "configs/slime_sft_qwen3_4b_single_gpu.yaml",
        "gpu": "0",
        "desc": "slime 单卡训练 (launch_sft_smoke, NUM_GPUS=1 TP=1 EP=1)",
    },
    "slime-multi": {
        "training_mode": "slime_sft",
        "slime_yaml": "configs/slime_sft_qwen3_4b_multi_gpu.yaml",
        "gpu": "0,1",
        "desc": "slime 多卡训练 (launch_sft_smoke, NUM_GPUS=2 TP=2 EP=1)",
    },
}


class Result:
    def __init__(self, case: str, desc: str) -> None:
        self.case = case
        self.desc = desc
        self.ok: bool | None = None
        self.detail = ""
        self.elapsed = 0.0

    def finish(self, ok: bool, detail: str, elapsed: float) -> None:
        self.ok = ok
        self.detail = detail
        self.elapsed = elapsed


# ── SSH 助手 ────────────────────────────────────────────────────────────────


def _ssh(host: str, cmd: str, timeout: int = 600) -> tuple[int, str, str]:
    """Run a remote command; returns (rc, stdout, stderr)."""
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
         "-o", "StrictHostKeyChecking=no", host, cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _run_remote(host: str, cmd: str, timeout: int, label: str) -> str:
    """Run remote cmd; raises on nonzero with full output."""
    rc, out, err = _ssh(host, cmd, timeout=timeout)
    combined = (out + "\n" + err).strip()
    if rc != 0:
        raise RuntimeError(f"{label} exited {rc} on {host}:\n{combined}")
    return combined


# ── 合成冒烟数据 ─────────────────────────────────────────────────────────────


def _generate_alpaca_jsonl(path: Path, n: int) -> None:
    """Novel arithmetic Q/A (绝不与 gsm8k 评测 query 精确相等)."""
    rows = []
    for i in range(n):
        a, b = 37 + i, 29 + i
        rows.append(
            {
                "instruction": f"What is {a} + {b}? Answer with just the number.",
                "input": "",
                "output": str(a + b),
            }
        )
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


def _generate_slime_parquet(path: Path, n: int) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    messages = []
    metadata = []
    for i in range(n):
        a, b = 41 + i, 31 + i
        messages.append(
            [
                {"role": "user", "content": f"What is {a} + {b}?"},
                {"role": "assistant", "content": str(a + b)},
            ]
        )
        metadata.append({"source": "synthetic-validation", "index": i})
    table = pa.table(
        {
            "messages": pa.array(messages),
            "metadata": pa.array(metadata),
        }
    )
    pq.write_table(table, path)


def _gsm8k_query_loader(*_args, **_kwargs) -> list[str]:
    """Real gsm8k test-set queries (question column of the local HF parquet)."""
    import pyarrow.parquet as pq

    parquet = REPO_ROOT / "public" / "datasets" / "gsm8k" / "main" / "test-00000-of-00001.parquet"
    if not parquet.is_file():
        raise ValueError(f"gsm8k test parquet not found: {parquet}")
    table = pq.read_table(parquet, columns=["question"])
    return [str(q) for q in table.column("question").to_pylist() if q]


def _setup_lf_run_copy(root: Path) -> None:
    """Create the run's public llama-factory data registry (dataset_info.json).

    与 playground.setup 的 public seed 一致: pipeline 强制把 dataset_dir 指到
    ``<base>/public/backend/llama_factory/llama_factory/data``, 训练只认这里的
    registry。harness 把合成数据注册在 ``identity`` 名下 (示例 YAML 的 dataset:).
    """
    data_dir = root / LF_DATA_REL
    data_dir.mkdir(parents=True, exist_ok=True)
    registry = {
        LF_DATASET: {
            "file_name": "smoke_train.jsonl",
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
            },
        }
    }
    (data_dir / "dataset_info.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


# ── llama_factory 路径: 真实 WorkerService.submit + render-only backend ─────


class _RenderOnlyBackend:
    """Real SubmitBackend contract but renders pipeline.sh instead of gpu_backend."""

    def __init__(self) -> None:
        self.submitted: list[dict] = []

    def submit(self, **kwargs):  # noqa: ANN003
        from expos.backends.models import SubmitRequest
        from expos.backends.pipelines.render import build_pipeline_script

        request = SubmitRequest(
            experiment_id=kwargs["experiment_id"],
            worker_id=kwargs["worker_id"],
            benchmark=kwargs["benchmark"],
            train_data_path=kwargs["train_data_path"],
            training_config_path=kwargs["training_config_path"],
            training_mode=kwargs["training_mode"],
            experiment_workspace=str(kwargs["experiment_workspace"]),
            bench_id=kwargs.get("bench_id"),
            eval_all=kwargs.get("eval_all") or {},
            gpu_nums=kwargs.get("gpu_nums") or 1,
            slime_config=kwargs.get("slime_config") or {},
        )
        script, paths = build_pipeline_script(request, Path(kwargs["experiment_workspace"]))
        self.submitted.append(kwargs)
        return {
            "status": "running",
            "trial_path": str(kwargs["experiment_workspace"]),
            **paths,
        }


def _render_llama_factory_case(root: Path, case: dict, limit: int, samples: int) -> dict:
    """Run the real submit-side flow (decontamination + render), return submit ret."""
    run_root = root / case["training_mode"]
    run_root.mkdir(parents=True, exist_ok=True)
    train_jsonl = run_root / "smoke_train.jsonl"
    _generate_alpaca_jsonl(train_jsonl, samples)
    _setup_lf_run_copy(run_root)
    train_yaml = Path(__file__).resolve().parent / case["train_yaml"]

    os_root = run_root / "experiment_os"
    os_ = ExperimentOS(os_root)
    os_.initialize(
        task_description=case["desc"],
        benchmark="gsm8k",
        eval_all={"bench_id": "gsm8k", "limit": limit},
    )
    worker = os_.register_worker(parent_worker_id=None, objective=case["desc"])
    svc = WorkerService(
        os_,
        submit_backend=_RenderOnlyBackend(),
        # 真实去污染路径: 直接读 gsm8k test parquet 的 question 列作为评测 query
        # (走 bench_id 分支的 load_benchmark_queries 需要 format/gsm8k/*.jsonl, 不存在)。
        # 合成数据是全新算术题, 不会与任何 query 精确相等 -> 实际不剔行, 但管线是真的。
        benchmark_query_loader=_gsm8k_query_loader,
    )
    ret = svc.submit(
        worker_id=worker.worker_id,
        data_path=str(train_jsonl),
        training_config_path=str(train_yaml),
        training_mode=case["training_mode"],
        gpu_nums=1,
    )
    if ret["status"] != "running":
        raise RuntimeError(f"submit failed: {ret.get('failure_reason')} {ret}")
    return ret


def _run_pipeline_on_host(host: str, ret: dict, gpu: str, timeout: int) -> None:
    workspace = Path(ret["workspace"])
    pipeline = Path(ret["pipeline_script_path"])
    if not pipeline.is_file():
        raise RuntimeError(f"pipeline.sh not rendered: {pipeline}")
    _run_remote(
        host,
        f'cd {shlex.quote(str(workspace))} && CUDA_VISIBLE_DEVICES={gpu} bash {shlex.quote(str(pipeline))}',
        timeout=timeout,
        label="pipeline.sh",
    )
    result_json = Path(ret["result_json_path"])
    result = json.loads(result_json.read_text(encoding="utf-8"))
    if result.get("status") != "completed":
        raise RuntimeError(
            f"result.json status={result.get('status')} (want completed): "
            f"{result.get('failure_reason')}"
        )


# ── slime 路径: launch_sft_smoke.sh (4B 稠密入口) ───────────────────────────


def _preflight_slime_node(host: str) -> None:
    """slime 冒烟前置只读检查 (无破坏).

    launch_sft_smoke.sh 只用 ``ray stop --force`` 清理自己的 ray 集群, **禁止 pkill
    无差别杀 python** (见 prompts_0812 公共 prompt 的「进程管理纪律」)。所以这里不
    告警, 只打印节点进程与 GPU 供人工观察。
    """
    rc, out, _err = _ssh(host, "ps -eo pid,etime,cmd | grep -E 'python' | grep -v grep | head -20", timeout=60)
    print("[INFO] 节点 python 进程 (只读观察; 需清理时先 ps 查 PID 再精确 kill):")
    if rc == 0 and out.strip():
        for line in out.splitlines():
            print(f"       {line}")
    else:
        print("       (无)")
    rc, out, _err = _ssh(host, "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader", timeout=60)
    print("[INFO] 节点 GPU:")
    if rc == 0 and out.strip():
        for line in out.splitlines():
            print(f"       {line}")
    else:
        print("       (nvidia-smi 不可用)")


def _convert_torch_dist(host: str, timeout: int = 5400) -> None:
    """One-time HF -> torch_dist conversion (约 10-30 分钟), 轮询完成标记."""
    print(f"[convert] HF -> torch_dist: {BASE_MODEL} -> {TORCH_DIST_DIR}")
    # env 赋值必须放在 `bash` 之前才生效; 放脚本名后面只是位置参数, 脚本会落到默认值。
    env_part = (
        f"MODEL_DIR={shlex.quote(str(BASE_MODEL))}"
        f" SAVE_DIR={shlex.quote(str(TORCH_DIST_DIR))}"
        f" MODEL_SPEC={shlex.quote(SLIME_MODEL_SPEC)}"
        f" WORKSPACE={shlex.quote(SLIME_WORKSPACE)} GPU_ID=0"
    )
    _run_remote(
        host,
        f"{env_part} bash {shlex.quote(str(SLIME_CONVERT))}",
        timeout=120,
        label="convert_hf_to_torch_dist (launch)",
    )
    marker = TORCH_DIST_DIR / "latest_checkpointed_iteration.txt"
    deadline = time.time() + timeout
    while time.time() < deadline:
        rc, out, _err = _ssh(host, f"test -f {shlex.quote(str(marker))} && echo YES || echo NO", timeout=60)
        if rc == 0 and "YES" in out:
            print(f"[convert] 完成: {marker}")
            return
        # 转换进程 (nohup python convert...) 是否还活着
        rc, out, _err = _ssh(
            host, "pgrep -f convert_hf_to_torch_dist >/dev/null && echo RUN || echo DONE", timeout=60
        )
        if rc == 0 and "DONE" in out and not (TORCH_DIST_DIR / "iter_0000001").exists():
            raise RuntimeError("convert_hf_to_torch_dist 提前退出且无 ckpt, 见 "
                               f"{SLIME_WORKSPACE}/training_scripts/logs/convert_hf_to_torch_dist.log")
        print(f"[convert] 转换中... ({int(deadline - time.time())}s 剩余)")
        time.sleep(30)
    raise TimeoutError(f"torch_dist conversion timed out after {timeout}s ({marker})")


def _run_slime_smoke(host: str, run_root: Path, slime_yaml: Path, gpu: str, timeout: int) -> None:
    cfg = load_slime_config_file(str(slime_yaml))
    if not TORCH_DIST_DIR.is_dir():
        raise RuntimeError(
            f"torch_dist weights not found: {TORCH_DIST_DIR}. Run with --convert first "
            "(或手动: MODEL_DIR=... SAVE_DIR=... MODEL_SPEC=... bash convert_hf_to_torch_dist.sh)"
        )
    data_file = run_root / "smoke_sft.parquet"
    _generate_slime_parquet(data_file, 8)
    ckpt_dir = run_root / "slime_ckpt"
    env_map = {
        "WORKSPACE": SLIME_WORKSPACE,
        "MODEL_SPEC": SLIME_MODEL_SPEC,
        "MODEL_DIR": str(BASE_MODEL),
        "TORCH_DIST_DIR": str(TORCH_DIST_DIR),
        "DATA_FILE": str(data_file),
        "CKPT_DIR": str(ckpt_dir),
        "NUM_GPUS": str(cfg.get("num_gpus") or "1"),
        "TP_SIZE": str(cfg.get("tp_size") or "1"),
        "EP_SIZE": str(cfg.get("ep_size") or "1"),
        "HAS_NVLINK": str(cfg.get("has_nvlink") or "0"),
    }
    env_str = " ".join(f"{k}={shlex.quote(v)}" for k, v in env_map.items())
    print(f"[slime] env: {env_str}")
    output = _run_remote(
        host,
        f"cd {shlex.quote(str(SLIME_SCRIPTS))} && CUDA_VISIBLE_DEVICES={gpu} {env_str} bash {shlex.quote(str(SLIME_SMOKE))}",
        timeout=timeout,
        label="launch_sft_smoke.sh",
    )
    if "SMOKE PASSED" not in output:
        raise RuntimeError(f"slime smoke did not pass:\n{output[-2000:]}")


# ── main ──────────────────────────────────────────────────────────────────────


def _resolve_cases(only: str) -> list[str]:
    if only == "all":
        return list(CASES)
    if only not in CASES:
        raise SystemExit(f"unknown --only {only!r}; choices: all, {', '.join(CASES)}")
    return [only]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", required=True, help="SSH host alias of the GPU machine to validate on")
    parser.add_argument("--only", default="all", help="all | slime-single | slime-multi")
    parser.add_argument("--dry-run", action="store_true", help="只渲染 pipeline.sh, 不 SSH / 不烧 GPU")
    parser.add_argument("--convert", action="store_true", help="slime 前置: 先跑 HF->torch_dist 转换")
    parser.add_argument("--limit", type=int, default=10, help="stage-3 eval_all 采样数 (llama_factory)")
    parser.add_argument("--samples", type=int, default=12, help="合成训练样本数 (llama_factory)")
    parser.add_argument("--timeout", type=int, default=3600, help="每个 GPU 任务的超时 (秒)")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="验证运行根目录 (默认 <repo>/runs/submit_docs_validation/<ts>, 已 gitignore)",
    )
    args = parser.parse_args(argv)

    cases = _resolve_cases(args.only)
    if args.dry_run:
        return _dry_run(cases, args)

    out_root = (
        Path(args.out_dir)
        if args.out_dir
        else REPO_ROOT / "runs" / "submit_docs_validation"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_root.mkdir(parents=True, exist_ok=True)

    # host 可达性 + slime 前置
    rc, out, err = _ssh(args.host, "hostname", timeout=60)
    if rc != 0:
        print(f"[FAIL] host unreachable {args.host}: {err}", file=sys.stderr)
        return 2
    print(f"[host] {args.host} -> {out.strip()}  out_root={out_root}")

    need_slime = any(c.startswith("slime") for c in cases)
    if need_slime and args.convert and not TORCH_DIST_DIR.is_dir():
        _preflight_slime_node(args.host)
        _convert_torch_dist(args.host)
    elif need_slime and not TORCH_DIST_DIR.is_dir():
        print("[WARN] slime 配置已选但 torch_dist 权重缺失; 用 --convert 先转换。跳过 slime 配置。")

    results: list[Result] = []
    for case_name in cases:
        case = CASES[case_name]
        res = Result(case_name, case["desc"])
        print(f"\n=== [{case_name}] {case['desc']} ===")
        t0 = time.time()
        try:
            if case["training_mode"].startswith("llama_factory"):
                if args.dry_run:  # unreachable; handled above
                    continue
                ret = _render_llama_factory_case(out_root, case, args.limit, args.samples)
                print(f"    渲染完成: workspace={ret['workspace']}")
                print(f"    pipeline: {ret['pipeline_script_path']}")
                _run_pipeline_on_host(args.host, ret, case["gpu"], args.timeout)
                result_json = Path(ret["result_json_path"])
                result = json.loads(result_json.read_text(encoding="utf-8"))
                res.finish(
                    True,
                    f"result.json status=completed, score={result.get('score')}, "
                    f"eval_status={result.get('eval_status')}",
                    time.time() - t0,
                )
            else:  # slime
                if not TORCH_DIST_DIR.is_dir():
                    res.finish(False, "torch_dist weights missing (run with --convert)", time.time() - t0)
                else:
                    _preflight_slime_node(args.host)
                    run_root = out_root / case_name
                    run_root.mkdir(parents=True, exist_ok=True)
                    _run_slime_smoke(
                        args.host,
                        run_root,
                        Path(__file__).resolve().parent / case["slime_yaml"],
                        case["gpu"],
                        args.timeout,
                    )
                    res.finish(True, "launch_sft_smoke.sh 完成 (SMOKE_PASSED)", time.time() - t0)
        except Exception as exc:  # noqa: BLE001 — 逐配置捕获, 汇总表格
            res.finish(False, str(exc)[:400], time.time() - t0)
            print(f"    [FAIL] {exc}")
        results.append(res)

    print("\n==== 验证汇总 ====")
    ok_count = 0
    for res in results:
        mark = "PASS" if res.ok else "FAIL"
        ok_count += 1 if res.ok else 0
        print(f"  [{mark}] {res.case:<12} {res.desc}")
        print(f"        {res.detail or '(无输出)'}  ({res.elapsed:.0f}s)")
    print(f"==== {ok_count}/{len(results)} PASS ====")
    return 0 if ok_count == len(results) else 1


def _dry_run(cases: list[str], args: argparse.Namespace) -> int:
    """本地自检: slime 配置解析 OK; llama_factory 提交必须被守卫 (c) 拦截."""
    print("[dry-run] 只渲染, 不 SSH / 不烧 GPU")
    ok = True
    root = (
        Path(args.out_dir)
        if args.out_dir
        else REPO_ROOT / "runs" / "submit_docs_validation" / "dry_run"
    )
    for case_name in cases:
        case = CASES[case_name]
        try:
            cfg = load_slime_config_file(str(Path(__file__).resolve().parent / case["slime_yaml"]))
            print(f"[{case_name}] slime 配置解析 OK: num_gpus={cfg.get('num_gpus')} "
                  f"tp={cfg.get('tp_size')} ep={cfg.get('ep_size')} "
                  f"torch_dist={'有' if TORCH_DIST_DIR.is_dir() else '缺(--convert)'}")
        except Exception as exc:  # noqa: BLE001
            print(f"[{case_name}] dry-run 失败: {exc}")
            ok = False

    # llama-factory 整体下线回归 (2026-08-27): 真实提交路径必须拒绝 LF 请求。
    lf_case = {
        "training_mode": "llama_factory_sft",
        "train_yaml": "configs/llama_factory_sft_qwen3_4b_full.yaml",
        "desc": "llama-factory (应被守卫 (c) 整体拦截)",
    }
    try:
        _render_llama_factory_case(root / "lf-blocked", lf_case, args.limit, args.samples)
        print("[llama-factory] 未拦截 —— 守卫 (c) 失效!")
        ok = False
    except RuntimeError as exc:
        msg = str(exc)
        good = "已整体下线" in msg
        print(f"[llama-factory] 拦截 {'OK' if good else '但消息不符'}: {msg[:200]}")
        ok = ok and good
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
