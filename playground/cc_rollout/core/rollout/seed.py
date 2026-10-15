"""Workspace seeding for the cc_rollout playground (both modes).

两种 rollout 方式的 initialize workspace (docs/README.md):

- **cc_naive** — backend copy + README.md / BENCHMARK.md / GLOBAL_MEMORY.md 注入,
  **不**建 experiment_os / SUBMIT_GUIDE.md / expos launcher。
- **cc_with_expos** — 同 naive, 再额外:
    * ``workspace/experiment_os/`` — ExperimentOS 根 (state.json / events.jsonl /
      worker / experiments / reviews / data_pool), 由 ``ExperimentOS.initialize``
      落盘 (benchmark + eval_all.bench_id 与 config 的 eval.bench_id 对齐);
    * ``workspace/SUBMIT_GUIDE.md`` — 面向 expos CLI 的 submit 训练管线指南;
    * ``workspace/expos`` — expos CLI launcher (PYTHONPATH 烘焙 repo root +
      tools/ExpOS, EXPOS_ROOT 指向 experiment_os)。

大部分 helper 逐字移植自 ``playground/cc_branching/core/rollout/orchestrator.py``
(``_seed_public_modules`` / launcher / ``_seed_benchmark_doc`` / ``_eval_contract`` /
``build_seed_task``), GLOBAL_MEMORY.md 是新写的 stub (cc_branching 无此文件)。
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import yaml

from . import backend_readmes, layout, permissions, promptlib

logger = logging.getLogger(__name__)

#: cc_with_expos 模式 = 写 CLI 形式 ExpOS (submit 走 serverless submit 训练管线)。
MODE_WITH_EXPOS = "cc_with_expos"
MODE_NAIVE = "cc_naive"


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------

def seed_workspace(run_root: Path, config: dict, mode: str, session_id: str) -> Path:
    """Initialize the workspace for a cc_rollout run; returns the workspace dir.

    *run_root* must already exist (orchestrator creates it).  ``configs/``
    (claude-invisible) is filled with the permission policy + run.json; the
    workspace gets the seeded docs + public/ backend modules.  ``mode`` selects
    whether the ExpOS CLI layer (experiment_os / SUBMIT_GUIDE / expos launcher)
    is also seeded.
    """
    mode = mode or MODE_NAIVE
    ws = layout.workspace_dir_for(run_root)
    ws.mkdir(parents=True, exist_ok=True)

    # Seeded docs (both modes).
    _seed_benchmark_doc(ws, config)
    seeded = _seed_public_modules(ws, config)
    backend_desc = backend_readmes.build_backend_descriptions(seeded, ws)
    protocol = "protocol_with_expos.md" if mode == MODE_WITH_EXPOS else "protocol_naive.md"
    (ws / "README.md").write_text(
        promptlib.compose(
            protocol,
            backend_desc,
            eval_contract=_eval_contract(config, mode),
            submit_driver=_submit_driver(mode),
            run_window=_run_window(config),
        )
    )
    _write_global_memory(ws, config)

    # Claude-INVISIBLE system state: permission policy + run.json + transcript dir.
    cfg_dir = layout.config_dir_for(ws)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    permissions.write_settings(cfg_dir, config.get("run", {}).get("public_dirs", []))
    (cfg_dir / ".claude-config").mkdir(parents=True, exist_ok=True)
    (cfg_dir / "run.json").write_text(json.dumps({
        "mode": mode,
        "session_id": session_id,
        "task": str(config.get("run", {}).get("task", "") or ""),
    }, indent=2, ensure_ascii=False))

    if mode == MODE_WITH_EXPOS:
        _seed_expos_layer(ws, config)
        logger.info("seeded cc_with_expos workspace %s (session %s)", ws, session_id)
    else:
        logger.info("seeded cc_naive workspace %s (session %s)", ws, session_id)
    return ws


def _seed_expos_layer(ws: Path, config: dict) -> None:
    """cc_with_expos extras: experiment_os state + SUBMIT_GUIDE.md + expos launcher."""
    from expos.service.experiment_os import ExperimentOS  # tools/ExpOS 在 PYTHONPATH
    from expos.backends.pipelines.recommendations import (
        effective_recommendations_cfg,
        resolve_run_defaults,
    )

    from ._paths import rsi_master_root

    bench_id = str((config.get("eval", {}) or {}).get("bench_id", "") or "").strip()
    eval_cfg = config.get("eval", {}) or {}
    eval_all = {"bench_id": bench_id} if bench_id else {}
    # 透传全部 eval_all knobs, 与 v2 的 evaluation.eval_all 对齐:
    #   limit / max_tokens / max_workers / serve_gpu_memory_utilization / serve_max_model_len
    # (max_tokens:0 与 max_workers:0 是有意义的显式值 — 0=bench default / auto, 必须保留)。
    _INT_KEYS = ("limit", "max_tokens", "max_workers")
    for key in _INT_KEYS + ("serve_gpu_memory_utilization", "serve_max_model_len"):
        val = eval_cfg.get(key)
        if val is None:
            continue
        eval_all[key] = int(val) if key in _INT_KEYS else val
    # split (如 livecodebench "recent_2025") —— benchmark 子集名, 逐字透传进
    # state.eval_all → render stage-3 组 ``--split <v>`` (render.py ea.get("split"))。
    if eval_cfg.get("split"):
        eval_all["split"] = str(eval_cfg["split"])
    # judge (grader) bench (healthbench_easy / arena_hard_writing ...) 需要凭据进
    # state.eval_all, 供 pipeline 渲染 export OPENAI_BASE_URL / OPENAI_API_KEY ——
    # 与 v2 整体透传 evaluation.eval_all 对齐。judge_env 显式 false 也保留
    # (覆盖 pipeline 的静态默认); judge 是 {base_url, api_key} dict。
    if eval_cfg.get("judge_env") is not None:
        eval_all["judge_env"] = bool(eval_cfg["judge_env"])
    if eval_cfg.get("judge"):
        eval_all["judge"] = eval_cfg["judge"]
    run_cfg = config.get("run", {}) or {}

    # 档位默认 (worker 无感, 2026-09-03): 与 v2 同一 resolve —— run.base_model 命中
    # 档位时 profile 尺寸键权威化, run.slime_config 只贡献 profile 未定义键, 末尾带
    # model_tier 使 worker 裸提交自动命中档位默认 + preflight。
    base_model = str(run_cfg.get("base_model") or "")
    seed_slime = resolve_run_defaults(
        base_model, run_cfg.get("slime_config") or {}, run_cfg.get("training_recommendations") or {}
    )
    run_contract = dict(run_cfg.get("run_contract") or {})
    if base_model:
        run_contract["base_model"] = base_model  # 与 v2 对齐: state 带模型身份

    os_root = ws / "experiment_os"
    init_kwargs: dict = {
        "task_description": str(run_cfg.get("task") or build_seed_task(config, mode=MODE_WITH_EXPOS)),
        "benchmark": bench_id or "gsm8k",
        "eval_all": eval_all,
        "run_contract": run_contract,
        # 始终传 (空 slime 也传 {}) —— ExperimentOS.initialize 内部 or {} 兜底
        "slime_config": seed_slime,
        "training_recommendations": effective_recommendations_cfg(
            run_cfg.get("training_recommendations") or {}
        ),
    }
    ExperimentOS(os_root).initialize(**init_kwargs)
    logger.info("seeded experiment_os at %s (benchmark=%s)", os_root, bench_id or "gsm8k")

    (ws / "SUBMIT_GUIDE.md").write_text(_SUBMIT_GUIDE_TEXT, encoding="utf-8")
    _seed_expos_launcher(ws, rsi_master_root(), os_root)
    _seed_cc_rollout_expos(ws)


# ---------------------------------------------------------------------------
# public seeds: register backend modules into the workspace (cc_branching port)
# ---------------------------------------------------------------------------

def _public_readme(seeded: list[str]) -> str:
    """Render ``public/README.md``.

    The ``pipeline-submit`` launcher bullet is only emitted when that module is
    actually seeded (with_expos); cc_naive 手搓基线不 seed 它, README 不指向
    不存在的 launcher。模块列表行保持通用 (backend/backend/* 里的模块)。
    """
    base = """# Public workspace

Registered backend modules, snapshotted from the repo's `backend/` for the agent.
**Read-only** — edit the authoritative copy at the project root, not here.

- `gpu_backend/`      — FastAPI submit client + config (from `backend/gpu_backend/`).
- `gpu-backend`       — PATH/pip 无关的 gpu-backend CLI launcher:
                       runs `python -m backend.gpu_backend` with the launch env's python.
- `backend/<module>/` — each enabled module from `backend/backend/*`
                       (eval_all / slime / search_cli / submit_pipeline / vllm ...).
"""
    if "submit_pipeline" in seeded:
        base += (
            "- `pipeline-submit`   — submit-train pipeline CLI launcher (提交即训练):\n"
            "                       `pipeline-submit submit --data <raw> --train-config <yaml> --bench-id <id>`.\n"
            "                       Experiments live in `experiments/exp_####/`.\n"
        )
    return base


def _seed_public_modules(workspace: Path, config: dict) -> list[str]:
    """Materialize the configured backend modules under ``<workspace>/public/``.

    移植自 cc_branching: ``gpu_backend`` 默认开, 每个 ``backend/backend/*`` 模块按
    ``workspace.public_seeds.backend_modules`` 单独开关, eval 系模块按
    ``eval.backend`` 漏斗过滤, 让 agent 只看到本次 run 真正用的 eval backend。
    """
    from ._paths import rsi_master_root

    root = rsi_master_root()
    backend_root = root / "backend"
    public = workspace / "public"
    public.mkdir(parents=True, exist_ok=True)

    seeds = config.get("workspace", {}).get("public_seeds", {}) or {}
    gpu_enabled = bool(seeds.get("gpu_backend", True))
    modules_cfg = seeds.get("backend_modules") or {}
    eval_backend = str(config.get("eval", {}).get("backend", "eval_all") or "eval_all").strip().lower()

    def copy_module(src: Path, dst: Path, label: str) -> bool:
        if not src.exists():
            logger.warning("public seed source missing: %s", label)
            return False
        if dst.exists():
            shutil.rmtree(dst)  # idempotent: drop stale copy first
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            src, dst,
            symlinks=False,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git", "*.log"),
        )
        return True

    seeded: list[str] = []
    if gpu_enabled and copy_module(backend_root / "gpu_backend", public / "gpu_backend", "gpu_backend"):
        seeded.append("gpu_backend")
        _seed_gpu_backend_launcher(public, root)

    for name, enabled in modules_cfg.items():
        if not enabled:
            continue
        # eval funnel: only seed the eval backend this run actually uses.
        if name.startswith("eval_") and name != eval_backend:
            logger.info("public seed %s skipped: eval.backend=%r", name, eval_backend)
            continue
        src = backend_root / "backend" / name
        dst = public / "backend" / name
        if copy_module(src, dst, f"backend/{name}"):
            seeded.append(name)

    if "submit_pipeline" in seeded:
        _seed_pipeline_submit_launcher(public, root)

    (public / "README.md").write_text(_public_readme(seeded), encoding="utf-8")
    logger.info("public seeds copied: %s", ", ".join(seeded) if seeded else "(none)")
    return seeded


def _seed_gpu_backend_launcher(public: Path, project_root: Path) -> Path:
    """Seed ``public/gpu-backend`` — a PATH/pip-independent gpu-backend CLI launcher.

    移植自 cc_branching。``gpu-backend`` 作为 console script 需要 pip install +
    PATH, agent 的 shell 通常没有; launcher 烘焙 project root 和启动环境 python
    (``sys.executable``), 跑 ``python -m backend.gpu_backend``。
    """
    launcher = public / "gpu-backend"
    python = sys.executable
    launcher.write_text(
        "#!/usr/bin/env bash\n"
        "# gpu-backend CLI launcher (seeded by cc_rollout, rsi_master pattern).\n"
        "# Uses the launch env's python (sys.executable) — PATH/pip independent.\n"
        f'export PYTHONPATH="{project_root}"${{PYTHONPATH:+:${{PYTHONPATH}}}}\n'
        f'exec "{python}" -m backend.gpu_backend "$@"\n',
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    logger.info("seeded public/gpu-backend launcher -> %s", launcher)
    return launcher


def _seed_pipeline_submit_launcher(public: Path, project_root: Path) -> Path:
    """Seed ``public/pipeline-submit`` — submit-pipeline CLI launcher.

    移植自 cc_branching。launcher 跑的是 **repo-root 活模块** (不是 public 下的
    拷贝) —— 拷贝供 agent 读文档/模板, 执行的是当前代码 (共享卷: seed 可能落后于树)。
    """
    launcher = public / "pipeline-submit"
    python = sys.executable
    launcher.write_text(
        "#!/usr/bin/env bash\n"
        "# submit-pipeline CLI launcher (seeded by cc_rollout, rsi_master pattern).\n"
        "# Uses the launch env's python (sys.executable) — PATH/pip independent.\n"
        "# Runs the repo-root module (not the seeded copy) so pipeline code stays current.\n"
        f'export PYTHONPATH="{project_root}"${{PYTHONPATH:+:${{PYTHONPATH}}}}\n'
        f'exec "{python}" -m backend.backend.submit_pipeline "$@"\n',
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    logger.info("seeded public/pipeline-submit launcher -> %s", launcher)
    return launcher


def _seed_expos_launcher(workspace: Path, project_root: Path, experiment_os_root: Path) -> Path:
    """Seed ``workspace/expos`` — the CLI-form ExpOS launcher (cc_with_expos).

    一个 CC 兼任 MainAgent/Worker/Reviewer 三角色, 通过 ``./expos`` CLI 调多角色
    工具 —— 与当前 ExpOS 相同的工具接口, 但 submit 走 serverless submit 训练管线
    (``cc_rollout_expos`` 注入 ``SubmitPipelineBackend``)。launcher 烘焙 PYTHONPATH
    (**workspace/public 最前** + repo root + tools/ExpOS) 和 EXPOS_ROOT, 跑
    ``python -m cc_rollout_expos``:
    - ``cc_rollout_expos`` 从 ``workspace/public/cc_rollout_expos/`` 解析 ——
      seed 复制进 workspace 的副本就是被执行的模块 (tools/ 不在 PYTHONPATH 上,
      避免歧义);
    - repo root 供 ``backend`` (SubmitPipelineBackend 延迟 import), tools/ExpOS
      供 ``expos`` (expos.cli.main)。
    """
    launcher = workspace / "expos"
    python = sys.executable
    public = workspace / "public"
    launcher.write_text(
        "#!/usr/bin/env bash\n"
        "# expos CLI launcher (cc_with_expos) — ExpOS 以 CLI 形式供单 CC 三角色调用.\n"
        "# PYTHONPATH: workspace/public (resolved cc_rollout_expos copy) + repo root\n"
        "# + tools/ExpOS; EXPOS_ROOT points at the seeded experiment_os. Submit flows\n"
        "# through the serverless submit pipeline.\n"
        f'export PYTHONPATH="{public}:{project_root}:{project_root}/tools/ExpOS"${{PYTHONPATH:+:${{PYTHONPATH}}}}\n'
        f'export EXPOS_ROOT="{experiment_os_root}"\n'
        f'exec "{python}" -m cc_rollout_expos "$@"\n',
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    logger.info("seeded workspace/expos launcher -> %s", launcher)
    return launcher


def _seed_cc_rollout_expos(ws: Path) -> Path:
    """Copy ``tools/cc_rollout_expos`` -> ``workspace/public/cc_rollout_expos/``.

    cc_with_expos 的 CLI 形式 ExpOS 入口是一个**顶层独立包** (tools/, 不依赖
    playground/), 在 workspace initialize 时复制进 public/ —— 与 gpu_backend /
    backend 模块同款只读拷贝。``workspace/expos`` launcher 的 PYTHONPATH 把
    workspace/public 放在最前, ``python -m cc_rollout_expos`` 解析执行的就是这
    份副本。
    """
    from ._paths import rsi_master_root

    src = rsi_master_root() / "tools" / "cc_rollout_expos"
    dst = ws / "public" / "cc_rollout_expos"
    if not src.is_dir():
        logger.warning("cc_rollout_expos source missing: %s", src)
        return dst
    if dst.exists():
        shutil.rmtree(dst)  # idempotent: drop stale copy first
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        src, dst,
        symlinks=False,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git", "*.log"),
    )
    logger.info("seeded public/cc_rollout_expos -> %s", dst)
    return dst


def _submit_driver(mode: str) -> str:
    """The mode-specific "how to drive the submit-train pipeline" block.

    注入 shared_context 的 ``{submit_driver}`` 占位符:
    - cc_naive: 手搓管线 (无 pipeline-submit launcher), 自写训练/serve/eval 脚本
      用 ``public/gpu-backend`` 驱动, experiments 在 workspace 根;
    - cc_with_expos: ``./expos worker submit`` (读 SUBMIT_GUIDE.md), experiments
      在 experiment_os/experiments/。
    """
    if mode == MODE_WITH_EXPOS:
        return _SUBMIT_DRIVER_WITH_EXPOS
    return _SUBMIT_DRIVER_NAIVE


_SUBMIT_DRIVER_NAIVE = """\
### Running experiments — drive training / serve / eval yourself

cc_naive has **no ready-made submit-train launcher** — you write the training,
serving and eval steps yourself and run them on the remote GPU machine. This is
deliberate: the baseline is your ability to assemble the pieces, not a provided
driver.

The framework modules are seeded under `public/backend/*` for you to read and
call: `slime/` (Megatron+Ray training, read its README + `launch_sft.sh`),
`eval_all/` (benchmark scoring), `vllm/` (serving), `search_cli/` (dataset
search). `public/gpu-backend` submits scripts to the GPU machine. Sketch:

    # 1) train — hand-written bash script (see public/backend/slime/), then:
    public/gpu-backend submit -p train.sh       # --gpu-nums / priority via flags
    # 2) serve the checkpoint (public/backend/vllm/) and 3) eval it
    #    (public/backend/eval_all/) — again hand-written scripts
    # 4) read the score + eval trajectories

Keep every experiment's artifacts under **`experiments/exp_####/`** in your
workspace (work_order / result / metrics / samples / logs) so a reviewer can
audit the chain. **Pass the run's eval contract exactly** — bench-id / limit /
serve args — see **Run Contract** below; a diverging flag breaks comparability.
"""


_SUBMIT_DRIVER_WITH_EXPOS = """\
### Running experiments — the submit-train pipeline through the expos CLI

Training, serving and eval are **one pipeline** — the framework owns data
preflight + decontamination (stage-0), training (stage-1), HF conversion if
needed (stage-1.5), vLLM serve (stage-2) and eval_all scoring (stage-3). You
drive it through the ExpOS DAG with `./expos worker submit` — **read
`SUBMIT_GUIDE.md` before your first submit** for the exact contract. Sketch:

    ./expos worker submit --worker-id <id> \\
        --data-path data/train_v2.jsonl \\
        --training-config-path config/train_v2.yaml \\
        --training-mode llama_factory_sft
    ./expos dump-state               # watch the DAG land the experiment
    cat experiment_os/experiments/exp_0002/result.json

Every experiment materializes under **`experiment_os/experiments/exp_####/`**:
`work_order.json` (what you submitted), `config.json` (resolved run
parameters), `pipeline.sh` (the exact script that ran), `result.json` (score +
eval status + failure reason), and the stage artifacts (decontaminated dataset,
ckpt, eval outputs). The eval contract (bench-id / limit / serve args) is
already wired into the DAG's state — do **not** re-specify it (see **Run
Contract** below).
"""


# ---------------------------------------------------------------------------
# seeded docs: BENCHMARK.md (eval spec) + GLOBAL_MEMORY.md (scratchpad)
# ---------------------------------------------------------------------------

def _bmk_root(config: dict) -> Path | None:
    """Resolve the benchmark root: ``eval.bmk_root`` > ``$BENCHMARK_ROOT`` > in-repo.

    Precedence: the run config, then the environment, then the ``bmk_everything/``
    checkout that ships alongside this repo. Point ``$BENCHMARK_ROOT`` at a
    different checkout to use one that lives elsewhere. If the resolved root has
    no ``format/<bench_id>/BENCHMARK.md``, a stub is seeded instead.
    """
    from ._paths import rsi_master_root

    override = str((config.get("eval", {}) or {}).get("bmk_root", "") or "").strip()
    if not override:
        override = os.environ.get("BENCHMARK_ROOT", "").strip()
    if not override:
        return (rsi_master_root() / "bmk_everything").resolve()
    return Path(override).expanduser().resolve()


def _seed_benchmark_doc(workspace: Path, config: dict) -> Path:
    """Seed ``BENCHMARK.md`` into the workspace — the eval's authoritative spec.

    移植自 cc_branching。Source: benchmark root ``format/<bench_id>/BENCHMARK.md``。
    幂等 —— 绝不覆盖已存在的 workspace 副本 (agent 可能追加了笔记)。缺源写 stub。
    """
    target = workspace / "BENCHMARK.md"
    if target.exists():
        return target
    bench_id = str((config.get("eval", {}) or {}).get("bench_id", "") or "")
    src = None
    bmk_root = _bmk_root(config)
    if bench_id and bmk_root is not None:
        cand = bmk_root / "format" / bench_id / "BENCHMARK.md"
        if cand.is_file():
            src = cand
    if src is None:
        target.write_text(
            f"# BENCHMARK: {bench_id or '(unspecified)'}\n\n"
            "(No authoritative BENCHMARK.md was found for this benchmark. See "
            "eval_all's bench config and `public/backend/eval_all/README.md` "
            "for the eval contract: prompt template, answer format, scoring.)\n",
            encoding="utf-8",
        )
        logger.warning("no BENCHMARK.md for bench %r; wrote stub", bench_id)
        return target
    target.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    logger.info("seeded %s <- %s", target, src)
    return target


def _write_global_memory(workspace: Path, config: dict) -> None:
    """Seed ``GLOBAL_MEMORY.md`` — a shared scratchpad (docs/README.md 明确要求注入).

    单 CC 的 rollout 里没有多个 agent, 但上下文可能被压缩, 这份自由格式 markdown
    是 CC 自己跨上下文刷新时的持久记忆: lessons / warnings / dead ends / open
    hypotheses。setup 只 *seed* 一个带书写指南的头 (若已存在绝不覆盖)。
    """
    memory_path = workspace / "GLOBAL_MEMORY.md"
    if memory_path.exists():
        return
    run_cfg = config.get("run", {}) or {}
    eval_cfg = config.get("eval", {}) or {}
    task = str(run_cfg.get("task") or "(from run.task)")
    bench = str(eval_cfg.get("bench_id") or "(unspecified)")
    base = str(run_cfg.get("base_model") or "")
    base_line = f"- **Base model**: `{base}`\n" if base else ""
    content = (
        "# GLOBAL_MEMORY.md\n\n"
        "Shared cross-agent scratchpad for this cc_rollout run.\n\n"
        f"- **Task**: {task}\n"
        f"- **Benchmark**: `{bench}`\n"
        f"{base_line}"
        "- **Location**: `./GLOBAL_MEMORY.md` (workspace root). Append with:\n"
        "  `cat >> GLOBAL_MEMORY.md <<'EOF' ... EOF`\n\n"
        "## What belongs here\n\n"
        "Free-form notes that outlive a single context refresh and are useful "
        "later in the run:\n"
        "- Lessons: \"benchmark X's grader rejects trailing whitespace — strip outputs before submit\".\n"
        "- Warnings: \"data source Y contains test-set near-duplicates, do not use as-is\".\n"
        "- Dead ends: \"tried LoRA rank=64 on aime2025 → +0.2 vs baseline at 3x cost; skip\".\n"
        "- Open hypotheses: considered but not yet tested, so a later refresh can pick them up.\n\n"
        "## Sections\n\n"
        "### Lessons\n"
        "_None yet._\n\n"
        "### Warnings\n"
        "_None yet._\n\n"
        "### Dead ends\n"
        "_None yet._\n\n"
        "### Open hypotheses\n"
        "_None yet._\n"
    )
    memory_path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# eval contract + seed task (mode-aware; cc_branching port + with_expos variant)
# ---------------------------------------------------------------------------

def _eval_contract(config: dict, mode: str = MODE_NAIVE) -> str:
    """Render the run's eval contract.

    这些是 run 级常量 (bench_id / limit / serve args), 不是实验变量;
    ``promptlib.compose`` 把它注入 shared context 的 Run Contract 段。

    - cc_naive: 手搓管线 —— agent 自写训练/serve/eval 脚本, 需要把这些 flags
      原样传进自己的脚本 (``--bench-id`` 等)。
    - cc_with_expos: ``expos worker submit`` 从 experiment_os 的 state 读
      ``bench_id`` (seed 已写入), agent 不需要传 flags —— 这里只列出 run 常量
      说明"评测身份已在 experiment_os 里配好, 别去改它"。
    """
    eval_cfg = config.get("eval", {}) or {}
    bench_id = str(eval_cfg.get("bench_id", "") or "").strip()
    if not bench_id:
        return ""
    limit = int(eval_cfg.get("limit") or 500)
    # split (如 livecodebench "recent_2025") —— benchmark 子集; naive 需逐字传
    # ``--split``, with_expos 已由 _seed_expos_layer 写进 state.eval_all。
    split = str(eval_cfg.get("split") or "").strip()
    serve_util = eval_cfg.get("serve_gpu_memory_utilization", 0.6)
    # serve_max_model_len 是 v2 命名 (submit_pipeline 消费); 旧 max_model_len 兼容回退。
    max_model_len = int(eval_cfg.get("serve_max_model_len") or eval_cfg.get("max_model_len") or 32768)
    max_tokens = int(eval_cfg.get("max_tokens") or 0)
    max_workers = int(eval_cfg.get("max_workers") or 0)
    bmk_root = _bmk_root(config)
    bmk_root_text = str(bmk_root) if bmk_root is not None else "(unset — set eval.bmk_root or $BENCHMARK_ROOT)"
    if mode == MODE_WITH_EXPOS:
        return "\n".join([
            f"- benchmark: ``{bench_id}``  (benchmark root ``format/{bench_id}/``)",
            f"- limit: ``{limit}`` samples scored per eval",
            *([f"- split: ``{split}``  (benchmark subset — already in state.eval_all, "
               "render adds `--split`)"] if split else []),
            f"- serve: ``--serve-util {serve_util}`` ``--max-model-len {max_model_len}`` "
            f"``--max-tokens {max_tokens}`` ``--max-workers {max_workers}``",
            f"- bmk_root: ``{bmk_root_text}``",
            "",
            "These are already wired into the seeded experiment_os state "
            "(``state.json → eval_all``). `expos worker submit` reads them — do "
            "**not** re-specify or change them mid-run; they are run parameters, "
            "not experiment variables.",
        ])
    return "\n".join([
        f"- ``--bench-id {bench_id}``   (benchmark root ``format/{bench_id}/``)",
        *([f"- ``--split {split}``         (benchmark subset)"] if split else []),
        f"- ``--limit {limit}``         (samples scored per eval)",
        f"- ``--serve-util {serve_util}`` ``--max-model-len {max_model_len}`` "
        f"``--max-tokens {max_tokens}`` ``--max-workers {max_workers}``   (vLLM serve + eval args)",
        f"- ``--bmk-root {bmk_root_text}``",
        "",
        "Source: this run's config ``eval.bench_id / eval.split / eval.limit / "
        "eval.serve_gpu_memory_utilization / eval.serve_max_model_len`` — pass them "
        "verbatim into your own training / serve / eval scripts (naive 手搓管线).",
    ])


def _run_window(config: dict, start: datetime.datetime | None = None) -> str:
    """Render the run's wall-clock window (start / end / budget) for the README.

    The window is anchored on the **seed moment** (the agent starts a few seconds
    later) and the config's ``run.single_agent_running_time_seconds`` budget —
    the same budget the runner enforces as its kill timeout.  ``max_running_time_seconds``
    (global deadline) is applied as a cap when set.

    All timestamps are rendered **in UTC and Asia/Shanghai (+08:00)** — the same
    instant in two zones — so the agent can pick one zone and stay in it without
    mixing clock domains (a known failure mode on cluster pods).  ``date -u`` gives
    UTC; ``date`` gives the machine's local zone.
    """
    run_cfg = config.get("run", {}) or {}
    budget = float(run_cfg.get("single_agent_running_time_seconds", 120))
    cap_raw = run_cfg.get("max_running_time_seconds")
    cap = float(cap_raw) if cap_raw else None
    if cap is not None:
        budget = min(budget, cap)
    if budget <= 0:
        return "(run window: budget <= 0 — no fixed wall-clock end.)"

    if start is None:
        start = datetime.datetime.now(datetime.timezone.utc)
    elif start.tzinfo is None:
        start = start.replace(tzinfo=datetime.timezone.utc)
    end = start + datetime.timedelta(seconds=budget)

    utc = datetime.timezone.utc
    sh = datetime.timezone(datetime.timedelta(hours=8))  # Asia/Shanghai, +08:00 fixed

    def _fmt(dt: datetime.datetime, tz: datetime.tzinfo) -> str:
        # no %Z: emit "YYYY-MM-DD HH:MM:SS" then label the zone explicitly so the
        # string reads ".. 2026-09-07 08:00:00 (+08:00)" — no doubled zone text.
        return dt.astimezone(tz).strftime("%Y-%m-%d %H:%M:%S")

    hours = budget / 3600.0
    hh = int(hours)
    mm = int(round((hours - hh) * 60))
    budget_str = f"{hh}h{mm:02d}m" if hh else f"{int(budget)}s"
    return "\n".join([
        f"- **start**: `{_fmt(start, utc)} UTC`  ==  `{_fmt(start, sh)} (+08:00)`",
        f"- **end**  : `{_fmt(end, utc)} UTC`  ==  `{_fmt(end, sh)} (+08:00)`",
        f"- **budget**: {budget:.0f}s ({budget_str}) of agent wall-clock time",
    ])


def build_seed_task(config: dict, mode: str = MODE_NAIVE) -> str:
    """Assemble the seed agent's init prompt from run config.

    移植自 cc_branching, 增加 GLOBAL_MEMORY.md (以及 with_expos 的 expos CLI)
    到 Begin 指引。若 config 里仍有 ``run.task`` 就追加为 override。
    """
    run_cfg = config.get("run", {}) or {}
    eval_cfg = config.get("eval", {}) or {}
    bench = str(eval_cfg.get("bench_id", "") or "(unspecified)")
    base = str(run_cfg.get("base_model", "") or "(unspecified)")
    budget = int(run_cfg.get("single_agent_running_time_seconds", 120))
    run_id = str(run_cfg.get("run_id") or run_cfg.get("name") or "untitled")
    lines = [
        f"Post-training run **{run_id}**: improve the base model's score on benchmark **{bench}**.",
        "",
        "Run parameters:",
        f"- benchmark: {bench} — read `BENCHMARK.md` in your workspace: it defines "
        "what is scored and the exact answer format the grader parses.",
        f"- base_model: `{base}`",
        f"- eval: eval_all (`bench_id={bench}`)",
        f"- time budget: {budget}s — see your protocol for how to use it",
        "- all your work lives in this workspace (your current directory).",
        "",
        "Begin: read `README.md` (your protocol), `BENCHMARK.md`, "
        "`public/README.md`, `GLOBAL_MEMORY.md`, and the seeded backend module "
        "READMEs. Write your framing + first hypothesis into `AGENT.md`, then "
        "start the research loop.",
    ]
    if mode == MODE_WITH_EXPOS:
        lines.insert(len(lines) - 1, "You operate through the `./expos` CLI (see "
                                     "SUBMIT_GUIDE.md for the submit workflow).")
    task = run_cfg.get("task")
    if task:
        lines += ["", str(task)]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SUBMIT_GUIDE.md (cc_with_expos) — 面向 expos CLI 的 submit 训练管线指南
# ---------------------------------------------------------------------------

_SUBMIT_GUIDE_TEXT = """# SUBMIT_GUIDE — submitting training through the expos CLI (cc_with_expos)

This run drives the **submit training pipeline** end-to-end. One Claude-Code
session plays **MainAgent / Worker / Reviewer** by issuing different `./expos`
verbs — the ExperimentOS root lives at `./experiment_os` and is the single
source of truth for the DAG (workers / experiments / reviews / data_pool).
Submit flows through the **serverless submit pipeline**: `./expos worker submit`
delegates straight to the repo's `backend/backend/submit_pipeline` (`run_trial`)
→ renders `pipeline.sh` → hands it to `gpu_backend` (localhost:8989). No
ExpOS HTTP server is involved; everything here is plain filesystem + one async
submit.

## 1. One config channel, interpreted per mode

`training_config_path` is the only hyper-parameter channel.

| training_mode | what training_config_path is | notes |
|---|---|---|
| `llama_factory_sft` | a LLaMA-Factory training YAML | must set `finetuning_type: full` (LoRA 已下线) |
| `slime_sft` | a slime config file (YAML/JSON) | optional inline `--slime-config <json>` merges on top |

## 2. What the framework does when you `worker submit`

`./expos worker submit --worker-id <id> --data-path <data> --training-config-path <yaml> --training-mode llama_factory_sft`

1. validates the worker exists in state (create it first with
   `./expos state add-new-worker --objective "..."`);
2. resolves absolute paths, checks the run contract (`allowed_training_modes`);
3. **decontaminates** your data against the benchmark's queries — the backend
   only ever sees the clean copy in `experiments/<id>/artifacts/`;
4. writes `experiments/<id>/work_order.json` + a placeholder `result.json`;
5. renders `pipeline.sh` and submits it to gpu_backend → returns
   `status=running` + a `gpu_script_id`;
6. you poll with `public/gpu-backend script-status --script-id <id>` /
   `public/gpu-backend script-log --script-id <id> --tail 200`; the pipeline's
   EXIT trap overwrites `experiments/<id>/result.json` with the real score.

## 3. Training data: prepare + validate

- Input formats: `llama_factory_sft` → a `.jsonl` (conversation / instruction
  format accepted by LLaMA-Factory); `slime_sft` → a parquet with a `messages`
  column.
- The framework runs stage-0 preflight + decontamination (exact-match to the
  benchmark's eval queries; near-duplicates also flagged). The clean copy is
  what actually trains — never put eval/test questions in your raw data.
- Sample 5–10 rows and judge them against `BENCHMARK.md` before spending GPU
  (train-inference consistency: the eval's answer format must be what your
  training rows teach).

## 4. Hyper-parameters per mode

- **llama_factory_sft**: your YAML must set the framework-owned keys —
  `output_dir` / `report_to` / `dataset_dir` are overridden by the pipeline
  (your ckpt + log land under `experiments/<id>/`). `deepspeed:` declared →
  `FORCE_TORCHRUN=1` + `NPROC_PER_NODE=<gpu_nums>` injected (real ZeRO
  distributed training). Size the run to your time budget — a checkpoint that
  actually lands beats the "ideal" epoch count.
- **slime_sft**: keys map to `launch_sft.sh` env (`model_dir` / `torch_dist_dir`
  required). Passing a `--slime-config` JSON on top of the file is allowed;
  overriding framework-managed defaults yields `slime_config_warnings` in the
  return dict (informational, not blocking).

## 5. After the trial finishes — self-report

Read `experiments/<id>/result.json` (score + eval_status). Then:
1. append the outcome + diagnosis to `AGENT.md` and a durable lesson/warning to
   `GLOBAL_MEMORY.md`;
2. hang a review on the experiment:
   `./expos worker add-reviewer --worker-id <id> --experiment-ids <eid> --objective "..."`;
3. write the review yourself as the reviewer role:
   `./expos reviewer inspect-experiment --experiment-id <eid>` →
   `./expos reviewer submit-review --reviewer-id <rid> --experiment-id <eid> --report <file.json>` →
   `./expos reviewer finalize --reviewer-id <rid>`;
4. when your analysis is complete, `echo <experiment_id> > FINISH`.

## 5b. Parallel submissions, early signal, and data at scale

`worker submit` is asynchronous and gpu_backend **queues** jobs, allocating free
cards as they appear — you are not limited to one running experiment at a time.
Submit 2+ independent experiments in parallel whenever the node has free GPUs (a
baseline/cheap run beside the main run; two ablations differing in one factor).
Submitting ahead of a free slot is safe (it just queues); keep total demand
(`gpu_nums` per job) within the node's cards. Do **not** `sleep` a long block to
watch one pipeline — use that time to prepare the next experiment, read a
completed run's trajectories, or mine data.

A `worker submit` scores only the run's *final* checkpoint. While a long
training writes checkpoints into `experiments/<id>/`, you may score the ones it
has already saved yourself on a free GPU: lift the convert → serve → eval tail
from a finished experiment's `pipeline.sh` (or `public/backend/`) into a small
script and `public/gpu-backend submit -g 1` it. A mid-training checkpoint can
show a broken recipe (no-code / length-cap explosion, format drift) hours before
the full run lands — log such informal reads in `AGENT.md` as *mid-training
signal*; they are not DAG experiment scores (a DAG experiment is still a full
`worker submit`).

Source training data at scale: build one big clean pool (tens of thousands of
rows, target content + eval format) from the largest available corpus in a
single pass, then slice per-experiment subsets from it — data is cheap to
produce once, and it keeps the GPU from ever waiting on data.

## 6. Hard rules

- **No LoRA** — only `llama_factory_sft` (with `finetuning_type: full`) or
  `slime_sft`.
- **Model tier defaults auto-apply from the run's base_model** — you never
  declare `model_tier`. 35B / MoE: `slime_sft` only, `--gpu-nums 4` (default)
  or `2` (**never 8**), `tp×ep×cp == gpu_nums`. 4B: default `--gpu-nums 2`
  (TP2); explicit `1` is allowed. If you lower `seq_length`, make sure it does
  not truncate your longest training rows.
- **Never set `CUDA_VISIBLE_DEVICES`** anywhere (YAML or shell) — gpu_backend's
  scheduler picks the card and injects it.
- **Never hand-edit `experiment_os/state/state.json`** — every structural change
  goes through `./expos` verbs (they append to `state/events.jsonl` too).
- **Never put eval/test questions into training data** (leakage).
- Don't blind-resubmit the *same* experiment while its pipeline is still
  running — read `experiments/<id>/result.json` first. **Different** experiments
  running in parallel is expected and encouraged (see §5b).
"""


__all__ = [
    "MODE_NAIVE",
    "MODE_WITH_EXPOS",
    "seed_workspace",
    "build_seed_task",
]
