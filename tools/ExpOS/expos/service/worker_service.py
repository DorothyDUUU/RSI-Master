"""Worker-facing service — CLI backbone for a single worker's tools.

对应 redesign.md 中 Worker 的工具:
    worker.show_datapool()
    worker.inspect_data(data_id)
    worker.update_readme(data_id, update_summary)
    worker.add_data(data_id, ...)
    worker.submit(data_path, training_config_path, training_mode)
    worker.add_reviewer(experiment_ids)

其中 :meth:`submit` 是**最核心的工具组件**, 这里预留清晰的契约但把真正的训练流水
线交给可插拔的 backend (:class:`SubmitBackend`), 从而实现 Service / Backend 解耦。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Protocol

import yaml

from .contracts import (
    ArtifactRecord,
    ExperimentNode,
    ExperimentResult,
    ReviewerNode,
    WorkOrder,
    dump_model,
)
from .data_pool import DataPoolService
from .decontamination import (
    decontaminate_jsonl,
    decontaminate_parquet,
    load_benchmark_queries,
)
from .experiment_os import ExperimentOS
from .io_utils import ensure_dir, read_json, sha256_of_file, write_json
from ..backends.pipelines.recommendations import (
    PreflightContext,
    apply_preflight,
    effective_recommendations,
    resolve_profile,
)


#: slime 配置里由框架托管默认值的键。worker 用 ``training_config_path`` (slime_sft
#: 下按配置文件解析) / 内联 ``slime_config`` 覆盖这些键时, 返回里会带 warning,
#: 但**不阻止**覆盖 (worker
#: 对训练方向有最终决定权)。render 侧的真实默认值见 ``backends/pipelines/render.py``。
_SLIME_MANAGED_DEFAULT_KEYS = ("model_dir", "torch_dist_dir", "output_dir", "ckpt_dir")

#: output_dir / ckpt_dir 在 run 默认缺失时的隐式默认 (render 侧落点), 只用于把
#: 告警写清楚, 不是实际值。
_SLIME_IMPLICIT_DEFAULT_LABELS = {"output_dir": "workspace", "ckpt_dir": "workspace/checkpoints"}


def load_slime_config_file(path: str) -> dict[str, Any]:
    """Parse an agent-supplied slime config file (YAML or JSON) into a dict.

    worker_submit 工具与 ``worker submit`` 共用这一个解析器 (两者都把
    ``training_config_path`` 作为 slime_sft 的配置文件传入), 避免两处实现漂移。
    文件缺失 / 解析失败 / 不是 mapping 时抛 ``ValueError`` (上层把它变成对 agent
    友好的错误返回, 不是裸 traceback)。
    """
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise ValueError(f"slime config file does not exist: {config_path}")
    try:
        parsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 — 统一成对 agent 友好的错误
        raise ValueError(
            f"slime config file is not valid YAML/JSON: {config_path}: {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"slime config file must contain a mapping: {config_path}")
    return {str(k): v for k, v in parsed.items() if v is not None}


def slime_override_warnings(
    merged: dict[str, Any], run_default: dict[str, Any]
) -> list[str]:
    """框架托管默认键被 worker 配置覆盖时的一条条 warning (只提示, 不阻止).

    ``run_default`` 是 run 级默认 (``state["slime_config"]``); ``merged`` 是
    state 默认 + 配置文件 + 内联 JSON 合并后的**实际生效**配置。``output_dir`` /
    ``ckpt_dir`` 在 run 默认缺失时视为 workspace 的隐式默认 (render 侧落到
    workspace / workspace/checkpoints), 只要被显式设置就算覆盖; ``model_dir`` /
    ``torch_dist_dir`` 在 run 没给默认时是 worker 在填必填项, 不算覆盖、不提示。
    """
    warnings: list[str] = []
    defaults = dict(run_default or {})
    defaults.setdefault("output_dir", "")  # render 默认 -> workspace
    defaults.setdefault("ckpt_dir", "")  # render 默认 -> workspace/checkpoints
    for key in _SLIME_MANAGED_DEFAULT_KEYS:
        value = merged.get(key)
        if value in (None, ""):
            continue
        base = defaults.get(key)
        if base is None:
            continue
        if str(base) != str(value):
            base_label = str(base) if base else _SLIME_IMPLICIT_DEFAULT_LABELS.get(key, "未设置")
            warnings.append(
                f"slime_config 覆盖了默认 {key}={value} (默认 {base_label}); "
                "框架允许覆盖, 但请确认这是有意的。"
            )
    return warnings


class SubmitBackend(Protocol):
    """Contract for the training + evaluation backend.

    真正的实现放在 :mod:`expos.backends.pipeline_llama_factory_gsm8k`, 这里只声明
    契约, 便于在测试/CI 中注入 stub。当前 pipeline 是**异步**的: backend 组装
    并提交 pipeline.sh 到 gpu_backend 之后立即返回 ``status="running"`` 以及
    对应的 ``gpu_script_id`` / ``pipeline_log_path``, 真正的 metrics 由 pipeline
    bash 收尾时写进 workspace/result.json。
    """

    def submit(
        self,
        *,
        experiment_id: str,
        worker_id: str,
        benchmark: str,
        train_data_path: str,
        training_config_path: str,
        training_mode: str,
        experiment_workspace: Path,
        objective: str = "",
        hypothesis: str = "",
        changed_variable: str = "",
        parent_experiment_id: str | None = None,
        bench_id: str | None = None,
        eval_all: dict[str, Any] | None = None,
        gpu_nums: int | None = None,  # WorkerService 解析后必传 int; None 容忍 (or 1)
        slime_config: dict[str, Any] | None = None,
        allowed_training_modes: list[str] | None = None,
    ) -> dict[str, Any]:
        """Kick off one training + evaluation trial (async).

        Must return a JSON-safe dict with at least ``status``. ``status="running"``
        means the pipeline was successfully queued on gpu_backend; ``status="failed"``
        means submission itself failed (should include ``failure_reason``). ``score``
        will typically be ``None`` at this point and is filled in later by
        ``result.json`` refresh.

        ``bench_id`` / ``eval_all``: 非空时 stage-3 走 benchmark suite 统一评测
        (eval_all backend)。可选, 便于 stub 后端忽略。

        ``slime_config``: ``training_mode="slime_sft"`` 时的训练配置, 键映射到
        ``launch_sft.sh`` 的 env (model_dir / torch_dist_dir 必填)。可选, 便于
        stub 后端忽略; llama_factory 路径不读它。
        """


class WorkerService:
    """Bash-callable service scoped to a single worker."""

    def __init__(
        self,
        os_: ExperimentOS,
        *,
        submit_backend: SubmitBackend | None = None,
        data_pool: DataPoolService | None = None,
        benchmark_query_loader: Callable[..., list[Any]] = load_benchmark_queries,
    ):
        self.os = os_
        self.submit_backend = submit_backend
        self.data_pool = data_pool or DataPoolService(os_)
        self.benchmark_query_loader = benchmark_query_loader

    # ── Data pool passthrough ────────────────────────────────────────────────

    def queue_status(self) -> dict[str, Any]:
        """Live, run-scoped snapshot; unavailable is never interpreted as empty."""
        if not callable(getattr(self.submit_backend, "get_queue", None)):
            return {"available": False, "reason": "backend has no queue API"}
        try:
            snapshot = self.submit_backend.get_queue()
        except Exception as exc:
            return {"available": False, "reason": str(exc)}
        prefix = f"{self.os.root.resolve()}::"
        graph = self.os.load_graph()
        script_to_exp = {}
        for eid, node in graph.experiments.items():
            result = read_json(Path(node.workspace) / "result.json", default={}) or {}
            sid = (result.get("submit_result") or {}).get("gpu_script_id")
            if sid:
                script_to_exp[sid] = eid
        scoped = {"available": True, "policy": snapshot.get("policy"),
                  "max_queued_per_owner": snapshot.get("max_queued_per_owner", 0)}
        for key in ("queued", "running"):
            scoped[key] = [dict(row, experiment_id=script_to_exp.get(row["script_id"]),
                                worker_id=row["queue_owner"][len(prefix):])
                           for row in snapshot.get(key, [])
                           if str(row.get("queue_owner") or "").startswith(prefix)]
        scoped["backend_totals"] = {key: len(snapshot.get(key, [])) for key in ("queued", "running")}
        scoped["hint"] = "Empty queue is not proof of idle GPUs. Prepare independent small tests; when queue is sufficient, analyze/review."
        return scoped

    def cancel_queued_experiment(self, experiment_id: str, reason: str) -> dict[str, Any]:
        """Only cancel an experiment owned by this run; never kill running work."""
        if not reason.strip():
            raise ValueError("A withdrawal reason is required")
        node = self.os.load_graph().experiments.get(experiment_id)
        if node is None:
            raise KeyError(experiment_id)
        path = Path(node.workspace) / "result.json"
        result = read_json(path, default={}) or {}
        sid = (result.get("submit_result") or {}).get("gpu_script_id")
        if not sid:
            return {"cancelled": False, "reason": "no GPU script receipt"}
        receipt = self.submit_backend.cancel_queued_script(sid)
        if receipt.get("cancelled"):
            result.update(status="aborted", eval_status="skipped", score=None,
                          notes=f"Withdrawn before start: {reason}")
            write_json(path, result)
            self.os.update_experiment(experiment_id, status="aborted", score=None)
            self.os.events.emit("experiment_queue_withdrawn", experiment_id=experiment_id,
                                worker_id=node.worker_id, payload={"reason": reason, "gpu_script_id": sid})
        return dict(receipt, experiment_id=experiment_id)

    def show_datapool(self) -> list[dict[str, Any]]:
        return self.data_pool.list_data()

    def inspect_data(self, data_id: str) -> dict[str, Any]:
        return self.data_pool.inspect_data(data_id)

    def update_readme(self, data_id: str, update_summary: str) -> dict[str, Any]:
        return self.data_pool.update_readme(data_id, update_summary)

    def add_data(self, **kwargs: Any) -> dict[str, Any]:
        return self.data_pool.add_data(**kwargs)

    # ── Core: submit ─────────────────────────────────────────────────────────

    def submit(
        self,
        *,
        worker_id: str,
        data_path: str,
        training_config_path: str,
        training_mode: str,  # 必填 —— 不设默认, 防静默走 llama_factory_sft
        objective: str = "",
        hypothesis: str = "",
        changed_variable: str = "",
        parent_experiment_id: str | None = None,
        # 2026-09-03 显式性重构: 默认 None, 让"没传卡数"与"显式 1"可区分 —— 没传则按
        # 命中档位的 default_config.gpu_nums 回填 (4B=2 卡); 显式传值仍逐位校验。
        gpu_nums: int | None = None,
        need_exclusive_gpu: bool = True,  # 默认独占 (2026-08-29): 整卡到会话结束
        slime_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Submit one async training+eval pipeline and register the experiment.

        流程 (纯异步):
            1. 分配 experiment_id + 创建 experiments/{experiment_id}/;
            2. 落盘 WorkOrder (experiments/{...}/work_order.json);
            3. 委托 SubmitBackend.submit -> pipeline 提交到 gpu_backend, 立即
               返回 ``status="running"`` 以及 SCRIPT_ID / SESSION_ID / log_path;
            4. 落盘一份"占位版" ExperimentResult (status=running, score=None,
               submit_result 里带全部 gpu_backend 元数据), 之后 pipeline bash
               会自行覆盖它;
            5. 更新 ExperimentNode.status="running" + submit_trial_path;
            6. 返回给 tool 层, 由 tool 层拼出面向 agent 的 hint (让它用
               `gpu-backend script-status/script-log` 自行观察).

        ``training_mode="slime_sft"`` 时, 污染检查走 parquet 分支
        (``decontaminate_parquet`` -> ``artifacts/decontaminated_train.parquet``),
        并把 ``slime_config.data_file`` 强制指到净化副本再透传给 backend。
        ``training_config_path`` 一条通道按 mode 解释: llama_factory 模式当训练 YAML
        原样透传, slime_sft 当 slime 配置文件 (YAML/JSON) 解析, 再与内联
        ``slime_config`` 一起叠到 run 默认 ``state["slime_config"]`` 之上; 覆盖框架
        托管默认键 (model_dir / torch_dist_dir / output_dir / ckpt_dir) 时, 返回
        dict 带 ``slime_config_warnings``, 但不阻止。

        NOTE: 我们**不再**为 ``status="pending"`` 做 dry-run 兜底。没有配 backend
        直接算失败, 让 agent 立刻感知。
        """
        # 校验 worker 存在
        graph = self.os.load_graph()
        worker = graph.workers.get(worker_id)
        if worker is None:
            raise KeyError(f"unknown worker_id: {worker_id}")

        from .research_context import ResearchContext
        refresh = ResearchContext(self.os).before_submit(worker_id)
        if refresh is not None:
            return refresh

        queue = self.queue_status()
        limit = queue.get("max_queued_per_owner", 0)
        if queue.get("available") and limit and sum(r["worker_id"] == worker_id for r in queue["queued"]) >= limit:
            return {"status": "deferred", "reason": "queue_full", "queue": queue,
                    "hint": "No experiment allocated. Prepare/analyze instead of spin-retrying; submit after capacity frees."}

        # 路径绝对化 (本次 e2e 测试排掉的坑): pipeline 在 LLAMA_FACTORY_CWD
        # 下执行 llamafactory-cli, 相对路径会相对那个目录解析 -> stage-1 直接
        # 找不到 train config / data。这里以调用进程 CWD 为基准 resolve 两个
        # 用户传入路径, CLI 与 agent tool 都汇入本方法, 一处修复两边生效。
        data_path = str(Path(data_path).expanduser().resolve())
        training_config_path = str(Path(training_config_path).expanduser().resolve())

        # 守卫 (a): llama_factory_lora_sft 已下线 —— 兜底拦截绕过模型层 (CLI
        # argparse choices / SubmitRequest validator) 的程序化调用方。硬拒绝在
        # 服务层, 不在 preflight 脚本里做 (pre-check 保持宽松)。
        if training_mode == "llama_factory_lora_sft":
            raise ValueError(
                "`llama_factory_lora_sft` 已下线 (2026-08-20, LoRA 停用): "
                "训练只能走全参 SFT —— `training_mode=llama_factory_sft` + "
                "训练 YAML `finetuning_type: full`"
            )

        state = self.os.load_state()
        contract = state.get("run_contract") or {}
        allowed_modes = contract.get("allowed_training_modes") if isinstance(contract, dict) else None
        if isinstance(allowed_modes, list) and allowed_modes and training_mode not in allowed_modes:
            raise ValueError(
                f"training_mode={training_mode!r} is not allowed by this run contract; "
                f"allowed_training_modes={allowed_modes}"
            )

        # benchmark 不再由调用方传入 (worker_submit 已删掉该参数): 统一从 state
        # 读。playground setup 时 initialize(benchmark=...) 已把 run 的 benchmark
        # 身份写进 state["benchmark"] (eval_all 模式下收敛到 eval_all.bench_id)。
        run_bench = state.get("benchmark", "gsm8k")
        # eval_all: state 里由 playground set up 时写入的评测配置。``bench_id``
        # 非空即让 pipeline stage-3 走 benchmark suite 统一评测 (任意 bench)。
        eval_all_cfg = self.os.load_state().get("eval_all") or {}
        bench_id = eval_all_cfg.get("bench_id") or None
        eval_all_forward = dict(eval_all_cfg) if isinstance(eval_all_cfg, dict) else {}

        # slime_sft: state 里 playground initialize 时写入的 slime 训练配置,
        # worker 侧的 --slime-config 覆盖项合并进来, data_file 由本方法强制指到
        # 净化副本 (见下)。
        is_slime = training_mode == "slime_sft"
        slime_cfg_forward: dict[str, Any] = {}
        slime_config_warnings: list[str] = []
        if is_slime:
            state_slime = self.os.load_state().get("slime_config") or {}
            if isinstance(state_slime, dict):
                slime_cfg_forward.update(state_slime)
            # 统一走 training-config-path 一条通道, 不同 mode 按不同语义导入:
            # llama_factory 模式把它当 LLaMA-Factory 训练 YAML 原样透传; slime_sft
            # 把它当 slime 配置文件 (YAML/JSON) 解析成 dict。优先级: run 默认
            # (state) < 配置文件 < 内联 slime_config (JSON, 最高)。data_file 除外
            # —— 下方强制指到净化副本, worker 传什么都被覆盖。
            slime_cfg_forward.update(load_slime_config_file(training_config_path))
            if isinstance(slime_config, dict) and slime_config:
                slime_cfg_forward.update(slime_config)
            slime_config_warnings = slime_override_warnings(
                slime_cfg_forward, state_slime
            )

        # expos_training_recommendations (档位默认 + 预检, 2026-09-03 起 worker 无感):
        # model_tier 由 run seed 期按 base_model 自动派生 (state["slime_config"] 自带,
        # 见 recommendations.resolve_run_defaults); 提交层若仍显式传了 model_tier (历史
        # 遗留) 沿用旧语义参与匹配。命中档位 -> 1) 档位默认 slime 配置补缺 (优先级最低:
        # worker 显式 / 配置文件 / run 默认更高; 只补缺 + 纠正 run 级旧默认的尺寸不匹配
        # 键); 2) 跑档位预检 —— HARD_REFUSE 入队前 raise (明确理由传给 worker),
        # WARNING_REFUSE 进 preflight_warnings (对 agent 可见, 不阻止)。
        # 只对 slime_sft 提交生效; 无档位命中 -> 不预检不用默认。
        # 对 agent 不可见 (只进 state, 不进 public/ / prompts)。
        preflight_warnings: list[str] = []
        if is_slime:
            recs = effective_recommendations(
                self.os.load_state().get("training_recommendations")
            )
            model_tier = str(slime_cfg_forward.get("model_tier") or "").strip()
            profile = resolve_profile(model_tier, recs)
            if profile is not None:
                # gpu_nums 回填: 没传卡数 (None) -> 按命中档位 default_config.gpu_nums
                # 落地 (4B=2, 8B=1, 35B=4); 显式传值保留, 仍由 gpu_nums_must_match
                # (allowed_gpu_nums) + render 守卫逐位校验。
                if gpu_nums is None:
                    gpu_nums = int(profile.default_config.get("gpu_nums") or 1)
                profile_slime = profile.default_config.get("slime") or {}
                if isinstance(profile_slime, dict):
                    slime_cfg_forward = {**profile_slime, **slime_cfg_forward}
                preflight_warnings = apply_preflight(
                    profile,
                    PreflightContext(
                        model_tier=model_tier,
                        model_path=str(slime_cfg_forward.get("model_dir") or ""),
                        training_mode=training_mode,
                        gpu_nums=gpu_nums,
                        slime_config=slime_cfg_forward,
                        train_data_path=data_path,
                        recommendations=recs,
                    ),
                )

        if gpu_nums is None:
            # 非 slime / 无档位命中: None 语义 = 1 (与 render `or 1` / 旧默认一致)
            gpu_nums = 1
        exp_id = self.os.new_experiment_id()
        exp_workspace = self.os.experiments_dir / exp_id
        ensure_dir(exp_workspace)
        ensure_dir(exp_workspace / "artifacts")

        # 在任何 GPU 工作入队前执行 benchmark query 去污染。原始数据保持不变，
        # backend 只接收 experiment workspace 中的净化副本。
        data_path_obj = Path(data_path)
        # Decontamination query load is best-effort: a broken / unavailable
        # benchmark query source must not block submission. The submit-side
        # decontamination is defense-in-depth on top of the worker's own
        # leakage check, not a hard gate — on failure we degrade to a
        # pass-through copy (original rows retained, 0 removed) and record the
        # reason, so the pipeline still runs and the score is reported honestly.
        _decontam_note = ""
        try:
            benchmark_queries = self.benchmark_query_loader(
                run_bench,
                bench_id=str(bench_id) if bench_id else None,
                eval_all=eval_all_forward,
            )
        except Exception as exc:  # noqa: BLE001 — degrade, don't block the trial
            benchmark_queries = []
            _decontam_note = (
                f"benchmark query load failed ({exc}); decontamination pass-through "
                "(worker-side leakage check is the authority)"
            )
        if is_slime:
            # slime 训练数据是 parquet (messages 列), 净化副本也写 parquet。
            clean_data_path = exp_workspace / "artifacts" / "decontaminated_train.parquet"
            decontamination_report = decontaminate_parquet(
                data_path_obj,
                clean_data_path,
                benchmark_queries,
                benchmark=str(bench_id or run_bench),
            ).to_dict()
            slime_cfg_forward["data_file"] = str(clean_data_path)
        else:
            clean_data_path = exp_workspace / "artifacts" / "decontaminated_train.jsonl"
            decontamination_report = decontaminate_jsonl(
                data_path_obj,
                clean_data_path,
                benchmark_queries,
                benchmark=str(bench_id or run_bench),
            ).to_dict()
        train_sha = sha256_of_file(data_path_obj) if data_path_obj.exists() else ""
        clean_train_sha = sha256_of_file(clean_data_path)

        work_order = WorkOrder(
            experiment_id=exp_id,
            worker_id=worker_id,
            parent_experiment_id=parent_experiment_id,
            benchmark=run_bench,
            objective=objective,
            hypothesis=hypothesis,
            changed_variable=changed_variable,
            training_config_path=str(training_config_path),
            training_mode=training_mode,
            train_data_path=str(data_path_obj),
        )
        write_json(exp_workspace / "work_order.json", dump_model(work_order))

        # 先落一个 running 的 ExperimentNode, 让 state 立即反映"有实验在跑"
        exp_node = ExperimentNode(
            experiment_id=exp_id,
            worker_id=worker_id,
            parent_experiment_id=parent_experiment_id,
            status="running",
            training_mode=training_mode,
            benchmark=run_bench,
            workspace=str(exp_workspace),
        )
        self.os.register_experiment(exp_node)
        self.os.events.emit(
            "worker_submit_started",
            worker_id=worker_id,
            experiment_id=exp_id,
            tool="worker.submit",
            payload={
                "train_data_path": str(data_path_obj),
                "train_data_sha256": train_sha,
                "decontaminated_train_data_path": str(clean_data_path),
                "decontaminated_train_data_sha256": clean_train_sha,
                "decontamination_report": decontamination_report,
                "training_mode": training_mode,
                "training_config_path": str(training_config_path),
            },
        )

        if self.submit_backend is None:
            # 没配 backend -> 直接失败, 不再走 dry-run
            result_payload = {
                "status": "failed",
                "score": None,
                "failure_reason": "no submit backend configured; refuse to run.",
                "training_mode": training_mode,
                "benchmark": run_bench,
                "trial_path": str(exp_workspace),
            }
        else:
            try:
                result_payload = self.submit_backend.submit(
                    experiment_id=exp_id,
                    worker_id=worker_id,
                    benchmark=run_bench,
                    train_data_path=str(clean_data_path),
                    training_config_path=str(training_config_path),
                    training_mode=training_mode,
                    experiment_workspace=exp_workspace,
                    objective=objective,
                    hypothesis=hypothesis,
                    changed_variable=changed_variable,
                    parent_experiment_id=parent_experiment_id,
                    bench_id=bench_id,
                    eval_all=eval_all_forward,
                    gpu_nums=gpu_nums,
                    need_exclusive_gpu=need_exclusive_gpu,
                    allowed_training_modes=list(allowed_modes or []),
                    **(
                        {"slime_config": slime_cfg_forward}
                        if is_slime
                        else {}
                    ),
                )
            except Exception as exc:  # noqa: BLE001 — capture backend crashes explicitly
                result_payload = {
                    "status": "failed",
                    "score": None,
                    "failure_reason": f"submit backend crashed: {exc}",
                    "trial_path": str(exp_workspace),
                }
                self.os.events.emit(
                    "worker_submit_crashed",
                    worker_id=worker_id,
                    experiment_id=exp_id,
                    payload={"error": str(exc)},
                )

        # 组装 ExperimentResult 并落盘.  score 在此刻通常还没有, 由 pipeline
        # 自己覆盖 result.json 完成结果回填。
        raw_status = str(result_payload.get("status") or "failed")
        if raw_status == "deferred":
            # Atomic backend admission may lose a race after the cheap precheck.
            # Preserve its audit node as aborted, NEVER as a failed training.
            write_json(exp_workspace / "result.json", dump_model(ExperimentResult(
                experiment_id=exp_id, worker_id=worker_id, status="aborted", eval_status="skipped",
                benchmark=run_bench, training_mode=training_mode, submit_result=result_payload,
                notes="Queue admission deferred; no GPU script started.",
            )))
            self.os.update_experiment(exp_id, status="aborted", score=None)
            self.os.events.emit("worker_submit_deferred", worker_id=worker_id, experiment_id=exp_id,
                                payload={"reason": result_payload.get("summary", "queue_full")})
            return {"status": "deferred", "experiment_id": exp_id, "reason": "queue_full",
                    "hint": result_payload.get("summary", "Wait for queue capacity; do not spin-retry.")}
        if raw_status not in {"completed", "failed", "pending", "running"}:
            raw_status = "failed"
        score = result_payload.get("score")
        result = ExperimentResult(
            experiment_id=exp_id,
            worker_id=worker_id,
            status=raw_status,  # type: ignore[arg-type]
            eval_status="completed" if score is not None else "pending",
            benchmark=run_bench,
            training_mode=training_mode,
            score=score if isinstance(score, (int, float)) else None,
            effective_config={"training_config_path": str(training_config_path)},
            submit_result=result_payload,
            failure_reason=str(result_payload.get("failure_reason") or ""),
            artifacts=self._collect_result_artifacts(
                result_payload, data_path_obj, clean_data_path
            ),
        )
        write_json(exp_workspace / "result.json", dump_model(result))

        # 同步 ExperimentNode。NodeStatus 集合 pending/running/finished/failed/aborted.
        node_status = {
            "completed": "finished",
            "failed": "failed",
            "pending": "pending",
            "running": "running",
        }.get(result.status, "failed")
        self.os.update_experiment(
            exp_id,
            status=node_status,
            score=result.score,
            submit_trial_path=str(result_payload.get("trial_path") or exp_workspace),
        )
        self.os.events.emit(
            "worker_submit_finished",
            worker_id=worker_id,
            experiment_id=exp_id,
            tool="worker.submit",
            payload={
                "status": result.status,
                "score": result.score,
                "gpu_script_id": result_payload.get("gpu_script_id", ""),
            },
        )
        gpu_script_id = str(result_payload.get("gpu_script_id") or "")
        hint = ""
        if raw_status == "running" and gpu_script_id:
            hint = (
                "Pipeline queued on gpu_backend. Poll with:\n"
                f"  gpu-backend script-status --script-id {gpu_script_id}   "
                "(not installed? use `public/gpu-backend`, seeded in the workspace)\n"
                f"  gpu-backend script-log    --script-id {gpu_script_id} --tail 200\n"
                f"When the script finishes, read {exp_workspace / 'result.json'} for the "
                "final score. Do NOT duplicate this trial. Independent small experiments may queue "
                "while it runs; inspect experiment_queue first. Wait for evidence before dependent expansion."
            )
        # 覆盖告警塞进返回 + hint, 让 agent 在 observation 和提示里都看得到;
        # 但只是提示, 不阻止 (worker 对训练方向有最终决定权)。
        if slime_config_warnings:
            hint = (
                (hint + "\n\n" if hint else "")
                + "WARNINGS (slime_config overrides, not blocked):\n- "
                + "\n- ".join(slime_config_warnings)
            )
        # expos_training_recommendations 的 WARNING_REFUSE 预检结果: 同款通道
        # (对 agent 可见, 不阻止)。
        if preflight_warnings:
            hint = (
                (hint + "\n\n" if hint else "")
                + "PREFLIGHT WARNINGS (not blocked):\n- "
                + "\n- ".join(preflight_warnings)
            )
        ret = {
            "experiment_id": exp_id,
            "workspace": str(exp_workspace),
            "status": result.status,
            "score": result.score,
            "gpu_script_id": gpu_script_id,
            "gpu_log_path": str(result_payload.get("gpu_log_path") or ""),
            "pipeline_script_path": str(result_payload.get("pipeline_script_path") or ""),
            "result_json_path": str(exp_workspace / "result.json"),
            "decontamination_report": decontamination_report,
            "hint": hint,
            "submit_result": result_payload,
            "queue": self.queue_status(),
        }
        if slime_config_warnings:
            ret["slime_config_warnings"] = slime_config_warnings
        if preflight_warnings:
            ret["preflight_warnings"] = preflight_warnings
        if is_slime:
            ret["slime_config"] = slime_cfg_forward
        return ret

    def _collect_result_artifacts(
        self,
        result_payload: dict[str, Any],
        train_data_path: Path,
        decontaminated_train_data_path: Path,
    ) -> list[ArtifactRecord]:
        artifacts: list[ArtifactRecord] = []
        if train_data_path.exists():
            artifacts.append(
                ArtifactRecord(
                    type="train_data",
                    path=str(train_data_path),
                    sha256=sha256_of_file(train_data_path),
                    summary="original training data before benchmark decontamination",
                )
            )
        artifacts.append(
            ArtifactRecord(
                type="decontaminated_train_data",
                path=str(decontaminated_train_data_path),
                sha256=sha256_of_file(decontaminated_train_data_path),
                summary="benchmark-query decontaminated training data used by backend",
            )
        )
        for key, atype in (
            ("checkpoint_dir", "checkpoint"),
            ("eval_dir", "eval"),
            ("tensorboard_dir", "tensorboard"),
            ("pipeline_script_path", "pipeline_script"),
            ("gpu_log_path", "gpu_log"),
        ):
            path = result_payload.get(key)
            if not path:
                continue
            artifacts.append(
                ArtifactRecord(
                    type=atype,
                    path=str(path),
                    summary=f"produced by submit backend ({key})",
                )
            )
        return artifacts

    # ── worker.add_reviewer(experiment_ids) ──────────────────────────────────

    def add_reviewer(
        self,
        *,
        worker_id: str,
        experiment_ids: list[str],
        objective: str = "",
    ) -> dict[str, Any]:
        """Instantiate a reviewer node bound to the given experiments.

        ``objective`` 描述这次 review 要重点核对的点 (验证什么 / 怀疑什么),
        派发 reviewer agent 时会注入到它的 task description 里。
        """
        graph = self.os.load_graph()
        if worker_id not in graph.workers:
            raise KeyError(f"unknown worker_id: {worker_id}")
        missing = [eid for eid in experiment_ids if eid not in graph.experiments]
        if missing:
            raise KeyError(f"unknown experiment_ids: {missing}")

        reviewer_id = self.os.new_reviewer_id()
        reviewer = ReviewerNode(
            reviewer_id=reviewer_id,
            worker_id=worker_id,
            reviewed_experiment_ids=list(experiment_ids),
            objective=objective,
            status="pending",
        )
        node = self.os.register_reviewer(reviewer)
        return {
            "reviewer_id": reviewer_id,
            "worker_id": worker_id,
            "objective": node.objective,
            "workspace": node.workspace,
            "reviewed_experiment_ids": node.reviewed_experiment_ids,
            "hint": (
                "The reviewer workspace at reviews/{reviewer_id}/ is now created. "
                "Reviewer must call `reviewer.inspect_experiment` for each id and "
                "then submit ReviewReport via `reviewer.submit_review`."
            ),
        }

    # ── worker.finalize / status update ─────────────────────────────────────

    def finalize(self, worker_id: str, *, status: str = "finished", notes: str = "") -> dict[str, Any]:
        """Mark a worker as done. Called when the worker agent finishes its turn.

        收尾前先把该 worker 已完成实验的 result.json 同步回 state.json
        (否则 pipeline 跑完了分数还停在 None, 变成孤儿节点)。
        """
        self.os.sync_experiment_results(worker_id=worker_id)
        node = self.os.update_worker(worker_id, status=status, updated_at=time.time())
        if notes:
            history_path = Path(node.workspace) / "HISTORY.md"
            existing = history_path.read_text(encoding="utf-8") if history_path.exists() else ""
            history_path.write_text(
                existing.rstrip() + f"\n\n## Finalize ({status})\n\n{notes.strip()}\n",
                encoding="utf-8",
            )
        return {"worker_id": worker_id, "status": status}
