# CLI 形式 ExpOS — 内部机制（cc_with_expos）

todo 要求：“写一份 **CLI 形式的 ExpOS** 供 agent 调用——一个 CC 兼任 Reviewer/Worker/MainAgent
角色，不再通过注册工具表，而是通过 CLI 让一个 agent 可以调用多个角色的工具；与当前 ExpOS 保持
**相同工具接口**，但实现逻辑有变化——工具不会影响 main-agent-loop，只会做对应文件系统的修改。”

这里记录它怎么做到的。

## 一、复用骨架：`expos.cli.main(argv, submit_backend=...)`

框架自带 ExpOS 的 CLI 入口 `tools/ExpOS/expos/cli/main.py::main` 自带一个注入缝：
`submit_backend`。cc_with_expos 原样复用整个子命令树，只替换 backend。

```python
# tools/cc_rollout_expos/expos_cli.py
from .expos_cli_backend import SubmitPipelineBackend

def main(argv=None, *, submit_backend=None):
    from expos.cli.main import main as expos_main   # 延迟导入（tools/ExpOS 在 PYTHONPATH 上）
    return expos_main(argv, submit_backend=submit_backend or SubmitPipelineBackend())

if __name__ == "__main__":
    raise SystemExit(main())
```

`submit_backend` 是测试注入缝（`test/cc_rollout/test_expos_cli.py` 传 `FakeSubmitBackend`，
hermetic 全流程验证）。

`cc_rollout_expos` 是**顶层独立包**（`tools/cc_rollout_expos/`，不依赖 playground/）。
cc_with_expos seed 时把它复制进 `workspace/public/cc_rollout_expos/`（与 gpu_backend /
backend 模块同款只读拷贝），launcher 执行的就是 workspace 里的**副本**（见下）。

## 二、`./expos` launcher（seed 烘焙）

`seed.py::_seed_expos_launcher` 写 `workspace/expos`（755）：

```bash
#!/usr/bin/env bash
export PYTHONPATH="<workspace>/public:<repo_root>:<repo_root>/tools/ExpOS"${PYTHONPATH:+:${PYTHONPATH}}
export EXPOS_ROOT="<workspace>/experiment_os"
exec "<sys.executable>" -m cc_rollout_expos "$@"
```

- `cc_rollout_expos` 从 **`workspace/public/cc_rollout_expos/`**（seed 复制的副本）解析：
  launcher 把 workspace/public 放在 PYTHONPATH **最前**、`tools/` 不在其上，`python -m
  cc_rollout_expos` 只会解析到 workspace 里的副本（已用 seeded workspace 实测验证）。
- repo root 供 `backend`（`SubmitPipelineBackend` 延迟 import `backend.backend.submit_pipeline`），
  tools/ExpOS 供 `expos`（`expos.cli.main`）。
- `EXPOS_ROOT` 指向 seed 的 experiment_os；`expos.cli.main` 的 `build_context` 从 `--root` 或 `EXPOS_ROOT` 取根。

## 三、SubmitPipelineBackend —— 实现 SubmitBackend 协议

`worker_service.SubmitBackend` 是一个 typing Protocol（`submit(*, experiment_id, worker_id,
benchmark, train_data_path, training_config_path, training_mode, experiment_workspace, ...)`）。
`SubmitPipelineBackend.submit` 把它翻译成 `SubmitRequest` 并**进程内**执行：

```python
# tools/cc_rollout_expos/expos_cli_backend.py
def submit(self, *, experiment_id, worker_id, benchmark, train_data_path,
           training_config_path, training_mode, experiment_workspace,
           objective="", hypothesis="", changed_variable="", parent_experiment_id=None,
           bench_id=None, eval_all=None, gpu_nums=1, slime_config=None,
           allowed_training_modes=None, need_exclusive_gpu=True, **extra):
    from backend.backend.submit_pipeline.driver import run_trial     # 延迟导入
    from backend.backend.submit_pipeline.models import SubmitRequest
    request = SubmitRequest(
        experiment_id=experiment_id, worker_id=worker_id,
        benchmark=benchmark or "gsm8k",
        train_data_path=str(train_data_path),            # 已是去污染净本
        training_config_path=str(training_config_path),
        training_mode=training_mode, experiment_workspace=str(experiment_workspace),
        gpu_nums=int(gpu_nums), need_exclusive_gpu=bool(need_exclusive_gpu),
        objective=objective, hypothesis=hypothesis, changed_variable=changed_variable,
        parent_experiment_id=parent_experiment_id, bench_id=bench_id,
        eval_all=eval_all or {}, slime_config=slime_config or {},
    )
    return run_trial(request).to_dict()
```

要点：
- **路径/去污染/契约校验在 WorkerService 侧已完成**，这里只做翻译 + 执行，不重复校验。
- `need_exclusive_gpu=True` 是协议之外的 kwarg（调用点总是传，默认 True）；`**extra` 兜底协议演进，绝不因多传字段崩。
- `slime_config` 只在 slime_sft 时到达；`allowed_training_modes` 已在 run_contract gate 生效，不传给 run_trial。
- `run_trial(request).to_dict()` 返回 dict（`status` / `gpu_script_id` / `result_json_path` / `decontamination_report` / `hint` …），`WorkerService.submit` 把它记进 result.json 的 `submit_result`。

## 四、一条 submit 命令里 service 层的上游步骤

`./expos worker submit` 先走 `WorkerService.submit`（与 v2 完全相同的逻辑），再落 backend：

1. worker 存在性校验 + 路径绝对化；
2. LoRA 硬拒 —— `llama_factory_lora_sft` 不在 argparse `choices`（{llama_factory_sft, slime_sft}），在 **argparse 层**就被 `SystemExit(2)` 挡掉（handler 都进不去）；
3. `state["run_contract"]["allowed_training_modes"]` gate；
4. 去污染：`benchmark_query_loader`（默认 `load_benchmark_queries`，WorkerService `__init__` 捕获）读 `bmk_root/format/<bench_id>/*.jsonl` 的 query 字段，写 `experiments/<eid>/artifacts/decontaminated_train.jsonl|parquet`；
5. 落 `work_order.json` + 占位 `result.json`（status=running, score=None）；
6. 注册 `ExperimentNode(running)` 进 state 图；
7. 调 `submit_backend.submit(...)` → 本实现 = 进程内 `run_trial` → 渲染 `pipeline.sh` → 提交 gpu_backend :8989 → async 返回。

## 五、完整工具接口（子命令树）

`expos.cli.main` 的树（原样可用，`--root` 顶层 flag，`--json` 全局输出开关；**每条命令**都 append 一条 `cli_invoked` 审计事件到 `state/events.jsonl`）：

```
init                # 初始化 ExperimentOS 根（seed 已做，agent 一般不需要）
events tail         # 最近事件流
dump-state          # 打印 state.json（Main 开场定位）
state add-new-worker --objective "..."            # 实例化 worker 节点（Main）
state get-current-worker
state inspect-worker --worker-id <id>
state sync [--worker-id <id>]                      # 磁盘结果 reconcile 进 state（修孤儿）
state finalize-worker --worker-id <id> [--status <s>]
worker show-datapool                             # data_pool 全量列表（无 --worker-id）
worker inspect-data --data-id <id>               # 单个 data_pool 条目
worker update-readme --data-id <id> --summary <text>   # 追加 README 笔记
worker add-data --worker-id <id> --artifact-path <file> [--artifact-type train_data] \
        [--readme <md>] [--tags ...] [--source-experiment-id <eid>] [--benchmark <b>] \
        [--score <f>] [--num-rows <n>] [--data-id <id>]
worker submit --worker-id <id> --data-path <file> --training-config-path <yaml> \
        --training-mode {llama_factory_sft,slime_sft} \
        [--gpu-nums <n>] [--slime-config <json>] [--objective] [--hypothesis] \
        [--changed-variable] [--parent-experiment-id] [--need-exclusive-gpu]
worker add-reviewer --worker-id <id> --experiment-ids <eid,...> --objective "..."
worker finalize --worker-id <id> [--status <s>] [--notes]
reviewer show-datapool
reviewer inspect-data --data-id <id>
reviewer inspect-experiment --experiment-id <eid>      # 读 work_order/result/轨迹
reviewer submit-review --reviewer-id <rid> --experiment-id <eid> --report <file.json>
reviewer finalize --reviewer-id <rid> [--status <s>]
```

## 六、与 v2 的 ExpOS 对比

| 维度 | v2（rsi_master） | cc_with_expos（本实现） |
|---|---|---|
| agent 如何调 ExpOS | 注册工具表（`agent_loop/tools.py` 注入三个 agent） | `./expos` CLI（一个 CC 换动词 = 换角色） |
| 多 agent 编排 | main 起 worker/reviewer 子 agent（copy_agent + ThreadPoolExecutor） | **无**——一个 CC 自己兼任三角色 |
| `worker submit` 传输 | `expos.backends` → ExpOS FastAPI :8990 → gpu_backend | 进程内 `run_trial` → gpu_backend :8989（无 :8990） |
| 影响 main-agent-loop | 是——submit 是主循环的工具调用 | **否**——纯文件系统修改 + 一次异步提交 |
| 工具接口 | 与 `expos.cli` 动词 1:1 | **同一个** `expos.cli.main` 子命令树 |
| 审计 | `events.jsonl` | 同（`cli_invoked`，tool=`cli.worker.submit` 等） |
| 去污染 / work_order / 占位 result | WorkerService 落盘 | **完全相同**（同一份 WorkerService） |

一句话：**换掉的是传输层和编排层，工具接口和文件系统产物不变。** 测试用 `FakeSubmitBackend`
注入即可无 GPU/网络跑通 `add-new-worker → worker submit → 断言产物与 kwargs` 全流程
（`test/cc_rollout/test_expos_cli.py`，3/3 PASS）。
