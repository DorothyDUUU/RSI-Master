# cc_with_expos — agent 执行逻辑

> config `mode: cc_with_expos`。workspace 初始化同 cc_naive，另注入三样东西：`experiment_os/`
> （ExpOS DAG 根，seed 时已 initialize）、`SUBMIT_GUIDE.md`（submit 工作流指南）、`./expos`
> launcher。一个 CC 通过 **CLI 形式 ExpOS** 兼任 MainAgent / Worker / Reviewer 三角色——
> **不再通过注册工具表**，而是用不同的 `./expos` 动词切换角色。工具接口与框架自带 ExpOS
> **完全一致**，但 submit 走 serverless submit 训练管线（见 [`expos_cli.md`](expos_cli.md)）。

## 一、开场差异（vs naive）

- 系统 prompt = `compose("protocol_with_expos.md", …, submit_driver_with_expos)`：协议里多了
  “三个角色 / `./expos` verb 序列” 章节，workspace 布局图里多了 `experiment_os/`、`SUBMIT_GUIDE.md`、`expos`。
- seed 阶段已 `ExperimentOS(experiment_os).initialize(...)`：`state/state.json` 写入
  `benchmark`（= `eval.bench_id`）、`eval_all`（bench_id/limit）、`run_contract`
  （`allowed_training_modes: [llama_factory_sft, slime_sft]`）——**eval contract 已经配进 DAG 的 state**，
  协议明确说“别去改它”。
- `workspace/expos` launcher（755）设好 PYTHONPATH（`workspace/public` 最前 + repo root +
  tools/ExpOS）+ `EXPOS_ROOT` 后 `python -m cc_rollout_expos "$@"`。`cc_rollout_expos` 是
  **顶层独立包**（`tools/cc_rollout_expos/`），seed 时复制进 `workspace/public/cc_rollout_expos/`，
  执行的就是 workspace 里的副本（详见 [`expos_cli.md`](expos_cli.md)）。

## 二、一个 CC 的三个角色（`protocol_with_expos.md` §3）

agent 每研究一轮按 MAIN → WORKER → REVIEWER 的顺序打 `./expos` 动词：

```text
MAIN      ./expos dump-state                                  # 定位自己
          ./expos state add-new-worker --objective "..."      # 声明一个 worker 目标
          ./expos worker show-datapool                        # data_pool 里有什么

WORKER    ./expos worker add-data --worker-id <id> --artifact-path <file> [--readme <md>]
          ./expos worker submit --worker-id <id> \
              --data-path <file> --training-config-path <yaml> \
              --training-mode llama_factory_sft | slime_sft   # ASYNC submit-train
          ./expos worker add-reviewer --worker-id <id> \
              --experiment-ids <eid> --objective "..."        # 挂一个 review
          public/gpu-backend script-status --script-id <id>   # 轮询（禁止 sleep-loop）
          ./expos worker finalize --worker-id <id>

REVIEWER  ./expos reviewer inspect-experiment --experiment-id <eid>   # 读 work_order/result/轨迹
          ./expos reviewer submit-review --reviewer-id <rid> --experiment-id <eid> --report <file.json>
          ./expos reviewer finalize --reviewer-id <rid>
```

“文件系统即 DAG”：`experiment_os/state/state.json` 是唯一事实源，每条 `./expos` 动词都经
service 层落盘并 append 一条 `cli_invoked` 审计事件到 `state/events.jsonl`。**禁止手改 state.json**。

## 三、每个 ./expos 动词的副作用

这一节把 §二 动词表里**每一个工具到底改了什么**完整展开。先立三条铁律，再给一个
一页速查表，然后逐动词细述。副作用归结为四类：**事件（events.jsonl）/ DAG 状态
（state.json）/ 文件系统（experiment_os/ 下的目录与文件）/ 外部（gpu_backend 提交）**。

### 3.0 三条铁律（所有动词都成立）

1. **`state.json` 是唯一事实源**。任何结构 / 节点状态变化都走 service 层（
   `WorkerService` / `StateService` / `ReviewerService` / `DataPoolService`），最终经
   `ExperimentOS.save_state` / `commit_graph` 原子落盘；服务模块**不直接读写文件**。
2. **`state/events.jsonl` 是 append-only 审计流**。`expos.cli.main` 在 `finally` 块里给
   **每一条 CLI 命令——无论读写**——都追加一条 `cli_invoked` 事件（`tool="cli.<verb路径>"`，
   参数经 `_audit_params` 清洗，丢掉内部字段和超长字符串）；除此之外，service 层每次状态迁移
   还会发一条对应的**语义事件**。事件行是 `OSEvent` 结构：

   ```json
   {"schema_version":"rsi_master.event.v2","type":"worker_registered",
    "worker_id":"worker_0001","experiment_id":null,"reviewer_id":null,"data_id":null,
    "tool":null,"payload":{"parent_worker_id":null,"objective":"..."},"time":1755000000.0}
   ```

3. **文件系统 = DAG 的实体**。每个节点一个目录，目录里的 `config.json` 是节点状态的
   落盘镜像（随 service 层同步重写）：`worker/<id>/`、`experiments/<id>/`、`reviews/<id>/`、
   `data_pool/<data_id>/`。另有 `HISTORY.md`/`TASK.md`/`README.md` 等可读产物，和 submit 的
   `work_order.json`/`result.json`/`artifacts/`。

### 3.1 一页速查表

| 动词 | 读写 | state.json | events.jsonl | 文件系统 | 外部 |
|---|---|---|---|---|---|
| `dump-state` | 读 | — | `cli_invoked` | — | — |
| `events tail` | 读 | — | `cli_invoked` | — | — |
| `state get-current-worker` | 读 | — | `cli_invoked` | — | — |
| `worker/reviewer show-datapool` | 读 | — | `cli_invoked` | — | — |
| `worker/reviewer inspect-data` | 读 | — | `cli_invoked` | — | — |
| `state inspect-worker` | 读+sync | 可能学回分数 | +`experiment_synced` | 可能重写 exp config.json | — |
| `reviewer inspect-experiment` | 读+sync | 可能学回分数 | +`experiment_synced` | 可能重写 exp config.json | — |
| `state add-new-worker` | 写 | 新 worker 节点 + `current_worker_id` + counter | `worker_registered` | `worker/<id>/`+种子文件 | — |
| `worker add-data` | 写 | `data_pool_index` + counter | `data_added` | `data_pool/<id>/`+拷贝+`data.jsonl` | — |
| `worker update-readme` | 写 | — | `data_readme_updated` | 追加 `data_pool/<id>/README.md` | — |
| `worker submit` | 写 | 新 exp 节点 + counter + score 上卷 | `worker_submit_started/finished/crashed` + `experiment_registered/updated` | `experiments/<id>/` 全套 + 去污染 | **gpu_backend :8989** |
| `worker add-reviewer` | 写 | 新 reviewer 节点 + counter + 图边 | `reviewer_registered` | `reviews/<id>/`+`reports/` | — |
| `worker finalize` / `state finalize-worker` | 写+sync | worker 状态终结 + 学回分数 | `worker_updated` +`experiment_synced` | worker config.json 重写 + HISTORY 追加 | — |
| `state sync` | 写 | 学回分数 + 重建 `data_pool_index` | `experiment_synced` ×N | exp config.json + `data.jsonl` | — |
| `reviewer submit-review` | 写 | 图边 + reviewer 状态 | `review_registered` +`reviewer_updated` +(data 推荐事件) | `reviews/<id>/reports/<eid>.json` + config.json | — |
| `reviewer finalize` | 写 | reviewer 状态终结 | `reviewer_updated` | `reviews/<id>/config.json` | — |

> **读命令带 sync 写**：`state inspect-worker` / `reviewer inspect-experiment` / `worker
> finalize` / `state finalize-worker` / `state sync` 会先跑 `sync_experiment_results`——把磁盘上
> 已终结的 `result.json` 分数学回 DAG。这是设计好的，不是 bug（见 3.3 的同步缺口）。

### 3.2 逐动词副作用

#### 读类（只发 `cli_invoked`，零状态/文件副作用）

- **`dump-state`** `[--json]` —— 打印整个 `state.json`。不改任何东西。
- **`events tail --limit N [--worker-id] [--experiment-id] [--reviewer-id] [--type ...]`** ——
  按过滤条件回放 `events.jsonl` 尾部。不改任何东西。
- **`state get-current-worker`** —— DAG 摘要：`loop` 状态、`current_worker_id`、
  `task_description`/`benchmark`、`best`、`counters`、每个 worker 的
  id/parent/status/objective/best_score/experiments/reviewers、`root_worker_ids`。
- **`worker show-datapool`** / **`reviewer show-datapool`** —— 两条同名命令，逐条列出
  `data_pool/*/config.json` 的摘要 + 目录内文件名。**只读**，worker 和 reviewer 都能看。
- **`worker inspect-data --data-id`** / **`reviewer inspect-data --data-id`** —— 单条记录
  （record + README 全文 + 文件清单与大小）。**只读**。
- **`reviewer inspect-experiment --experiment-id <eid>`** —— 展开 `experiments/<eid>/`：
  `work_order.json` + `result.json` + 全部文件清单（rglob）+ 该实验事件尾。**注意：内部先跑
  一次全量 `sync_experiment_results()`**（不带 worker 过滤，整个 DAG），所以它可能在返回前
  把别的实验的分数也学回来——是“读命令带写”。

#### `state add-new-worker [--parent-worker-id] [--objective]`

- **state.json**：`counters.worker` +1 → 分配 `worker_NNNN`；`graph.workers[worker_NNNN]` 挂一个
  `WorkerNode{status:"pending", parent_worker_id, objective, workspace}`；`state.current_worker_id`
  指向新 worker（**每次 add 都会把 current 拨到最新**）。
- **events.jsonl**：`worker_registered`（payload: `parent_worker_id`, `objective`）+ `cli_invoked`。
- **文件系统**：建 `worker/<id>/` 并种三个文件——`HISTORY.md`（history log 头）、`TASK.md`
  （带 parent 的任务简报模板）、`config.json`（WorkerNode 落盘镜像）。
- **错误**：`--parent-worker-id` 不存在 → `ValueError`（`"null"`/`"None"`/`""` 归一化为 root，不算错）。
- **返回**：`worker_id` / `parent_worker_id` / `workspace` / `history_path` / `task_path` /
  `run_contract` + 提示“开工前先填 HISTORY.md / TASK.md”。

#### `worker add-data --worker-id --artifact-path <file> [--source-experiment-id --benchmark --artifact-type --num-rows --score --tags --readme --data-id]`

- **文件系统**：建 `data_pool/<data_id>/`；若给了 `--artifact-path`，把文件**拷贝**进池为
  `raw<ext>`（原文件不动），并自动算 sha256 + 行数（`.jsonl` 自动 `count_jsonl_rows`）；写
  `config.json`（`DataPoolRecord`，`promotion_status:"candidate"`）；写 `README.md`（`--readme`
  自定义，否则默认模板）；重建 `data_pool/data.jsonl`（全池一行一条记录，按 data_id 排序）。
- **state.json**：`data_pool_index` 加入/更新该条目；`--data-id` 缺省时 `counters.data` +1。
- **events.jsonl**：`data_added`（payload: `artifact_type`, `benchmark`, `rows`, `descriptions`）
  + `cli_invoked`。
- **错误**：`--artifact-path` 不存在 → `FileNotFoundError`。
- **语义**：worker 才能写池，reviewer 只能读。

#### `worker update-readme --data-id <id> --summary <text>`

- **文件系统**：`data_pool/<id>/README.md` 追加 `## Update\n\n<summary>`。
- **events.jsonl**：`data_readme_updated`（payload: `chars`）+ `cli_invoked`。
- **state.json**：**不变**（README 只是给 agent 看的笔记，模型直接编辑 README 也合法，
  CLI 只做记账）。

#### `worker submit` — 副作用最大的一条（详细链条见 §四）

- **校验/守卫**：worker 存在；`llama_factory_lora_sft` 硬拒（服务层 + argparse choices 双保险）；
  `training_mode ∈ state["run_contract"].allowed_training_modes`；training-config 文件存在。
- **state.json**：`counters.experiment` +1 → `exp_NNNN`；`graph.experiments[exp_NNNN]` 注册
  `ExperimentNode{status:"running"}`；`update_experiment` 把 submit 结果写回节点（
  status 视 backend 返回 running/failed/pending/finished），score 非 None 时上卷
  `worker.best` / `state.best`。
- **events.jsonl**：`worker_submit_started`（payload: 原数据/净化数据的路径+sha256、
  `decontamination_report`、training_mode、config 路径）→（backend 崩溃时 `worker_submit_crashed`）
  → `worker_submit_finished`（payload: `status`, `score`, `gpu_script_id`）；外加
  `experiment_registered`（节点注册时）与 `experiment_updated`（`update_experiment` 时）。
- **文件系统**：建 `experiments/<id>/` + `artifacts/`；**进程内去污染**——读 benchmark queries
  （bench_id 分支读 `<benchmark root>/format/<bench_id>/*.jsonl`，否则回落到 benchmark 驱动源），
  写 `artifacts/decontaminated_train.jsonl`（llama_factory）或
  `artifacts/decontaminated_train.parquet`（slime，且把 `slime_config.data_file` 强制指向净化副本）；
  写 `work_order.json`（WorkOrder：experiment/worker/benchmark/objective/hypothesis/改变变量/
  config 路径/原数据路径）；写占位 `result.json`（status=running, score=None, eval_status=pending,
  `submit_result` 带 gpu 元数据、`artifacts` 带 train_data/decontaminated_train_data/checkpoint/
  eval/tensorboard/pipeline_script/gpu_log 条目）。
- **外部**：`SubmitPipelineBackend.submit` → `SubmitRequest` → 进程内 `run_trial` → 渲染
  `pipeline.sh`（train→serve→eval）→ 提交 gpu_backend :8989 → 返回 `{"status":"running",
  "gpu_script_id": "sc_...", ...}`。backend 未配置 → status=failed（拒跑）；backend 抛异常 →
  status=failed + `worker_submit_crashed`。
- **返回**：`experiment_id` / `status` / `score` / `gpu_script_id` / `result_json_path` /
  `decontamination_report` + hint（slime 覆盖警告 / 预检警告也塞进 hint，不阻止）。

#### `worker add-reviewer --worker-id <id> --experiment-ids <eid>... [--objective]`

- **state.json**：`counters.reviewer` +1 → `rev_NNNN`；`graph.reviewers[rev_NNNN]` 注册
  `ReviewerNode{status:"pending", worker_id, reviewed_experiment_ids, objective}`；图里连上
  worker↔reviewer 边。
- **文件系统**：建 `reviews/<id>/` + `reports/`；写 `config.json`（ReviewerNode 镜像）。
- **events.jsonl**：`reviewer_registered`（payload: `reviewed_experiment_ids`）+ `cli_invoked`。
- **错误**：worker 或任一 experiment_id 不存在 → `KeyError`。
- **语义**：只**建节点**。cc_with_expos 没有 dispatcher，不会自动派 reviewer agent——闭环
  由 CC 自己接着用 `reviewer` 组动词完成。

#### `worker finalize --worker-id [--status finished] [--notes]` / `state finalize-worker --worker-id [--status aborted|failed] [--notes]`

- 两条命令共享 `WorkerService.finalize`：**先 `sync_experiment_results(worker_id)`**（学回该
  worker 名下已终结实验的分数，否则 finalize 完分数还停在 running/None 变成孤儿节点），再
  `update_worker(status, updated_at)`。
- **state.json**：`graph.workers[<id>].status` → `finished`/`aborted`/`failed`（终结态）。
- **文件系统**：重写 `worker/<id>/config.json`；若给了 `--notes` 在 `HISTORY.md` 追加
  `## Finalize (<status>)` + notes。**区别**：`worker finalize` 默认 `status=finished`、空 notes；
  `state finalize-worker` 默认 `status=aborted`、notes=`"Aborted via CLI `state finalize-worker`."`，
  是 abort 专用入口。
- **events.jsonl**：`worker_updated` +（有学回时）`experiment_synced` + `cli_invoked`。

#### `state sync [--worker-id]`

- **state.json**：对每个“磁盘 result.json 已终结（completed/failed）但 DAG 未终结”的实验
  `update_experiment(status=finished/failed, score=...)`（含 `worker.best`/`state.best` 上卷）；
  再经 `DataPoolService.reconcile()` **重建 `data_pool_index`**（覆盖绕过 service 手工创建的
  数据目录，消除 data 孤儿）。
- **文件系统**：重写对应 `experiments/<id>/config.json`；重建 `data_pool/data.jsonl`。
- **events.jsonl**：每个被学回的实验一条 `experiment_synced`（payload: `disk_status`, `score`）
  + `cli_invoked`。
- **这是反孤儿命令**——pipeline 异步写 `result.json` 后 DAG 不知道，sync 把它学回来
  （`result.json` 是权威源；`eval/metrics.json` 的 accuracy 只作 result.json 未终结时的兜底）。

#### `reviewer submit-review --reviewer-id <rid> --experiment-id <eid> --report <file.json>`

- **CLI 前置**：读 `--report` JSON → **强制用 CLI 的 `reviewer_id`/`experiment_id` 覆盖文件里
  的值**（文件忘写也对）→ 校验为 `ReviewReport` 结构。
- **文件系统**：写 `reviews/<rid>/reports/<eid>.json`（ReviewReport dump）；重写
  `reviews/<rid>/config.json`（`verdict_summary[eid]={verdict, primary_bottleneck, confidence}`，
  **status 置为 "running"**）。
- **state.json**：图里连 reviewer↔experiment 边（`link_reviewer_to_experiment`）；reviewer 节点
  的 `verdict_summary`/`status` 更新。
- **数据记忆推荐**（report 带 `data_memory_recommendation.target_data_id` 时）：把推荐应用到
  `data_pool/<id>/config.json`（`promotion_status` → promoted/discarded、`what_to_preserve`/
  `what_to_change` 落盘）、更新 `data_pool_index` + `data.jsonl`；target 已不存在 → 只发
  `data_memory_recommendation_target_missing`。
- **events.jsonl**：`review_registered`（payload: `verdict`, `confidence`）+ `reviewer_updated`
  +（推荐时）`data_memory_recommendation_applied`/`target_missing` + `cli_invoked`。
- **错误**：reviewer/experiment 不存在 → `KeyError`；report 非 JSON/结构非法 → `emit_error`。

#### `reviewer finalize --reviewer-id [--status finished]`

- **state.json**：`graph.reviewers[<rid>].status` → `finished`。
- **文件系统**：重写 `reviews/<rid>/config.json`。
- **events.jsonl**：`reviewer_updated` + `cli_invoked`。

#### `init`（seed 已做，agent 一般不用）

cc_with_expos 的 DAG 在 seed 时由 `ExperimentOS(...).initialize(...)` **直接 Python 调用**建好
（发 `os_initialized` 事件），所以 agent 开场不需要跑 `./expos init`。若手滑跑它：`--overwrite`
会**重置 state.json**（危险）；不传 `--overwrite` 时幂等（已存在就跳过）。

### 3.3 同步缺口：谁把分数学回 DAG

submit 是 async 的——pipeline 的 EXIT trap 覆盖 `experiments/<id>/result.json`，但**没有任何
东西自动写回 `state.json`**。DAG 里实验停在 `running`/score=None，直到以下任一触发
`sync_experiment_results` 把它学回来：

| 触发点 | 范围 |
|---|---|
| `state sync [--worker-id]` | 指定 worker / 全部 |
| `worker finalize` / `state finalize-worker` | 该 worker |
| `state inspect-worker --worker-id` | 该 worker |
| `reviewer inspect-experiment --experiment-id` | **全部**（无过滤） |

所以 agent 想在 REVIEWER 阶段看真实分数，`reviewer inspect-experiment` 或先跑一次
`state sync` 是必要的——否则读到的还是 submit 时的占位 `result.json`。这正是
“score 同步缺口”在 CLI 版里的解法（v2 框架版靠 `update_experiment` 工具，这里靠 sync 动词）。

## 四、worker submit 全链路（agent 触发一次提交时发生什么）

`./expos worker submit` 是 **async** 的。agent 拿回 `status=running` + `gpu_script_id` 就继续干活，
轮询时用 `public/gpu-backend script-status --script-id <id>`。一次 submit 的完整链路：

```
./expos worker submit --worker-id <id> --data-path data/train.jsonl \
    --training-config-path config/train.yaml --training-mode llama_factory_sft
        │
        ▼  [expos.cli.main]   （CLI 形式 ExpOS，同框架子命令树）
WorkerService.submit（tools/ExpOS/expos/service/worker_service.py）
  1. worker 存在性校验 + 路径绝对化
  2. LoRA 硬拒（llama_factory_lora_sft 在下线白名单外，argparse choices 层就挡掉）
  3. run_contract 校验：training_mode ∈ state["run_contract"].allowed_training_modes
  4. 进程内去污染：bench_id 分支读 bmk_root/format/<bench_id>/*.jsonl 的 queries，
     写 experiments/<eid>/artifacts/decontaminated_train.jsonl|parquet —— backend 只见净本
  5. 落 work_order.json + 占位 result.json（status=running, score=None）
  6. 注册 ExperimentNode(running) 进 state 图
        │
        ▼  SubmitPipelineBackend.submit(...)   ← tools/cc_rollout_expos/expos_cli_backend.py 注入
SubmitRequest(experiment_id, worker_id, benchmark, train_data_path=…净本…,
              training_config_path, training_mode, gpu_nums, need_exclusive_gpu=True,
              bench_id, eval_all, slime_config…) 
        │
        ▼  进程内 run_trial(request)（backend/backend/submit_pipeline/driver.py）
渲染 pipeline.sh（train → serve → eval）→ 提交 gpu_backend :8989
        │
        ▼  async 返回 {"status": "running", "gpu_script_id": "sc_...", ...}
agent 轮询：pipeline 的 EXIT trap 最后覆盖 experiments/<eid>/result.json 为真实 score
```

**关键差异**：不经过 ExpOS FastAPI HTTP 服务（:8990），也没有外部 main-agent-loop——
`WorkerService.submit` 的落盘原样发生，backend 换成进程内 `submit_pipeline.run_trial`。
整个机制细节见 [`expos_cli.md`](expos_cli.md)。

## 五、reviewer 闭环（protocol §5–6）

score 只告诉你“失败多少”，轨迹告诉你“为什么”。agent 扮演 reviewer 时：
`inspect-experiment` 翻 `metrics.json` / `samples.jsonl`（逐样本输出）/ `run_eval.log` / `vllm.log`，
判断是训练没学到（生成 ≠ 训练所教）还是格式漂移（生成 = 训练所教但 grader 判 0）；
把诊断**喂回数据**（缺技能加技能样例、格式全崩加格式锚点）。收敛前必须写 review：
`submit-review --report <file.json>`（写 `reviews/<rid>/reports/<eid>.json`），
并明确决定下一步：修 top bottleneck 重跑，或剩余预算不值得再跑就停。

## 六、FINISH 语义（同 naive，但内容语义不同）

- `echo <experiment_id> > FINISH` —— 指向你最好的实验，submit + 结束 run；
- `touch FINISH` —— 不提交任何实验，直接结束。
协议强调 FINISH 是“分析完成”才写的最后一个动作；host 侧读到 marker 即 `done_marker` 平滑回收
（见 [`README.md`](README.md) 的退出四态）。

## 七、硬规则（SUBMIT_GUIDE §6 + 协议）

- **无 LoRA** —— 只有 `llama_factory_sft`（`finetuning_type: full`）或 `slime_sft`。
- **禁设 `CUDA_VISIBLE_DEVICES`**（YAML 或 shell）—— gpu_backend scheduler 注入。
- **禁手改 `experiment_os/state/state.json`** —— 结构改动一律走 `./expos` 动词。
- **禁把 eval/test 题放进训练数据**（泄漏）。
- **run 的 pipeline 还在跑时别重提交** —— 先读 `experiments/<id>/result.json`。
