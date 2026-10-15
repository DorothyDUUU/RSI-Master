# expos.backends — ExpOS 训练/评测后端 (HTTP 层 + pipeline driver)

worker 的一次 `worker submit`（或 agent tool `worker_submit`）会穿过 `WorkerService`
（进程内服务层）→ 本包（HTTP 后端 + driver）→ `gpu_backend`（GPU 机器上的 tmux 执行）。
本文档讲清楚 **submit 在本包内的完整生命周期**：HTTP 契约、driver 派发、pipeline.sh
渲染、gpu_backend 提交、异步收尾。改了什么、产物在哪、`pipelines/` 里各 stage 做了什么，
见 `pipelines/README.md`（本文只覆盖到「提交进 gpu_backend」为止，运行时的逐 stage 修改
在那一份）。

## 定位

```
+-------------------+   httpx POST /submit   +--------------------+   gpu_backend   +------------------+
| WorkerService     | ---------------------> | expos.backends     | ---------------> | gpu_backend      |
|  (submit 服务层)  |                         |  server.py (FastAPI)|   submit_script  |  tmux 跑 pipeline|
+-------------------+   client.py 实现         +---------+----------+                  +------------------+
  service/                     SubmitBackend 协议         |
  worker_service.py                                      v
                                            pipelines.run_trial()
                                              (render pipeline.sh + 提交)
```

- **`SubmitBackend` 协议**（`expos.service.worker_service.SubmitBackend`）是 Service 与
  Backend 的接缝。本包的 `SubmitBackendClient` 用 HTTP 实现它；测试/CI 里可以注入一个
  stub 实现，Service 层代码不用改。
- **异步契约**：`POST /submit` 提交 pipeline 到 gpu_backend 后**立即**返回
  `status="running"` + `gpu_script_id` / `gpu_log_path`，**不等训练跑完**。最终
  `score` 由 pipeline bash 的 EXIT trap 写进 `experiment_workspace/result.json`，
  `WorkerService.finalize()` 再把它回收进 `state.json`。
- 唯一 driver 名集合：`{"pipelines", "pipeline_llama_factory_gsm8k"}`（后者是旧名
  re-export 壳，`server._run_trial` 双名兼容）。

## submit 生命周期（端到端）

| # | 层 | 动作 | 产物/效果 |
|---|---|---|---|
| 1 | 服务层 `WorkerService.submit` | 校验 worker 存在、路径绝对化、`llama_factory_lora_sft` 已下线拦截 | — |
| 2 | 服务层 | 读 `state`：run benchmark、`eval_all`（拿 `bench_id`）、`slime_config` 默认 | — |
| 3 | 服务层 | **污染检查**：加载 benchmark query → 逐行比对 → 净化副本写到 `experiments/{eid}/artifacts/`，**原始数据不动**；slime 强制 `data_file` 指向净化副本 | `artifacts/decontaminated_train.{jsonl,parquet}` |
| 4 | 服务层 | 落盘 `work_order.json`、注册 `ExperimentNode(status=running)`、发 `worker_submit_started` 事件 | `work_order.json` |
| 5 | 服务层→HTTP | `SubmitBackendClient.submit` → `POST /submit`（网络故障降级成 `status=failed` 记录，不抛异常） | — |
| 6 | server | `POST /submit` 注册 trial → `_run_trial` 按 `config.driver` 派发 | `trial_{hex}` |
| 7 | driver | `pipelines.run_trial` → `build_pipeline_script` 渲染 `workspace/pipeline.sh`（渲染期守卫 + 配置校验） | `pipeline.sh` |
| 8 | driver | `BackendClient.submit_script(pipeline.sh, gpu_nums, max_mem_quota, need_exclusive_gpu, priority=high, background=True)` | `SCRIPT_ID` / `SESSION_ID` |
| 9 | 服务层 | 写**占位** `result.json`（`status=running, eval_status=pending, score=None`）；更新 ExperimentNode；发 `worker_submit_finished` | `result.json`（占位） |
| 10 | 服务层 | 返回给 tool 层：`hint` 引导用 `gpu-backend script-status/log` 轮询 + 读 `result.json` | — |

> 第 7 步失败（如 slime 缺 `model_dir` / `torch_dist_dir`、卡数/并行度非法、llama-factory
> 被拦）→ `run_trial` 返回 `status="failed"` + `failure_reason`，**不碰 GPU**。
> 第 8 步失败（backend 不可达 / 包 import 失败）→ 同样 `failed`，绝不静默成功。

## 模块布局

| 文件 | 职责 |
|---|---|
| `server.py` | FastAPI 应用：`POST /submit`、`GET /trials/{trial_id}`、`GET /health`、`GET /`；`_run_trial` 按 driver 派发 |
| `client.py` | 同步 httpx 客户端 `SubmitBackendClient`，`SubmitBackend` 协议的 HTTP 实现；`build_default_client()` 工厂 |
| `models.py` | Pydantic v2 请求/响应：`SubmitRequest` / `SubmitResponse` / `TrialInfo` / `HealthResponse` |
| `config.py` / `config.yaml` | 后端 host/port/timeout + driver 选择 |
| `cli.py` | `python -m expos.backends` —— server start / health / submit / trial |
| `pipelines/` | driver 包：`build_pipeline_script` + `run_trial` + 各 stage 片段（详见其 README） |
| `pipeline_llama_factory_gsm8k.py` | 旧名 re-export 壳（`DRIVER_NAMES` 双名兼容） |

## SubmitRequest 契约（`models.py`）

| 字段 | 默认 | 说明 |
|---|---|---|
| `experiment_id` / `worker_id` | — | 实验/worker 身份 |
| `benchmark` | `"gsm8k"` | run 的 benchmark 身份（`bench_id` 设置后可为 7 个内建 id 之外的值） |
| `train_data_path` | — | **净化后**的训练数据路径（服务层传的已是副本） |
| `training_config_path` | — | llama_factory 训练 YAML；slime_sft 下按配置文件解析 |
| `training_mode` | `"llama_factory_sft"` | `Literal["llama_factory_sft", "slime_sft"]`；`llama_factory_lora_sft` 给友好已下线报错 |
| `experiment_workspace` | — | 实验 workspace（`experiments/{eid}`） |
| `gpu_nums` | `1` | 需要的卡数；**选卡归 gpu_backend**，提交方不设 `CUDA_VISIBLE_DEVICES` |
| `need_exclusive_gpu` | `True` | 默认独占（2026-08-29）；调度按 `max_mem_quota` 兜底 |
| `max_mem_quota` | `None` | 每卡最低可用显存 (MB) 放置需求，render 期估算；`None`=不限 |
| `bench_id` | `None` | 非空 → stage-3 走 benchmark suite 统一评测（`eval_all`） |
| `eval_all` | `{}` | 评测 knob（limit / max_tokens / split / serve 参数 / judge 凭据…） |
| `slime_config` | `{}` | `slime_sft` 训练配置，键映射 `launch_sft.sh` 的 env |

**SubmitResponse**：`status`（completed/failed/pending/running）+ `score` +
`gpu_script_id` / `gpu_session_id` / `gpu_log_path` + `pipeline_script_path` /
`checkpoint_dir` / `tensorboard_dir` / `eval_dir` / `result_json_path` + `summary` /
`failure_reason`。提交成功即 `status="running"`。

## 服务端 driver 派发（`server._run_trial`）

```python
driver = load_backend_config().driver   # config.yaml: server.driver
```
- **`pipelines`**（默认）/ 旧名 `pipeline_llama_factory_gsm8k` → `pipelines.run_trial`
  （渲染 + 提交 gpu_backend）。
- **`stub`** → 只保证 workspace 存在、返回 `status="pending"` 收据，不接任何训练。
  保留给单元测试 / 冒烟（`test_rsi_master_cli_submit_pipeline.py` 用 stub driver
  boot 整个 FastAPI server）。

## pipelines.run_trial（driver.py）

`run_trial(request)`：

1. `build_pipeline_script(request, workspace)` → 渲染 `workspace/pipeline.sh`
   （模板 `fragments.compose_llama_factory / compose_slime`，单次 `substitute`）。
   渲染期守卫 + 配置校验：
   - `llama_factory_sft` → `_reject_llama_factory`：**无条件拦截**（本集群只跑 slime，
     Co-Placement OOM 修复，2026-08-27）。
   - `slime_sft` → `_reject_bad_slime_gpu_nums`：大模型（≥13B/MoE）只允许
     `gpu_nums ∈ {2, 4}`（4=TP2×EP2×CP1 真训练；2=TP1×EP2×CP1 内存验证），小模型必须 1；
     并行度 `TP×EP×CP == gpu_nums`，否则 launch 断言会失败。
   - `_resolve_slime_config`：`model_dir` / `torch_dist_dir` **必填**（缺了 render 期直接
     ValueError，不等到 GPU 上才炸）；并行度/seq 等按模型尺寸给默认（大 262144 / 小 32768）。
2. `estimate_max_mem_quota(request)`：按训练峰值/卡估算每卡最低 free-MB 放置需求；
   卡总显存由调度器掌握，客户端不再假定 96GB。默认独占保证 serve 获得整卡，
   `eval_all.max_mem_quota` 仍可显式覆盖。
3. import `backend.gpu_backend`（**repo root 必须在 PYTHONPATH**，`backend.gpu_backend`
   是顶级子包；失败返回 `failed` 并给出提示）。
4. `BackendClient.submit_script(command_path=pipeline.sh, timeout=0, run_in_background=True,
   gpu_nums, max_mem_quota, need_exclusive_gpu, priority="high")`。
5. gpu_backend 返回 QUEUED/RUNNING → 翻译成 `status="running"`；`score=None`（异步）。

> slime 路径的 ESTALE 硬化（2026-08-24）：stage-1 把整个 `backend/backend/slime/scripts/`
> 拷到 `$WORKSPACE/scripts` 再执行，避免共享 NFS 上训练中被改写的 `launch_sft.sh` 被
> teardown 读尾部时 `Stale file handle` → 训练成功被误标 failed。

## pipeline.sh 的各 stage（运行时）

`set -eEuo pipefail`，`EXIT trap` 兜底收尾（`fragments.FRAGMENT_HELPERS`）：

| stage | 内容 |
|---|---|
| stage-0 | 数据预检：LF `llama_factory_data_preflight.py`（真实 DatasetAttr 校验列名注册）；slime `slime_data_preflight.py`（parquet `messages` 列格式）。失败立刻 exit，不烧 GPU |
| stage-0b | （仅 LF）模板-eos 预检，**咨询性**（只打印 turn-ender vs base eos 比对，不拦模板；缺脚本降级 warning） |
| stage-1 | SFT：LF `llamafactory-cli train`（deepspeed → 渲染期注入 `FORCE_TORCHRUN`）；slime 委托 `launch_sft.sh`（全部训练超参经 env 注入，不碰源码/安装） |
| stage-1.5 | （仅 slime）torch_dist → HF 转换（`convert_torch_dist_to_hf.py`） |
| stage-2 | vLLM serve：端口随机（`18100 + randrange(800)`，`eval_all.vllm_port` 可覆盖固定），`--gpu-memory-utilization 0.6`，`--max-model-len` 按 eval max_tokens 或模型尺寸兜底（大 262144 / 小 32768），轮询 `/v1/models`（`VLLM_READY_TIMEOUT` 默认 600s，35B MoE 冷启动需 1800） |
| stage-3 | 评测：`bench_id` 非空 → `eval_all.py`（limit 取正数，**绝不传 -1**）；否则按 `benchmark` 从 `_BENCHMARK_EVAL` 选脚本/metric key/max_tokens。judge-based bench（`arena_hard_writing`/`healthbench_easy`）export judge 凭据。`metrics.json` 里提取 `METRIC_KEY` 作 `SCORE` |
| EXIT trap | 无论成败，用 python 重写 `result.json`：`status` / `eval_status` / `score` / `failure_reason` + artifacts 四类产物路径 |

`slime_config.skip_eval=true` → 只训练 + 转换，走 epilogue（`eval_status=skipped`），
省掉 35a3b 那 ~10-30min 的转换/评测开销。

## 集群路径常量（`pipelines/constants.py`）

固定安装点，不接受覆盖：训练 env（`public/llama_factory_env`）、serve env
（`public/vllm_env_0819`）、slime conda / Megatron-LM / `launch_sft.sh`、各 benchmark 的
eval 目录（`backend/backend/eval_*`）、vLLM 端口基准 `18100`。换机器必须同步改这里。

## 配置（`config.yaml`）

```yaml
backend:  {host: localhost, port: 8990}     # base_url = http://{host}:{port}
timeouts: {submit: 7200, status: 30, connect: 10}   # submit 容忍 训练+评测 最坏墙钟
server:
  trials_dir: state/backend_trials          # 相对 ExperimentOS root 解析
  driver: pipelines                         # pipelines | pipeline_llama_factory_gsm8k | stub
```
`load_backend_config(path=...)` 优先级：显式路径 > 包内 `config.yaml`。不走环境变量。

## CLI 用法

```bash
export PYTHONPATH=tools/ExpOS                      # 本包依赖需要 tools/ExpOS 在路径上
PYTHONPATH=tools/ExpOS python -m expos.backends server start            # 起 FastAPI (driver=pipelines)
PYTHONPATH=tools/ExpOS python -m expos.backends health
PYTHONPATH=tools/ExpOS python -m expos.backends submit \
    --experiment-id exp_x --worker-id worker_x --benchmark gsm8k \
    --train-data /tmp/train.jsonl \
    --training-config-path /path/to/train.yaml \
    --training-mode llama_factory_sft --workspace /tmp/exp_x
PYTHONPATH=tools/ExpOS python -m expos.backends trial --trial-id trial_xxx
```

命令行 submit 只是冒烟/调试入口（`--training-mode` 只收 `llama_factory_sft`）；真正被
agent 用的是 `WorkerService.submit` → HTTP。standalone 提交时**记得 repo root 也在
PYTHONPATH**（`backend.gpu_backend` 是顶级子包，driver 第 3 步 import 它）。

## 边界与注意

- **选卡归 gpu_backend**：worker/agent 只传 `gpu_nums`，调度器按剩余显存 + `max_mem_quota`
  选卡并注入 `CUDA_VISIBLE_DEVICES`，整条 pipeline（train/serve/eval）继承。worker 绝不设
  `CUDA_VISIBLE_DEVICES`。
- **异步不丢结果**：提交后 `status="running"`；最终结果只来自 `result.json`（EXIT trap
  写）。轮询用 `gpu-backend script-status --script-id <id>`。
- **原始数据不动**：backend 只拿 `artifacts/decontaminated_train.*`（benchmark query 净化
  副本）；原始训练文件只读。
- **渲染期守卫是硬拦**：llama-factory 整体下线、slime 卡数/并行度非法、slime 缺
  model_dir/torch_dist_dir —— 全部入队前拒绝，不让 agent "试一把" 烧 GPU。
- **repo root on PYTHONPATH**：driver 要 import `backend.gpu_backend`，`backend.` 是顶级包。

## 相关文档

- `pipelines/README.md` — submit 之后 pipeline 内部四层修改 + 严格不变的边界。
- `tools/ExpOS/README.md` / `expos.cli` — ExperimentOS 状态机、worker/reviewer 工具。
- `backend/gpu_backend/` — GPU 侧调度与 tmux 执行。
