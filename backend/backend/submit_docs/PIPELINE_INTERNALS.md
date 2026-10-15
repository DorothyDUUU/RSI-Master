# pipelines — submit 工具内部会做什么修改

worker 调一次 `worker.submit(data_path, training_config_path, training_mode, ...)` 之后,
数据 / 配置 / 产物 / 结果会经历**四层修改**。本文档逐层讲清「工具内部到底动了什么」,
并列出**严格不改**的边界。

- 第 1 层:提交侧 —— `WorkerService.submit`(服务进程内,任何 GPU 工作入队之前)
- 第 2 层:渲染侧 —— `build_pipeline_script`(把请求渲染成一份自足的 `pipeline.sh`)
- 第 3 层:GPU 侧 —— `gpu_backend`(提交到 GPU 机器时的环境修改)
- 第 4 层:运行时 —— bash 各 stage 在 GPU 上跑时对产物做的修改

---

## 如何选择训练框架

每次 `worker submit` 用 `--training-mode` 选择训练框架；两种模式共用提交、去污染、
vLLM 部署和 benchmark suite 评测，但输入数据、训练配置和原始 checkpoint 格式不同。

| 场景 | 推荐 | 原因 |
|---|---|---|
| 4B/8B、单卡、短序列（通常 2K–8K）、快速数据实验 | `llama_factory_sft` | 启动简单，短任务端到端开销低，直接产出 HF checkpoint。 |
| 已有 Alpaca JSONL 和 LLaMA-Factory YAML | `llama_factory_sft` | 无需转换数据和基础模型 checkpoint。 |
| 多卡长序列训练，需要 TP/CP/SP | `slime_sft` | Megatron 并行能力更适合把模型或长序列拆到多卡。 |
| 大型 MoE，需要 EP/TP | `slime_sft` | Slime/Megatron 原生暴露专家并行和张量并行。 |

不要仅按模型名称选择：Qwen3-8B 在 H20 上可走两种框架。短序列单卡优先
LLaMA-Factory；只有确实需要长序列并行或多卡扩展时才优先 Slime。比较速度时应固定
模型、数据 token 数、GPU 数和 global batch size，以训练 token/s 为主，不能直接比较
两个不同配置的总运行时间。

当前限制：Slime 的 `CP_SIZE` 已透传 (render `SLIME_CP_SIZE=cfg["cp_size"]`) 但当前
档位默认恒 `cp=1` (parallel_defaults 返回 CP=1, CP>1 的 Megatron 路径尚未验证)；从
release torch-dist checkpoint 开始新 SFT 时还需保证 `start_rollout_id=0`，否则会跳过
首个 rollout。在这两项完成验证前，不应宣称超长上下文训练的 CP 并行已经跑通。

### CLI 示例

```bash
# LLaMA-Factory：Alpaca JSONL + 完整训练 YAML
expos worker submit --worker-id worker_0001 \
  --data-path /abs/train.jsonl \
  --training-config-path /abs/llama_factory.yaml \
  --training-mode llama_factory_sft --gpu-nums 1

# Slime：messages parquet + Slime YAML/JSON
expos worker submit --worker-id worker_0001 \
  --data-path /abs/train.parquet \
  --training-config-path /abs/slime.yaml \
  --training-mode slime_sft --gpu-nums 4
```

| 项 | `llama_factory_sft` | `slime_sft` |
|---|---|---|
| 输入数据 | Alpaca JSONL | Parquet，含 OpenAI `messages` 列 |
| 配置 | 完整 LLaMA-Factory YAML | Slime YAML/JSON，至少含 `model_dir`、`torch_dist_dir` |
| 原始训练产物 | HF checkpoint | Megatron torch-dist checkpoint |
| 评测前处理 | 无 | torch-dist 转 HF |
| 部署与评测 | vLLM → `eval_all` | vLLM → `eval_all` |

---

## 第 1 层:提交侧(WorkerService.submit,进程内)

纯 Python,不碰 GPU。顺序:

| # | 动作 | 修改什么 |
|---|---|---|
| 1 | **路径绝对化** | 把 `data_path` / `training_config_path` 解析成绝对路径(相对路径相对调用进程 CWD)。 |
| 2 | **读 state** | 读 `state["eval_all"]`(拿 `bench_id`)、`state["slime_config"]`(slime 训练配置),合并 worker 侧 `--slime-config` 覆盖项。 |
| 3 | **污染检查(数据修改的核心)** | 加载 benchmark 评测 query → 逐行比对 → 把**净化副本**写到 `experiments/{eid}/artifacts/`,**原始数据文件不动**。 |
| 4 | **落盘登记** | 写 `work_order.json`;注册 `ExperimentNode(status=running)`;发 `worker_submit_started` 事件。 |
| 5 | **委托 backend** | `train_data_path` 传的是**净化副本**路径(不是原始数据);slime 下还把 `slime_config["data_file"]` 强制指到净化副本。 |
| 6 | **写占位 result.json** | 写一份 `status=running, eval_status=pending, score=None` 的占位版,等第 4 层 bash 结束时**覆盖**它。 |

### 3a. 污染检查按 training_mode 分支

| training_mode | 输入格式 | 净化产物 | 比对文本 |
|---|---|---|---|
| `llama_factory_sft`(全参 SFT; LoRA 已下线) | Alpaca JSONL | `artifacts/decontaminated_train.jsonl` | 行的 `input`(空则回退 `instruction`) |
| `slime_sft` | parquet(`messages` 列) | `artifacts/decontaminated_train.parquet` | 最后一条 user 轮次的 content |

- 命中 = 归一化后文本与 benchmark query **精确相等**;命中行整行剔除。
- **不修改**原始训练文件(只读),backend 只拿净化副本。
- `decontaminate_parquet` 用 `table.take` 只过滤不重建,保留原始 schema。

---

## 第 2 层:渲染侧(build_pipeline_script)

把 SubmitRequest 渲染成一份 `workspace/pipeline.sh`(单个 bash 脚本,`EXPERIMENT_ID` 等全部
以变量注入)。**文件本身不落盘到源 tree**,只写到实验 workspace。

### 关键:训练 YAML **不改**,用 CLI 覆盖

`training_config_path` 指向的 LLaMA-Factory YAML **永不原地修改**。训练时用 llamafactory-cli
的 key=value 语法覆盖落盘位置与数据集路径,其余超参(learning_rate / epochs …)由 agent 在
YAML 里事先决定:

```
llamafactory-cli train <TRAINING_CONFIG_PATH> \
    output_dir=<CHECKPOINT_DIR> \
    logging_dir=<TENSORBOARD_DIR> \
    report_to=tensorboard \
    overwrite_output_dir=true \
    dataset_dir=<DATASET_DIR>
```

### deepspeed 配置:渲染期注入 FORCE_TORCHRUN (2026-08-25)

LLaMA-Factory 启动契约 (`hparams/parser.py:353`):配置声明 `deepspeed` 且
`parallel_mode != DISTRIBUTED` 时直接 `ValueError: Please use FORCE_TORCHRUN=1 to launch
DeepSpeed training.` launcher 只在 FORCE_TORCHRUN=1 或可见 GPU>1 时才自动包 torchrun ——
stage-1 直接 `llamafactory-cli train`, 所以**单卡 + deepspeed 配置被拦死** (2026-08-25
exp_0008/0009: deepspeed 补装后第一关过了, 撞上启动关)。渲染期**只读**训练 YAML 的
`deepspeed:` 键 (不原地改文件), 命中就往 stage-1 的 train 命令前注入
`export FORCE_TORCHRUN=1` + `export NPROC_PER_NODE=<gpu_nums>`:

- `gpu_nums>=2`:显式 nproc=gpu_nums, 真分布式训练 (ZeRO), 与 gpu_backend 注入的
  `CUDA_VISIBLE_DEVICES` 一致。
- `gpu_nums=1`:放行 + 打 WARN —— 不并行, 只有 ZeRO-offload 省显存的意义。
- 无 `deepspeed` 键:空注入, 脚本与旧版逐字节一致。

### 数据被「重新登记」到 run 副本

- `DATASET_DIR` 被强制指向本次 run 自带的 llama-factory 副本
  `…/workspace/public/backend/llama_factory/llama_factory/data`(**不是** source tree)。
- 净化副本被拷进 `workspace/llama_factory_data/`,并就地生成一份 `dataset_info.json`,
  把 `file_name` 改写为 `decontaminated_train.jsonl` —— 即训练实际读的 **dataset 名不变,
  文件指向净化副本**。
- 找不到 run 副本时只告警并让「dataset not found」大声失败,**不静默回退到 source tree**。

### 输出路径全部重定向进实验 workspace

| 项 | 默认落点(相对 `experiment_workspace`) |
|---|---|
| checkpoint | `checkpoints/`(slime 可被 `slime_config.ckpt_dir` 覆盖) |
| tensorboard | `tensorboard/` |
| eval | `eval/` |
| result.json | `result.json` |
| pipeline log | `pipeline.log` |

### vLLM serve 参数被推导并写死进脚本

- 端口:`_pick_vllm_port(experiment_id)` —— 用 experiment_id 哈希选 `18100 + (hash % 800)`
  的稳定高位端口,避免多实验踩踏。
- `gpu-memory-utilization` 默认 `0.6`、`max-model-len` 默认 `eval.max_tokens + 4096`
  (eval_all 模式 `32768`);可用 `eval_all.serve_gpu_memory_utilization` / `serve_max_model_len`
  覆盖(大模型场景)。
- **解析器 flag (2026-08-28, 用户要求)**: serve 命令自动追加
  `--tool-call-parser <p> --enable-auto-tool-choice --reasoning-parser qwen3`。
  `<p>` 按模型家族推断 (`render._tool_call_parser`, 依 `model_dir` 目录名):
  **qwen3-8b 系 → `hermes`, qwen3.5/3.6/35a3b → `qwen3_coder`**;可被
  `slime_config.vllm_tool_call_parser` 显式覆盖。`--enable-auto-tool-choice` 强制依赖
  `--tool-call-parser` (vLLM cli_args 校验, 只传前者会报错)。
- **就绪轮询上限 `VLLM_READY_TIMEOUT`**: 默认 2400s (2026-08-28 600→2400) —— 35B MoE
  + 262144 max-model-len 冷启动实测 >600s (权重加载 ~101s + torch.compile ~53s +
  CUDA graph 捕获), 旧默认会误杀。

### 评测派发被改写

- `bench_id` 非空 → stage-3 用 `eval_all.py`(benchmark suite 统一评测),`EVAL_ARGS` 被整体改写
  为 `--bench-id --split --trial-name --bmk-root`,limit 取正数(默认 500,**绝不传 -1**)。
- 否则按 `benchmark` 从 `_BENCHMARK_EVAL` 选 eval 脚本 / metric key / max_tokens / max_workers;
  limit 默认 `-1`(全量)。
- `arena_hard_writing` / `healthbench_easy` 需要 LLM judge:渲染进程里已有的
  `OPENAI_BASE_URL` / `OPENAI_API_KEY` 会被 export 进 pipeline.sh(judge 专用,生成仍走本地 vLLM)。

### slime 路径:env 覆盖,不碰安装/源码

- `slime_config.model_dir` / `torch_dist_dir` **必填**,缺了 render 期直接 `ValueError`
  (不等到 GPU 上才炸)。
- 其余键 `num_epoch` / `num_gpus` / `tp_size` / `ep_size` / `has_nvlink` / `output_dir` /
  `ckpt_dir` / `data_file` 有默认值,全部作为 env 变量注入 stage-1,委托 `launch_sft.sh`。
- **context-length 默认按尺寸兜底 (2026-08-28)**: `seq_length` / `max_tokens_per_gpu`
  render 兜底 = 大模型 (>=13B/MoE) `262144`, 小模型 (4B/8B) `32768` (模型原生
  max_position_embeddings)。`seq_length` 是**截断窗口** —— 超过最长的训练行就静默截断,
  答案尾部丢失会打崩精确答案 benchmark; 换长文本/长 CoT 数据必须先 `check_seq_length.py`
  复核。并行度契约 `TP × EP × CP == gpu_nums` 不变。
- **batch 默认 (2026-08-28 infra 实证)**: `rollout_batch_size` / `global_batch_size`
  档位默认 64 (config.yaml `expos_training_recommendations.profiles.*.default_config`),
  render 兜底 128。**小数据集必须显式传**, 否则数据行 < batch → `num_rollout_per_epoch=0`
  → launch assert。
- **不改** slime 源码 / conda 安装,也不重实现训练。
- **stage-1 跑 run-local 拷贝 (ESTALE 硬化, 2026-08-24)**: `launch_sft.sh` 在共享 NFS
  上, 训练期间被改写会 ESTALE (见第 4 层)。stage-1 先把整个 `scripts/` 拷到
  `$WORKSPACE/scripts` 再执行, 执行进程持有 run-local 文件, 对并发编辑免疫 —— 必须
  **整目录**拷 (launch_sft.sh 内部自定位 cd + 相对 source ./activate_env.sh)。

---

## 第 3 层:GPU 侧(gpu_backend)

- worker **只传 `gpu_nums`**,由 gpu_backend 调度器按剩余显存选卡。
- 调度器把 `CUDA_VISIBLE_DEVICES` 注入 tmux 训练环境,整条 pipeline(train/vLLM/eval)继承。
- **worker 绝不能自己设 `CUDA_VISIBLE_DEVICES`**(train YAML、shell wrapper 里都不能)。
- 提交立即返回 `status=running` + `SCRIPT_ID`,agent 用 `gpu-backend script-status/log` 轮询。

---

## 第 4 层:运行时(bash 各 stage 对产物的修改)

`set -eEuo pipefail`,`EXIT trap` 兜底收尾。

| stage | 对什么做什么修改 |
|---|---|
| **stage-0 数据预检** | GPU 入队前拦截坏数据。LF:`llama_factory_data_preflight.py` 用真实 DatasetAttr 校验 `dataset_info.json` 列名注册;slime:`slime_data_preflight.py` 校验 parquet 的 `messages` 列格式。失败立刻 `exit`,不烧 GPU。 |
| **stage-1 SFT** | `source llama_factory_env` → `llamafactory-cli train`(见第 2 层覆盖; 训练 YAML 声明 deepspeed 时已在渲染期注入 FORCE_TORCHRUN 走 torchrun)。训练结束后 `find` 挑最新 `checkpoint-*` 作 `LATEST_CKPT`,`SERVE_DIR` 恒等于它(全参 SFT 直接可服务; LoRA 已下线)。 |
| **stage-2 vLLM serve** | 单独 subshell `source vllm_env`(`vllm_env_0819`,transformers 5.x,与训练 env 格式兼容,训练产物原样 serve)→ `vllm serve <SERVE_DIR>` 后台起,带解析器 flag(见第 2 层),轮询 `/v1/models`(`VLLM_READY_TIMEOUT` 上限,默认 2400s,2026-08-28 由 600 提升:35B MoE 冷启动可 >600s)。`VLLM_PID` 由 EXIT trap 回收,不留孤儿进程。 |
| **stage-3 eval** | `source llama_factory_env`(有 datasets 模块)→ 跑 `EVAL_PY`。结束后从 `eval/metrics.json` 提取 `METRIC_KEY`(accuracy / pass_at_1)写进 `SCORE`。metrics.json 缺失 → 判失败。 |
| **EXIT trap** | 无论成败,最后用 python 重写 `result.json`:填 `status` / `eval_status` / `score` / `failure_reason`,`artifacts` 指向 checkpoint / tensorboard / eval / gpu_log 四类产物。 |

slime 默认在 stage-1 后把 torch-dist checkpoint 转成 HF，再复用相同的 stage-2/3。
只有 `slime_config.skip_eval=true` 时才直接 epilogue，写入
`STATUS=completed, EVAL_STATUS=skipped, SCORE=null`。

**stage-1 的 ESTALE 硬化 (slime, 2026-08-24)**: `launch_sft.sh` 位于共享 NFS 上 —— 若
训练期间被改写 (给下一 run 优化), 运行中 bash 进程持有旧 inode, teardown 读脚本尾部 →
`Stale file handle` → 训练成功被误标 failed (2026-08-23 平台 4 pod 全踩雷)。stage-1
因此先把整个 `scripts/` 拷到 run 本地 `$WORKSPACE/scripts` (每 run 专属、永不重写) 再
`bash` 执行 —— 执行进程持有 run-local 文件, 对 NFS 并发编辑免疫。launch_sft.sh 内部
`cd "$(dirname "$0")"` + 相对 `source ./activate_env.sh`, 故**整目录拷贝** (只拷
launch_sft.sh 会断 activate_env 链接)。

---

## 严格不变的边界

| 不变项 | 原因 |
|---|---|
| 原始训练数据文件 | 只读,backend 只用净化副本 |
| 训练 YAML(agent 提交的那份) | 覆盖用 CLI key=value,不原地改文件 |
| slime 源码 / conda 安装 | 只在配置层通过 `launch_sft.sh` env 接入 |
| 运行中正被执行的 `launch_sft.sh` / pipeline.sh | 改写会 ESTALE → 训练被误标 failed; 管线已拷 run-local 免疫 (2026-08-24), 手动路径仍要守 |
| `CUDA_VISIBLE_DEVICES` | worker 不设,GPU 调度归 gpu_backend |
| `state.json` | 永不手改,只能走 service 方法持久化 |

## 一次 submit 的产物落盘

```
experiments/{eid}/
  work_order.json            # 第 1 层:提交登记
  result.json                # 第 1 层写占位 → 第 4 层 EXIT trap 覆盖终值
  pipeline.sh                # 第 2 层:渲染出的自足 bash 脚本
  pipeline.log               # gpu_backend script log
  artifacts/decontaminated_train.{jsonl,parquet}   # 第 1 层:净化副本
  llama_factory_data/        # 第 2 层:staging 的净化数据集 + dataset_info.json
  checkpoints/               # 训练 ckpt(全参 SFT, LoRA 已下线 → 直接 serve)
  tensorboard/  eval/        # 训练日志 / 评测输出(含 metrics.json)
```
