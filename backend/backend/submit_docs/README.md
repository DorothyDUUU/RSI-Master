# submit_docs — 提交文档 / 预检脚本 / 示例配置 一览

`worker_submit` (与 `worker submit` CLI) 是 agent 提交一次 SFT 训练的唯一入口。
本目录收纳**面向 agent 的提交材料** + **Qwen3-4B-Base 三份示例配置** + **主机验证 harness**。

> 这些文件大部分是**副本/镜像**: 真正被执行的是原处文件 (见下表「源」)。
> 副本的唯一用途是把 agent 需要的材料集中到一处, 避免它去翻源码。

## 目录结构

| 文件 | 是什么 | 源 (真正执行处) |
|---|---|---|
| `SUBMIT_GUIDE.md` | **给 agent 的提交说明书**: 一次 `worker_submit` 后框架做什么、数据怎么被验证、各 training_mode 超参写哪、硬规则 | `playground/rsi_master/SUBMIT_GUIDE.md` (setup 时 seed 进 workspace) |
| `PIPELINE_INTERNALS.md` | pipeline 内部逐层修改机制 (提交侧/渲染侧/GPU 侧/运行时) | `tools/ExpOS/expos/backends/pipelines/README.md` |
| `TROUBLESHOOTING.md` | **常见踩坑 && 排查指南** (slime 为主): 按管线顺序组织, 现象→根因→修法, §0 快速定位表 | — |
| `preflight/llama_factory_data_preflight.py` | LLaMA-Factory 训练数据预检 (stage-0, 真实 DatasetAttr 校验 dataset_info.json) | `backend/backend/llama_factory/scripts/llama_factory_data_preflight.py` |
| `preflight/slime_data_preflight.py` | slime 训练数据预检 (stage-0, 校验 parquet 的 messages 列格式) | `backend/backend/slime/scripts/slime_data_preflight.py` |
| `configs/llama_factory_sft_qwen3_4b_full.yaml` | 示例: **llama-factory-sft** (全参 SFT; LoRA 已下线) | — |
| `configs/slime_sft_qwen3_4b_single_gpu.yaml` | 示例: **slime 单卡训练** | — |
| `configs/slime_sft_qwen3_4b_multi_gpu.yaml` | 示例: **slime 多卡训练** (TP=2×EP=1) | — |
| `validate_on_host.py` | **主机验证 harness**: 在 `gpu-host` (gpu-host, 4×H20) 端到端验证上面三份配置在 Qwen3-4B-Base 上的效果 | — |

## 关键参数速查 (2026-08-28 infra 实证)

worker 提交 slime_sft 时最该盯的几个键(详细语义见 §2.2/§2.3):

| 参数 | 怎么设 | 不设会发生什么 |
|---|---|---|
| `model_tier` | 显式声明 `"4b"/"8b"/"35b"` | 档位默认+预检不生效;靠 render 兜底(仍安全,但没预检告警) |
| `seq_length` / `max_tokens_per_gpu` | 4B/8B 默认 `32768`, 35B 默认 `262144`;换长文本/长 CoT 数据先跑 `check_seq_length.py` | 超过最长训练行静默截断 → 答案尾部丢失 → 精确答案 benchmark 分数断崖 |
| `rollout_batch_size` / `global_batch_size` | 默认 `64`;**小数据集(数据行 <64)必须显式传**(如 8/16/32) | 数据行 <batch → `num_rollout_per_epoch=0` → launch assert, 白烧一轮 |
| `tp_size`/`ep_size`/`cp_size` | 4B/8B 单卡 `1/1/1`;35B 4卡 `2/2/1`、2卡 `1/2/1` | 配错 `TP×EP×CP≠gpu_nums` → render 拒绝 / launch 前退出 |
| `gpu_nums` | 35B→4 (真训练) 或 2 (内存验证);8B→1 | 35B 传 1 → HARD 拒并引导传 4;传 8 → 硬拒 |
| serve 解析器 flag | **自动追加**, 不用配 | `qwen3-8b→hermes`, `35a3b→qwen3_coder`; 想覆盖用 `vllm_tool_call_parser` |
| `vllm_ready_timeout` | 默认 `2400` | 35B MoE 冷启动 >600s, 默认已够 |

**两条硬线**(违反即整 run 崩): ① 绝不设 `CUDA_VISIBLE_DEVICES`(卡由 gpu_backend 调度);
② 绝不手改 `state.json` / 正在跑的 pipeline 脚本。

## 四份配置与 training_mode 的对应

| 配置 | `worker_submit` 用法 | 验证口径 |
|---|---|---|
| `llama_factory_sft_qwen3_4b_full.yaml` | `training_mode=llama_factory_sft` (全参 SFT, LoRA 已下线), 文件当 LLaMA-Factory 训练 YAML 原样透传 | 真实 pipeline.sh: 预检 → 全参 SFT → vLLM serve → gsm8k 小样本评测 → `result.json` |
| `slime_sft_qwen3_4b_single_gpu.yaml` | `training_mode=slime_sft`, 文件当 slime 配置文件解析 (键 → launch_sft.sh env) | `launch_sft_smoke.sh`, NUM_GPUS=1 TP=1 EP=1, 断言 SMOKE_PASSED |
| `slime_sft_qwen3_4b_multi_gpu.yaml` | 同上 | `launch_sft_smoke.sh`, NUM_GPUS=2 TP=2 EP=1, 断言 SMOKE_PASSED |

**slime 4B 口径**: 框架 `slime_sft` 委托的 `launch_sft.sh` 是 35B MoE demo (架构参数
硬编码); Qwen3-4B-Base 是**稠密**模型, 正确入口是 `launch_sft_smoke.sh`
(source 官方 `qwen3-4B.sh` spec)。两种脚本吃同一套 env (MODEL_DIR / TORCH_DIST_DIR /
DATA_FILE / NUM_GPUS / TP_SIZE / EP_SIZE / HAS_NVLINK / OUTPUT_DIR / CKPT_DIR),
所以示例配置的键直接驱动 smoke 入口。

## seq_length & CP pre-check (每次 slime_sft 提交前必做)

**脚本**: `backend/backend/slime/scripts/check_seq_length.py`

```bash
python backend/backend/slime/scripts/check_seq_length.py \
    --data-file <训练数据.parquet> \
    --model-dir <slime_config.model_dir>
```

输出: 人类可读报告 + YAML 配置片段, 可直接粘贴进 slime 配置 YAML。

**为什么需要这个 pre-check**:
- `seq_length` 是 Megatron-LM 训练时每条样本的最大 token 数。超长样本被截断 → 答案尾部丢失 → 精确答案 benchmark (GSM8K/AIME) 分数断崖。
- 默认 `seq_length=4096` 对 gsm8k 足够 (最长样本 ~550 token), 但换成长文本/长 CoT 数据集时必须重新检查。
- 如果 `seq_length > 4096`, 建议开启 CP (context parallelism)。CP 把序列维度分到多张卡上, 减少每卡的激活内存。开 CP 后必须保证 `TP × EP × CP = NUM_GPUS` (launch_sft.sh 会校验)。

**配置示例**:
```yaml
slime_config:
  seq_length: 4096      # 推荐值, 每次提交前用 check_seq_length.py 确认
  max_tokens_per_gpu: 4096  # 必须与 seq_length 一致
  # cp_size: 1          # 默认 CP=1 (seq_length<=4096 时最优); 超过时脚本会推荐

## 训练时间轴 (耐心点, 别误杀)

slime 的 stage-1 在真正开始迭代前要依次做: 启动单节点 ray 集群 → Megatron 模型初始化
(35B MoE full-SFT 还要分配 ~843GiB RAM 的 CPU-offload 优化器) → rollout 数据准备。
**从 stage-1 提交到第一条 loss 日志实测 8-12 分钟甚至更久 (8B/35B 都这样)**, 期间
`pipeline.log` 停在 ray job 提交、`train.log` 停在 `RolloutManager import ...` ——
**这是正常的, 不是卡死**。确认还活着: `gpu-backend script-status` 是 running 且
`train.log` 在增长。stage-1 结束的标志: `stage-1: slime SFT finished`。

**别改正在跑的脚本 (ESTALE 硬化, 2026-08-24)**: 管线 stage-1 已把整个 `scripts/` 拷到
run 本地 (`$WORKSPACE/scripts`) 再执行 `launch_sft.sh`, 训练对共享 NFS 上的并发编辑
免疫 (机制见 PIPELINE_INTERNALS.md 第 4 层)。但**手动路径没有这层保护** —— 训练进行中
绝不改写 `launch_sft.sh` / pipeline 脚本, 否则收尾 `Stale file handle`, 训练成功被误标
failed。

**llama_factory 的 deepspeed 启动契约 (2026-08-25)**: 训练 YAML 声明 `deepspeed:` 时,
管线渲染期自动注入 `FORCE_TORCHRUN=1` + `NPROC_PER_NODE=<gpu_nums>`, launcher 自包
torchrun —— 提交侧只传 `gpu_nums` 即可, 不用手动加环境变量; 单卡放行但打 WARN
(不并行)。机制见 PIPELINE_INTERNALS.md 第 2 层「deepspeed 配置」。

## 运行主机验证

前置: `gpu-host` 的 SSH 别名可用; 与本地共享 `$SHARED_ROOT`
NFS volume (本地渲染的 pipeline.sh 在主机上路径一致, 无需拷文件)。

```bash
cd <repo>

# 只渲染 (不 SSH / 不烧 GPU) — 本地自检
python backend/backend/submit_docs/validate_on_host.py --dry-run

# 全量验证 (含 slime 前置的 torch_dist 转换, 约 10-30 分钟)
python backend/backend/submit_docs/validate_on_host.py --convert

# 只验某个配置
python backend/backend/submit_docs/validate_on_host.py --only full
python backend/backend/submit_docs/validate_on_host.py --only slime-multi
```

ℹ️ **slime 验证的进程纪律**: `launch_sft_smoke.sh` 只用 `ray stop --force` 清理自己的
ray 集群, **禁止 pkill 无差别杀 python** (见 prompts_0812 公共 prompt 的「进程管理
纪律」)。harness 在 slime 案例前打印节点进程/GPU 作只读观察。

验证产物 (pipeline.sh / result.json / ckpt) 落在 `<repo>/runs/submit_docs_validation/<ts>/`
(gitignore); torch_dist 转换产物在 `public/models/Qwen/Qwen3-4B-Base_torch_dist/`
(public/ 亦 gitignore)。

## 一致性说明

`SUBMIT_GUIDE.md` / `PIPELINE_INTERNALS.md` / `preflight/*` 与源文件逐字节一致
(由 `test/test_rsi_master_submit_docs.py` 断言)。改文档/预检脚本请改**源处**,
本目录只是镜像。

## Training Mode — 两种训练模式的完整解释

框架目前只有两种 `training_mode`(2026-08-20 起 LoRA/QLoRA 已整体下线,`lora`/`qlora`
提交在渲染期被硬拒,只剩全参路径):**`llama_factory_sft`** 和 **`slime_sft`**。
两者共享同一条**配置通道**(`training_config_path`)+ 同一条**评测管线**
(stage-2 vLLM serve + stage-3 eval,`_build_eval_config` 共用),差别在**训练引擎**与
**配置文件语义**。

### 0. 先看这张对照表

| 维度 | `llama_factory_sft` | `slime_sft` |
|---|---|---|
| 训练引擎 | LLaMA-Factory (`llamafactory-cli train`) | slime + Megatron-LM (`launch_sft.sh` → ray job → `train_async.py`) |
| `training_config_path` 是什么 | 一份**完整 LLaMA-Factory 训练 YAML**,原样透传给 CLI | 一份 **slime 配置文件**(YAML/JSON),键映射到 `launch_sft.sh` 的 env |
| 数据格式 (`data_path`) | Alpaca **JSONL**,每行 `{"instruction","input","output"}` | **parquet**,含 OpenAI chat 列 `messages`(`[{role,content},...]`) |
| 训练产物 | HF safetensors checkpoint(**直接可 serve**) | Megatron `torch_dist` ckpt(`iter_*/__*.distcp`,**vLLM 不能加载**) |
| 是否需离线转换 | 否 | **是** — stage-1.5 转成 HF safetensors 才能 serve |
| 模型加载 | 直接加载 HF base | `--load` 必须是 **torch_dist** 权重(先跑 `convert_hf_to_torch_dist.sh`) |
| 并行度 | 单进程 HF trainer,`per_device_train_batch_size` 控显存 | ray 集群,`TP×EP×PP(=1)×CP(=1) == NUM_GPUS` 硬约束 |
| 超参写哪 | YAML 里的标准 LLaMA-Factory 字段 | `slime_config` 键(见 §2.2) |
| 默认产物 | `workspace/checkpoints/` (HF) | `workspace/checkpoints/` (torch_dist) + `workspace/hf_model/` (HF) |

### 1. `llama_factory_sft` — LLaMA-Factory 全参 SFT

#### 1.1 配置语义

`training_config_path` 是一份**完整的 LLaMA-Factory 训练 YAML**(如
`configs/llama_factory_sft_qwen3_4b_full.yaml`)。框架**不解析、不改写**它,原样传给:

```bash
llamafactory-cli train <YAML> \
    output_dir=<workspace>/checkpoints \
    logging_dir=<workspace>/tensorboard \
    report_to=tensorboard \
    overwrite_output_dir=true \
    dataset_dir=<run 副本的 llama_factory/data>
```

即:超参(learning_rate / epochs / cutoff_len / batch ...)全部由 YAML 决定,框架只通过
CLI key=value **覆盖落盘位置**和 **`dataset_dir`**(强制指向本次 run 在
`workspace/public/` 里的 llama-factory 副本,挡掉对 source tree
`backend/backend/llama_factory/llama_factory/data` 的写入)。

#### 1.2 数据与预检

- **数据格式**: Alpaca JSONL。每行 `{"instruction": ..., "input": ..., "output": ...}`。
- **去污染**: 命中评测 query 的行在提交侧剔除,净化副本写进 workspace 再训练。
- **stage-0 预检**: `llama_factory_data_preflight.py` 用**真实 `DatasetAttr`** 校验
  `dataset_info.json` 里注册的列名 —— 在 stage-1 占住 GPU 之前拦截注册错误。
- **stage-0b 模板-eos 预检**: `llama_factory_template_preflight.py` 检查训练模板与
  serve 侧 tokenizer 的 eos 对齐(0819 后已降为**咨询性**,只打印不拒绝,可被
  `eval_all.skip_template_preflight: true` 跳过)。

#### 1.3 硬规则

- **`finetuning_type` 必须 `full`**。`lora`/`qlora` 在渲染期被拒(`_reject_lora_finetuning`
  守卫,失败于 `POST /submit` 前)。
- `dataset` 名必须注册在 **run 副本**的 `dataset_info.json`(绝不写 source tree)。
- 产物是 HF checkpoint,stage-1 后直接当 `SERVE_DIR`,无转换步骤。

### 2. `slime_sft` — slime / Megatron-LM 训练

#### 2.1 前置条件(缺了 render 期直接 `ValueError`)

```yaml
model_dir:      /.../Qwen3-8B             # HF base 权重,训练 --hf-checkpoint 用
torch_dist_dir: /.../Qwen3-8B_torch_dist  # 必须存在! --load 用;先跑 convert_hf_to_torch_dist.sh
```

两个都必填。`torch_dist_dir` 缺失时 **render 期抛错**,不等到 GPU 上(避免训练到一半才发现)。

#### 2.2 配置键(全部映射到 `launch_sft.sh` 的 env)

框架把 `slime_config` 展开成 `launch_sft.sh` 的 env,不重实现训练、不碰 slime 源码。
**合并优先级(低 → 高)**:档位默认 `expos_training_recommendations.profiles.*` ← run 默认
(`state["slime_config"]`) ← 配置文件 (`training_config_path`) ← 内联 `--slime-config`。
档位默认只补缺 + 纠正 run 级旧默认的尺寸不匹配键 —— 但**必须显式声明
`slime_config.model_tier: "4b"/"8b"/"35b"`** 才命中 (不做路径提取);未命中档位时走
**render 兜底** (下表「render 兜底」列, 按模型尺寸自动推断, 2026-08-28 起 4B/8B=32768、
35B=262144, 不再是 81920)。完整键表:

| 键 | 档位默认 (4b/8b/35b) | render 兜底 | 说明 |
|---|---|---|---|
| `model_dir` | **必填** | **必填** | HF base 权重(`--hf-checkpoint`);缺了 render 期 ValueError |
| `torch_dist_dir` | **必填** | **必填** | torch_dist 权重(`--load`);缺了 render 期 ValueError |
| `model_spec` | 空(=内置 35B MoE 架构) | 空 | **稠密模型必须传** `slime/scripts/models/<model>.sh`;MoE 留空用内置 35B 默认 |
| `model_tier` | — | — | **显式声明档位** `"4b"/"8b"/"35b"` 才命中档位默认+预检;不传 → 不预检、render 兜底 |
| `data_file` | `request.train_data_path` | 同左 | parquet 训练数据(**框架强制指向净化副本**,别传) |
| `output_dir` / `ckpt_dir` | `workspace` / `workspace/checkpoints` | 同左 | 训练中间产物 / torch_dist ckpt 落点 |
| `num_epoch` | 3 | 3 | 训练轮数 |
| `num_gpus` | `request.gpu_nums` | `request.gpu_nums` | ray 分配卡数 == world_size |
| `tp_size`/`ep_size`/`cp_size` | 4b/8b: `1/1/1`;35b: `2/2/1`(4卡) / `1/2/1`(2卡) | 同左 (按尺寸+卡数) | 并行度;硬约束 `TP×EP×CP == gpu_nums`(render 校验,配错 launch 前退出,不污染 GPU) |
| `has_nvlink` | 0 | 0 | `NCCL_NVLS_ENABLE`;H20 无 NVLink 桥接置 0 |
| `rollout_batch_size` | 64 | 128 | ⚠️ **小数据集必须显式传**,否则数据 <batch 行 → `num_rollout_per_epoch=0` → AssertionError |
| `global_batch_size` | 64 | 128 | 建议与 rollout 对齐 |
| `seq_length` | 32768/32768/262144 | 4B/8B 32768, ≥13B/MoE 262144 | **截断窗口**;超过最长训练行静默截断 → 答案尾部丢失 → 精确答案 benchmark 分数断崖。提交前跑 `check_seq_length.py` 复核 |
| `max_tokens_per_gpu` | 32768/32768/262144 | 同 `seq_length` | 必须与 `seq_length` 一致 |
| `vocab_size` | 0(不传) | 0 | 去 Megatron embedding padding(传真实 HF vocab,无 padding 时 no-op) |
| `vllm_tp_size` | 1;35b `4` | 1 | stage-2 serve 的 TP 拆分;MoE 大模型(35a3b)用 4 (任务独占 4 卡, TP4 用满) |
| `vllm_ready_timeout` | 2400(35b) | 2400 | stage-2 就绪轮询上限 (秒);35B MoE + 256K 冷启动实测 >600s |
| `vllm_tool_call_parser` | 空(自动按家族) | 空(自动按家族) | 覆盖 stage-2 serve 的 `--tool-call-parser`;**qwen3-8b→`hermes`, 35a3b→`qwen3_coder`**;另有 `--enable-auto-tool-choice --reasoning-parser qwen3` 自动追加 |
| `add_missing` | true | true | stage-1.5 转换时 `-a`:只补 ckpt 缺失的权重(EP 分片专家安全网,已训练权重优先) |
| `skip_eval` | false | false | true 时退回只训练+转换,`eval_status=skipped`(默认已接 serve/eval,不要轻易关) |

#### 2.3 并行度契约(最重要的一条)

```
world_size = TP × EP × PP(=1) × CP(=1) == NUM_GPUS
```

- **尺寸感知默认 (2026-08-28 起)**: 并行度按模型尺寸+卡数自动补缺, 不再固定 TP=2/EP=4:
  - 4B/8B dense (只允许 **1 卡**): `TP=1×EP=1×CP=1`。
  - 35a3b MoE (只允许 **2 或 4 卡**): 4 卡 `TP=2×EP=2×CP=1`, 2 卡 `TP=1×EP=2×CP=1`。
  - 卡数越界会被 render 守卫 **硬拒** (8B 传 2 卡 / 35B 传 1 或 8 卡 → 提交即 ValueError)。
- **稠密模型**(如 Qwen3-4B/8B):无 MoE,`ep_size=1`;GQA `num_query_groups=2` 限
  `tp_size ≤ 2`。
- **MoE**(如 35a3b):`ep_size` 必须整除专家数(256)。
- **CP (context parallelism)**: 透传 `SLIME_CP_SIZE`, 但当前档位默认恒 `cp=1` (CP>1 的
  Megatron 路径尚未验证); 长序列换 CP 前先等 infra 验证。
- 配置配错(TP×EP×CP≠NUM_GPUS)时 `launch_sft.sh` **在 ray 启动前**就退出,不污染 GPU。

#### 2.4 训练流程(launch_sft.sh 内部)

1. conda 激活 slime env → 前置检查(`DATA_FILE` 存在 + `TORCH_DIST_DIR` 存在)。
2. **ray 集群隔离**:8 个固定随机端口(gcs/dashboard/agent/client/object/node/
   dashboard_agent_grpc/runtime_env,全部 >20000 互斥)+ worker 区间抬到所有固定端口
   之上 + 唯一 `temp-dir`(WORKSPACE 哈希)。并行任务绝不共享端口/temp-dir —— 这是
   2026-08-23 两个 8B 并行撞车的直接教训(见 §4)。
3. `ray start --head` → 等 job agent 就绪(并行集群刚起时未注册,直接 submit 会
   HTTP 500;只重试**提交级**失败,`train.log` 出现 "Running entrypoint" 后绝不重试)。
4. `ray job submit train_async.py`(`--debug-train-only` = 离线 SFT,只跳过 SGLang
   引擎不跳过训练),训练日志落 `output_dir/train.log`。

#### 2.5 stage-1.5:torch_dist → HF 转换(serve 的硬前提)

- slime 训练产物是 Megatron `torch_dist`(iter_*/__*.distcp),**vLLM 不能加载**。
- `convert_torch_dist_to_hf.py`(slime conda env 解释器,纯 CPU 张量操作,不占 GPU):
  `--input-dir <iter_dir> --output-dir <hf_model> --origin-hf-dir <model_dir>`
  (复制 tokenizer/config/generation_config + 推断架构)+ `--vocab-size`(去 padding)
  + `-a`(补缺失权重)。转换产物落 `workspace/hf_model/`,成为 `SERVE_DIR`。
- **转换前必须释放 GPU**:`launch_sft.sh` 只在开头 `ray stop --force`、结尾**不停**
  Ray,head 进程持续占满 `NUM_GPUS` 张卡。stage-1.5 先 `pkill -9 -f <RAY_TMPDIR>`
  (按唯一 temp-dir 精确清理,绝不 `ray stop`/`pkill python` 无差别杀——那会误杀
  并行任务),否则 stage-2 vLLM serve 必然撞 Ray 占卡 OOM。

### 3. 共享阶段(stage-2/3,两种 mode 同一份代码)

- **stage-2 vLLM serve**:`vllm_env_0819`(vllm 0.27 / transformers 5.15,读得懂训练
  产物的 rope_parameters 格式)+ serve 前 `vllm_serve_guard.py` 对齐校验(0819 根因:
  训练 5.8 写新式 rope 格式,旧 env 4.57 读不懂 → 静默回落 rope_theta 差 100×)。
  端口 `_pick_vllm_port`:`eval_all.vllm_port` 显式覆盖 > 随机化 `18100-18899`
  (随机化是 2026-08-23 修并发 combo 撞端口的方案,见 §4)。
  **解析器 flag (2026-08-28)**: 自动追加 `--tool-call-parser <p> --enable-auto-tool-choice
  --reasoning-parser qwen3`(按模型家族: **qwen3-8b→`hermes`, 35a3b→`qwen3_coder`**;
  可被 `slime_config.vllm_tool_call_parser` 覆盖)。就绪轮询上限 `vllm_ready_timeout` 默认
  **2400s**(35B MoE 冷启动 >600s)。
- **stage-3 eval**:`request.bench_id` 非空 → benchmark suite `eval_all` 统一评测
  (`--bench-id`,limit 默认 500,**绝不传 -1**);否则按 `request.benchmark` 分立评测。
  judge 类 bench(healthbench_easy / arena_hard_writing)走 `evaluation.eval_all.judge`
  凭据。评分 `eval_status=completed` → `result.json` 写真分。

### 4. 注意事项 / 常见坑(2026-08-23 真机实证)

1. **并发 slime 任务必须端口隔离**:vLLM 端口 `_pick_vllm_port` 已随机化,但同一实验
   重跑/恢复时建议显式传 `eval_all.vllm_port` 保持稳定;ray 的 8 个端口全随机互斥 +
   worker 区间抬升已在 `launch_sft.sh` 内做死 —— 手动改脚本别破坏这两处。
2. **seq_length 已尺寸感知, 但 batch 兜底仍是雷 (2026-08-28)**:render 兜底
   `seq_length`/`max_tokens_per_gpu` 4B/8B=32768、≥13B/MoE=262144, 不再 81920 导致
   8B 单卡 OOM;但 `rollout_batch_size`/`global_batch_size` **兜底仍 128** —— 小数据集
   (数据行 <batch)跑默认 rollout batch 必然 AssertionError (`num_rollout_per_epoch=0`)。
   **小数据集必须显式配 `rollout_batch_size`** (建议 ≤ 数据行数, 一般 8/16/32)。
3. **MoE serve 需 TP** :35a3b 转换后 serve 用 `vllm_tp_size=4`(TP=4 拆参,
   任务独占 4 卡时 serve 用满 4 卡);dense 单卡不拆。
4. **失败后清孤儿 ray**:ray 进程 cmdline 含 temp-dir,精确清理用
   `pkill -f <RAY_TMPDIR>`;别 `pkill -9 python`(节点上还有并行任务/其它服务)。
5. **35B MoE full-SFT 内存峰值 ~843GiB** vs pod cgroup 880GiB:`launch_sft.sh` 已设
   `RAY_memory_usage_threshold=0.99`,**别降回去**(默认 0.95 会在 save_model 收尾时
   误杀 worker)。
6. **worker 绝不设 `CUDA_VISIBLE_DEVICES`**:卡由 gpu-backend 按 `gpu_nums` 调度
   (need_exclusive_gpu 独占整卡);设置了会和调度器打架。
7. **eval 端口 404 排查**:如果 stage-3 全是 `404 .../v1/chat/completions`,先查
   并行任务是否撞了同一 vLLM 端口(guard 的 `/v1/models` 200 可能是邻 server 答的),
   再查 `vllm.log` 里有没有 "Uvicorn running"(engine init 完但被 cleanup 杀掉 →
   serve 根本没监听)。这是 2026-08-23 slime-8b-nothink 的完整根因链。
8. **恢复/重跑**:训练 ckpt + `hf_model` 都在时,只需重新 serve + eval 不用重训
   (独立端口 + 独立 `trial-name`,避免撞原实验输出目录)。

### 5. 怎么选

- 要**标准 HF 生态**(模板丰富、YAML 直写、产物直接 serve)→ `llama_factory_sft`。
- 要 **Megatron 级并行**(大 MoE、多卡 TP/EP)、或跑 **slime 数据管线**
  (messages parquet + 训练/推理统一数据)→ `slime_sft`(代价:torch_dist 前置转换 +
  转换回 HF 才能 serve)。
- 两者评测完全相同;选型只影响**训练侧**。

