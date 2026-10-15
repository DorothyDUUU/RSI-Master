# SUBMIT_GUIDE — how to submit a training+eval experiment correctly

> 给 worker agent 的提交说明书。本文档讲清:一次 `worker_submit` 之后框架到底
> 做了什么 (管道是怎么组建的)、训练数据怎么准备/怎么被验证、不同 `training_mode`
> 的超参数该写到哪里。**提交前请先读一遍本文。** 内部实现的逐层细节见
> `tools/ExpOS/expos/backends/pipelines/README.md` (你不需要读完才能提交,
> 但想知道"框架动了什么"时去那里查)。

---

## 1. One config channel, interpreted per mode

`worker_submit(worker_id, data_path, training_config_path, training_mode, ...)`
只有一个配置通道:**`training_config_path`**。不同的 `training_mode` 按各自语义
导入同一个字段:

| `training_mode` | `training_config_path` 是什么 | 超参写在哪 |
|---|---|---|
| `llama_factory_sft` | 一份**完整的 LLaMA-Factory 训练 YAML** (绝对路径), 必须 `finetuning_type: full` (全参 SFT; **LoRA 已下线**) | 全部写进这个 YAML (框架不再接受 dict 覆写) |
| `slime_sft` | 一份 **slime 配置文件 (YAML 或 JSON)** | 键映射到 `launch_sft.sh` 的 env, 见 §4 |

框架会**帮你解析** `training_config_path`:
- llama_factory 模式: 原样透传给 `llamafactory-cli train`。
- `slime_sft`: 解析成 dict, 与 run 级默认合并 (见 §4.2), 覆盖框架托管默认键时
  返回里带 `slime_config_warnings`, **但不阻止** —— 你对训练方向有最终决定权。

---

## 2. What the framework does when you submit (pipeline assembly)

`worker_submit` 是**异步**的。提交后框架按下面顺序组装一条自足的 pipeline,
然后把 GPU 工作交给 gpu_backend, 立即返回 `experiment_id` + `gpu_script_id`:

1. **路径绝对化** — `data_path` / `training_config_path` 以调用进程 CWD 为基准解析。
2. **读 state** — `state["benchmark"]` / `state["eval_all"]` (bench_id, 决定 stage-3
   测哪个基准) / `state["slime_config"]` (slime_sft 的 run 级默认)。
3. **数据污染检查 (最关键的自动净化)** — 加载当前 benchmark 的评测 query,
   逐行比对训练数据, 把归一化后文本与 query **精确相等**的行整行剔除, 写一份
   **净化副本**到 `experiments/{id}/artifacts/decontaminated_train.{jsonl,parquet}`。
   **原始训练文件永不修改**, 后端只拿净化副本。
4. **落盘登记** — 写 `work_order.json`, 注册 `ExperimentNode(status=running)`,
   发 `worker_submit_started` 事件; 写一份 `status=running, score=None` 的占位
   `result.json` (训练/评测结束后由 pipeline 覆盖成终值)。
5. **渲染 pipeline.sh** — `build_pipeline_script` 按 `training_mode` 组装一份
   `experiments/{id}/pipeline.sh`:
   - llama_factory 路径: stage-0 数据预检 → stage-1 全参 SFT (LoRA 已下线, 无
     merge 步骤) → stage-2 vLLM serve (vllm_env_0819, 训练产物原样 serve) →
     stage-3 评测 → EXIT trap 写 `result.json`。
   - slime 路径: stage-0 数据预检 (格式校验) → stage-1 `launch_sft.sh` 训练 →
     epilogue (`status=completed, eval_status=skipped, score=null`)。
   **不修改你提交的 YAML** — 训练落盘位置等用 `llamafactory-cli` 的 key=value
   CLI 覆盖; slime 完全靠 env 覆盖, 不碰 slime 源码/安装。
6. **委托 GPU** — 通过 ExpOS FastAPI backend 提交到 `gpu_backend`, 由调度器选最闲的卡
   并把 `CUDA_VISIBLE_DEVICES` 注入训练环境。**返回 `status=running` + `gpu_script_id`。**

之后: 用 `gpu-backend script-status --script-id <id>` / `gpu-backend script-log
--script-id <id> --tail 200` 轮询; 结束后读 `experiments/{id}/result.json`
拿到终态 `status` / `score` / `failure_reason`。**别在 trial 还在跑时重复提交。**

---

## 3. Training data: prepare it, and what gets validated

| `training_mode` | 输入格式 (`data_path`) | 净化产物 | 污染比对文本 |
|---|---|---|---|
| `llama_factory_sft` (全参; LoRA 已下线) | Alpaca **JSONL**: 每行 `{"instruction", "input", "output"}` | `artifacts/decontaminated_train.jsonl` | 行的 `input` (空则回退 `instruction`) |
| `slime_sft` | **parquet**, 含 OpenAI chat 列 `messages` (每行一个 `[{role,content}, ...]`) | `artifacts/decontaminated_train.parquet` | 最后一条 **user** 轮次的 content |

两条**验证闸门** (都在真正烧 GPU 之前):

- **污染检查 (decontamination, 提交侧, 自动)** — 命中 benchmark 评测 query 的行被
  剔除到净化副本。**这是硬边界: 永远不要把评测集的题目原文/近似题放进训练数据**
  (`GLOBAL_MEMORY.md` 里也有同样的警告)。命中判定是归一化后**精确相等** (NFKC +
  空白折叠, 大小写敏感)。
- **数据预检 (stage-0, GPU 入队后跑, 快速失败)** —
  - llama_factory: `llama_factory_data_preflight.py` 用真实 `DatasetAttr` 校验
    `dataset_info.json` 的列名注册与数据格式。
  - slime: `slime_data_preflight.py` 校验 parquet 可读、有 `messages` 列、每条是
    非空 OpenAI chat 数组、行数 > 0。格式非法立刻 `exit`, 不烧 GPU。

---

## 4. Hyper-parameters, per mode

### 4.1 `llama_factory_sft` (全参 SFT, 唯一 llama_factory 训练方式)

所有超参都写进 `training_config_path` 指向的 LLaMA-Factory YAML。参考模板:
`backend/backend/llama_factory/config/train_sft_gsm8k_example.yaml` (带完整中文
注释)。至少需要: `model_name_or_path` (用 run 的 base model)、`stage: sft`、
`do_train: true`、`dataset` (数据集名, 需在 run 的 `llama_factory/data/
dataset_info.json` 里注册)、`template` (匹配模型族; base 模型下 `default` / `qwen3` / `qwen3_nothink` 均可 —
   0819 已证模板不是分水岭, 模板-eos 预检只做咨询性提示, 不再拒绝)、`cutoff_len`、
`finetuning_type: full` (**全参 SFT; LoRA/QLoRA 已下线, 提交侧会硬拒
`lora`/`qlora`**)、`per_device_train_batch_size`、`gradient_accumulation_steps`、
`learning_rate`、`num_train_epochs`、`lr_scheduler_type`、`warmup_ratio`、`bf16`。

框架会覆盖/接管下面这些 (你写了也会被 CLI 覆盖, 不用管):
- `output_dir` → 实验 workspace 的 `checkpoints/`; `report_to=tensorboard`;
  `overwrite_output_dir=true`。
- `dataset_dir` → 强制指向 run 自带的 `public/backend/llama_factory/
  llama_factory/data` (source tree 只读); 净化副本会被拷进 workspace 并就地生成
  `dataset_info.json`, 把 `file_name` 改写成 `decontaminated_train.jsonl` ——
  **dataset 名不变, 文件指向净化副本**。

> 参考模板里有 "MULTI-GPU TRAINING: set CUDA_VISIBLE_DEVICES=0,1,2,3" —— **这条
> 对本框架无效且是禁区**: 卡由 gpu_backend 调度, worker 绝不能设
> `CUDA_VISIBLE_DEVICES`。要多卡只传 `gpu_nums`。

**训练 YAML 声明 `deepspeed:` 时的启动契约 (2026-08-25):** LLaMA-Factory 对带
`deepspeed` 的配置拒绝直接启动 (`ValueError: Please use FORCE_TORCHRUN=1 to launch
DeepSpeed training.`)。管线 stage-1 渲染期检测训练 YAML 的 `deepspeed` 键并自动注入
`FORCE_TORCHRUN=1` + `NPROC_PER_NODE=<gpu_nums>` (launcher 自包 torchrun) —— 提交侧
**只传 `gpu_nums` 即可, 不用手动加环境变量**。`gpu_nums>=2` 真走 torchrun nproc=gpu_nums
分布式训练 (ZeRO); `gpu_nums=1` 也放行但打 WARN —— 不并行, 只有 ZeRO-offload 省显存的
意义。无 `deepspeed` 的普通全参 SFT 零注入, 脚本与旧版逐字节一致。

**LoRA 已下线 (2026-08-20):** `training_mode` 只有 `llama_factory_sft` / `slime_sft`;
训练 YAML `finetuning_type` 必须 `full` (提交侧会硬拒 `lora`/`qlora`)。

> **`training_mode` 必填, 无默认值 (2026-08-28):** 提交侧 (tool/CLI/service/SubmitRequest)
> 全部要求显式声明 `training_mode`, 漏传会被拒绝 —— 防止静默走 llama_factory_sft。

### 4.1.1 两条 0819 实证教训 (选超参前先读)

1. **全参 SFT 才能学会在回合末发 stop token —— 这正是 LoRA 下线的主因。**
   0819 六模板对比 (GSM8K limit=200): 同数据同模板下 **full-SFT 三个模板全
   200/200 stop、0.80-0.83**; 历史 LoRA (rank16/1ep) 三个模板几乎全部退化
   length-loop、0.01-0.56 (分数可能被答案提取器从循环文本里 grep 到 `ANSWER:`
   而**虚高**, 不代表模型好)。差别是训练容量, 不是模板。LoRA 已下线, 训练只走
   `finetuning_type: full`; 想省显存只能降数据行数/epochs。
2. **发现循环输出 (`finish_reason=length` 主导 / response 超长 / `ANSWER:` 复读),
   先自检 eos-token 对齐, 再下结论。** 跑
   `python backend/backend/llama_factory/scripts/check_eos_alignment.py
   --checkpoint-dir <ckpt> [--samples <samples.jsonl>]`:
   - `tokenizer.eos` 是词表里能发的 token → 循环是**训练侧** (数据/epochs/容量
     没教会模型发 stop token), 不是 infra;
   - `tokenizer.eos` 不在词表 → **infra** (模型发不出自己的终止符), 该修训练侧。
   注意: tokenizer 与 config/generation_config 的 eos 分叉 (qwen3 模板 =
   `<|im_end|>` 151645 vs base `<|endoftext|>` 151643) 在当前 serve 路径
   (vllm 0.27 + `--generation-config vllm`) 下**无害**, 不用为此改模板。

### 4.2 `slime_sft` — the slime config file

`training_config_path` 是一份 **YAML 或 JSON** 配置文件, 键直接映射到
`launch_sft.sh` 的 env。示例 (`slime_config.yaml`, Qwen3-4B 双卡 dense):

```yaml
model_dir: "<repo>/public/models/Qwen/Qwen3-4B-Base"
torch_dist_dir: "<repo>/public/models/Qwen/Qwen3-4B-Base_torch_dist"
model_spec: "$SLIME_ROOT/scripts/models/qwen3-4B.sh"   # 稠密必须传 spec
num_epoch: 3
num_gpus: 2
tp_size: 2          # 4B dense 默认 TP=2×EP=1×CP=1
ep_size: 1
cp_size: 1
seq_length: 32768   # 4B/8B 默认; 35B 用 262144 (截断窗口, 别低于最长训练行)
max_tokens_per_gpu: 32768
rollout_batch_size: 64   # 默认 64; 小数据集(<batch 行)必须显式传
global_batch_size: 64
has_nvlink: 0
```

| 键 | → env | 必填/默认 (2026-08-28 infra 实证) |
|---|---|---|
| `model_dir` | `MODEL_DIR` | **必填** (HF base 权重目录; 缺了 render 期直接 ValueError) |
| `torch_dist_dir` | `TORCH_DIST_DIR` | **必填** (Megatron torch_dist 权重, 训练 `--load` 用; 必须先跑 `convert_hf_to_torch_dist.sh` 转换, 否则 GPU 上会 ERROR) |
| `model_spec` | `MODEL_SPEC` | 默认空 (=内置 35B MoE 架构); **稠密模型必须传** `slime/scripts/models/<model>.sh` |
| `num_epoch` | `NUM_EPOCH` | 默认 `3` |
| `num_gpus` | `NUM_GPUS` | 默认 = `worker_submit.gpu_nums` (35B→4/2, 8B→1, 4B→2) |
| `tp_size`/`ep_size`/`cp_size` | `TP_SIZE`/`EP_SIZE`/`CP_SIZE` | 默认按尺寸+卡数: 大模型 4 卡 `2/2/1`、2 卡 `1/2/1`; 小模型 1 卡 `1/1/1`。硬约束 `TP×EP×CP == gpu_nums` (launch 前校验) |
| `seq_length` | `SEQ_LENGTH` | 默认按尺寸: 4B/8B `32768`, 35B `262144`。**截断窗口** —— 超过最长训练行就静默截断, 精确答案 benchmark 分数断崖 |
| `max_tokens_per_gpu` | `MAX_TOKENS_PER_GPU` | 默认同 `seq_length` |
| `rollout_batch_size` | `ROLLOUT_BATCH_SIZE` | 默认 `64` (档位默认; render 兜底 128)。⚠️ **小数据集必须显式传** (<batch 行 → `num_rollout_per_epoch=0` → assert) |
| `global_batch_size` | `GLOBAL_BATCH_SIZE` | 默认同 `rollout_batch_size` |
| `vocab_size` | `VOCAB_SIZE` | 默认 `0` (不传; 去 Megatron embedding padding) |
| `output_dir` | `OUTPUT_DIR` | 默认 `workspace` |
| `ckpt_dir` | `CKPT_DIR` | 默认 `workspace/checkpoints` (torch_dist ckpt 落这里) |
| `hf_dir` | `HF_DIR` | 默认 `workspace/hf_model` (stage-1.5 torch_dist→HF 转换落点, serve 用) |
| `vllm_tp_size` | (serve) | 默认 `1`; MoE 大模型 (35a3b) serve 用 `4` (任务独占 4 卡, TP4 用满) |
| `vllm_ready_timeout` | (serve) | 默认 `2400` (35B MoE + 256K max-model-len 冷启动 >600s, 2026-08-28 实证) |
| `vllm_tool_call_parser` | (serve) | 可选覆盖 serve 的 `--tool-call-parser`; 缺省按模型家族: **qwen3-8b→`hermes`, 35a3b→`qwen3_coder`** |
| `add_missing` | (转换) | 默认 `true` (stage-1.5 转换 `-a`; EP 分片专家安全网, 已训练权重优先) |
| `skip_eval` | (eval) | 默认 `false` (slime 已接 serve/eval, result.json 拿真分; true 才退回只训练) |
| `data_file` | `DATA_FILE` | **框架强制指向净化副本, 不要传** |

**合并优先级** (低→高): run 默认 (`state["slime_config"]`, 由 run 配置的顶层
`slime:` 段 seed) < 你这份配置文件 < 内联 `--slime-config` (仅 CLI, 调试用)。

**覆盖告警, 不阻止**: 框架托管默认键 `model_dir` / `torch_dist_dir` /
`output_dir` / `ckpt_dir` 被你的配置覆盖时 (相对 run 默认), submit 返回里带
`slime_config_warnings` 数组 + 提示, 但**照常提交**。看到告警先自查是否真的有意的。

slime 已接 **serve/eval** (2026-08-23): 训练委托 `launch_sft.sh` → stage-1.5 把
torch_dist ckpt 转成 HF safetensors (`hf_model`) → stage-2 vLLM serve → stage-3
评测, `result.json` 拿真分 (与 llama_factory 同一条 stage-2/3 管线)。torch_dist
ckpt 同时保留在 `ckpt_dir`。若配了 `slime_config.skip_eval: true`, 才退回只训练
不评测 (`eval_status=skipped, score=null`)。

**stage-1 跑的是 run-local 拷贝 (ESTALE 硬化, 2026-08-24)**: 管线先把整个 `scripts/`
拷到 `$WORKSPACE/scripts` 再执行 `launch_sft.sh` —— 训练期间改共享 NFS 上的脚本不再
影响正在跑的 run (每 run 拷贝时取最新, 想给下一 run 优化就改源处)。launch_sft.sh
内部 `cd "$(dirname "$0")"` + 相对 `source ./activate_env.sh`, 管线是**整目录拷贝**;
你手动拷脚本时也必须是整目录, 别只拷单文件。

---

## 5. After the trial finishes — self-report

trial 结束 (`result.json` 终态) 后, 向 main-agent 汇报时**必须自行填写**三个参数
(用 `gpu-backend script-log/status` 查, 绝不编造):
1. `checkpoint_path` — 产物 ckpt 位置;
2. `eval_score` — `result.json` 里的 score (slime_sft 已接 serve/eval, 除非配了
   `skip_eval` 否则也是真分);
3. `data_path` — 本次用的训练数据 (净化副本在 `artifacts/decontaminated_*`)。

---

## 6. Hard rules (violations break the run)

- **绝不使用 LoRA** —— `training_mode` 只有 `llama_factory_sft` (全参) / `slime_sft`;
  训练 YAML `finetuning_type` 必须 `full` (LoRA/QLoRA 已下线 2026-08-20, 提交侧
  会硬拒 `lora`/`qlora`)。
- **绝不设 `CUDA_VISIBLE_DEVICES`** (训练 YAML、shell wrapper 里都不行)—— GPU 调度
  归 gpu_backend。
- **绝不手改 `state.json`** —— 一切 DAG 变更走 ExpOS 工具 / `python -m expos.cli`。
- **绝不把评测集题目放进训练数据** —— 会被污染检查剔除, 更不该有这条心思。
- **`data_file` (slime) / `dataset_dir` (llama_factory) 由框架接管**, 别对着干。
- **提交前确认 `training_config_path` 存在且是当前 mode 的正确格式** —— 解析失败
  会在提交侧报错, 而不是到 GPU 上才炸。
- **同一 trial 还在跑就别重交** —— 等 `gpu-backend script-status` 到终态再决定。

---

## 7. When the submit_judge gate is on: `worker_submit` may return REJECT

当 run 的 `rsi_master.submit_judge.enabled: true` 时, 每次 `worker_submit` 真正派发
GPU 前会有一个 **submit_judger** (main-agent 的只读拷贝) 判定本次实验对 run 目标
是否**关键且必要**。这是**可选项, 默认关**; 开启后可能出现下面这种返回:

```json
{ "status": "rejected", "tool": "worker_submit", "gate": "submit_judge",
  "verdict": "REJECT", "source": "judge", "reason": "...", "report_path": "..." }
```

请这样处理:
- **REJECT 不是瞬态失败** —— 它发生在任何 GPU 调度之前, **没有烧卡时也没有烧
  experiment id**。`status:"rejected"` 与 `status:"running"`/错误是两回事, 别把它当
  提交失败无限重试同一份。
- **先读 `report_path` 里那份报告** (judge 写的 REJECT reason / detail)。
- **实质改变数据 / 配置 / 假说后再提一份不同的实验**, 或如果这个想法确实到头了就
  `worker_finalize`。
- 提交时**尽量填满 `objective` / `hypothesis` / `changed_variable`** —— judge 和
  DAG 报告都靠它们判断是否冗余; 全空既难判, 也让你没法解释"为什么这次值得跑"。
- 判定失败 / 超时 / 报告缺失时 gate **默认放行** (`fail_policy: allow`), 不会把你
  的提交卡死。
