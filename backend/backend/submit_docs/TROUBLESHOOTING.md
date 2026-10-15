# 常见踩坑 && 排查指南 (slime 为主)

> 本文档整理 slime_sft 管线全阶段的踩坑与排查方法 (2026-08-20 ~ 08-28 真机实证)。
> 按管线顺序组织: **提交前预检 → stage-0 数据 → stage-1 训练 → stage-1.5 转换 →
> stage-2 serve → stage-3 eval → 平台/运维**。每条 = 现象 → 根因 → 修法。
> 配置语义/合并优先级见 `SUBMIT_GUIDE.md`; 内部逐层机制见 `PIPELINE_INTERNALS.md`。

---

## 0. 快速定位 (症状 → 根因 → 修法)

| # | 症状 | 根因 | 修法 |
|---|---|---|---|
| 0.1 | 提交即 `ValueError` "只能用 gpu_nums=4 … 或 2" | 35B 传了 `gpu_nums=1`/`8` | 35B→`4` (真训练) 或 `2` (内存验证); 8B→`1` |
| 0.2 | 提交即 `ValueError` "llama-factory 大模型…" | 用 llama_factory_sft 跑 35B/MoE | 大模型必须走 `slime_sft` |
| 0.3 | stage-1 报 `num_rollout_per_epoch=0` | 数据行数 < `rollout_batch_size` (render 兜底 128) | 小数据集显式配 `rollout_batch_size` ≤ 行数 |
| 0.4 | 训练跑起来但 `--load` ERROR | `torch_dist_dir` 缺失/未转换 | 先跑 `convert_hf_to_torch_dist.sh`, 配好 `torch_dist_dir` |
| 0.5 | 提交后 stage-1 前 8-12 分钟无 loss | **正常静默期** (ray 启动 + Megatron init + CPU-offload) | 看 `train.log` 是否增长; 别 kill |
| 0.6 | ray `ValueError` 秒挂 / "No available agent" | ray 端口撞 (agent 固定 52365 / random 撞 worker 区间) | 8 端口全显式随机 + worker 区间抬到固定端口之上 |
| 0.7 | 35B save_model 收尾被杀 (无 OOM 硬限) | cgroup 880GiB vs 峰值 843GiB, Ray 0.95 阈值误杀 | `RAY_memory_usage_threshold=0.99` |
| 0.8 | 2 卡 35B "1/2 clients joined" 601s 超时 | torch dist 集结超时 (非 OOM), actor 串行创建争 CPU | 并发创建 actor + timeout |
| 0.9 | 训练结束但 `Stale file handle` → stage-1 failed | 训练中改了共享 NFS 上的脚本 | 别再改运行中脚本; 管线已整目录拷 run-local |
| 0.10 | serve 后 eval 复读撞 max_tokens (stop 不下来) | eos 错配 (训练发 im_end, serve 认 endoftext) | stage-1.5 已加 eos align patch; 旧 ckpt 手动 patch |
| 0.11 | stage-3 全 404 `/v1/chat/completions` | 并行任务撞同一 vLLM 端口 (guard 200 是邻 server) | 查 `vllm.log` "Uvicorn running"; 端口已随机化 |
| 0.12 | eval `finish_reason=length` 主导, 分数低 | `max_tokens` 太短, 输出被截断 (非模型失败) | 抬 `evaluation.eval_all.max_tokens` (起 16k) |
| 0.13 | 同 ckpt 同 prompt 两个 serve 出分天差地别 | rope_theta 配置分裂 (旧 vllm 读不懂新 rope 格式) | serve 用匹配版本的 vllm env |
| 0.14 | 提交 422 拒 slime_sft | 远程 ExpOS backend (8990) 没重启, 旧 schema | 改 tools/ExpOS 后 kill+restart 8990 |
| 0.15 | 35B 2 卡真训练 OOM / 内存验证总失败 | 2 卡全精度 440GiB RAM 硬墙 (数学不可能) | 35B 真训练必须 4 卡; 2 卡只做内存验证 |

---

## 1. 提交前 / 预检 (还没烧 GPU 就拦住)

### 1.1 batch 兜底 128 雷 (2026-08-28)

**现象**: stage-1 直接 `AssertionError: num_rollout_per_epoch=0`。

**根因**: `rollout_batch_size`/`global_batch_size` 有**两个默认**——档位默认 `64`
(命中 `model_tier` 时) vs **render 兜底 `128`** (未命中档位 / worker 没传)。数据行数
< batch 时 `num_rollout_per_epoch = 数据行数 / batch = 0` → launch assert, 白烧一轮。

**修法**: 小数据集 (行数 <64) 必须**显式**配 `rollout_batch_size` (≤ 行数, 一般
8/16/32), 别靠任何默认。预检规则 `data_rows_vs_rollout_batch` (WARNING) 会在 submit
时提示。

### 1.2 model_spec (2026-08-23)

**现象**: 8B dense 训练按 35B MoE 架构跑, 卡数/并行度全对不上, 或直接挂。

**根因**: `launch_sft.sh` 硬编码 35B MoE 架构; **稠密模型 (4B/8B) 必须传
`model_spec: slime/scripts/models/<model>.sh`**。

**修法**: dense 必传 `model_spec`; 35B MoE 留空用内置默认。

### 1.3 seq_length 截断

**现象**: 精确答案 benchmark (GSM8K/AIME) 分数断崖, 训练侧看不出问题。

**根因**: `seq_length` 是**截断窗口**。超过最长训练行的样本被静默截断 → 答案尾部丢失。

**修法**: 提交前跑 `python backend/backend/slime/scripts/check_seq_length.py
--data-file <parquet> --model-dir <model_dir>`, 按输出设 `seq_length` ≥ 最长行。
默认尺寸感知: 4B/8B=`32768`, 35B=`262144`。

### 1.4 gpu_nums 守卫 (2026-08-27)

- **35B**: `gpu_nums=4` (真训练, TP=2×EP=2×CP=1) 或 `2` (内存验证, TP=1×EP=2×CP=1)。
  `1` 或 `8` → **HARD 拒** (单卡放不下 / 8 卡挤占配额槽位)。
- **8B**: `gpu_nums=1`; `2` → 拒 (小模型多卡无意义)。
- **llama_factory_sft**: ≥13B/MoE → **HARD 拒** (35B full-FT 需 ~417GB VRAM, 必 OOM)。

### 1.5 torch_dist_dir (2026-08-23)

**现象**: stage-1 `--load` ERROR。

**根因**: Megatron 训练 `--load` 必须吃 **torch_dist** 权重, 不是 HF base 权重。

**修法**: 提交前先 `convert_hf_to_torch_dist.sh` 生成 `*_torch_dist`, 配好
`torch_dist_dir`。缺了预检会 WARNING 提示。

---

## 2. stage-0 数据预检

- **格式**: slime 数据是 **parquet + OpenAI chat `messages` 列** (每行 `[{role,content},…]`)。
  格式非法 → `slime_data_preflight.py` stage-0 `exit`, 不烧 GPU。
- **污染**: 命中 benchmark 评测 query 的行被自动剔除到净化副本
  (`artifacts/decontaminated_train.parquet`); 原始文件不改。
- **注意**: 接外部数据集需要预构建 messages parquet (`slime_messages_parquet` 旁路,
  infra-test 层), 不传 `SLIME_ROWS`。

---

## 3. stage-1 训练 (最长静默期, 别误杀)

### 3.1 静默期 8-12 分钟 (正常)

提交后到第一条 loss: 启动单节点 ray 集群 → Megatron 模型初始化 (35B full-SFT 额外
分配 ~843GiB CPU-offload 优化器) → rollout 数据准备。`pipeline.log` 停在 ray job
提交、`train.log` 停在 `RolloutManager import ...`。**确认活着**: `gpu-backend
script-status` = running 且 `train.log` 在增长。stage-1 结束标志:
`stage-1: slime SFT finished`。

### 3.2 ray 端口冲突 (双根因, 2026-08-23)

**现象 A**: `ray start` 报 bind 失败 / `No available agent` (HTTP 500)。

**根因 A**: agent 端口固定 52365 → 并行集群冲突。

**现象 B**: `ray start` ValueError **秒挂** (提交后立即崩)。

**根因 B**: 只固定 5 端口, 留 `random` 撞进 worker 端口区间。

**修法**: ray 的 **8 端口全显式随机** (gcs/dashboard/agent/client/object/node/
dashboard_agent_grpc/runtime_env, 全部 >20000 互斥) + worker 区间抬到所有固定端口
之上 + 探活重试。改动落在 `launch_sft.sh`; 手动改脚本别破坏这两处。

### 3.3 2 卡集结超时 (2026-08-28)

**现象**: 35a3b-2gpu 首次提交失败, 日志 `torch dist 601s: 1/2 clients joined`。
**不是 OOM**。

**根因**: slime **串行创建 actor** + Co-Placement CPU 争抢 → 第二个 rank 集结超时。

**修法**: 并发创建 actor + 集结 timeout 放宽 (已落地)。再遇到先确认是集结不是显存。

### 3.4 内存: cgroup 880GiB vs 峰值 843GiB (2026-08-23)

**现象**: 35B full-SFT `save_model` 收尾被杀; `nvidia-smi` 无 OOM, 日志像是 Ray 杀。

**根因**: GPU 主机是 **k8s pod**, `cgroup memory.max=880GiB` (`free` 显示的 2014GiB
是宿主机)。35B full-SFT save_model 峰值 **843GiB**, 顶到 Ray 默认 **0.95 阈值**误杀
(未超硬限)。

**修法**: `RAY_memory_usage_threshold=0.99` (别降回去) + 控制 seq/批次。失败后清孤儿
ray worker。

### 3.5 35B 2 卡全精度不可能 (2026-08-28)

**现象**: 2 卡 35B 内存验证总失败, MEM-DIAG fp32=0。

**根因**: GPU 显存可修 (f=0.85 init 81GB, cycle-1 存活), 但 **440GiB RAM 硬墙**:
`CPU = f×260+221` 约束集为空 (f≤0.68 vs f≥0.84)。

**结论**: **35B 真训练必须 4 卡**; 2 卡只做内存验证。

### 3.6 MoE 专家只按 EP 分片 (2026-08-27)

**现象**: 2 卡 `TP=2×EP=1` → OOM。

**根因**: `launch_sft.sh` 硬编码 `ET=1` (专家线程), **专家数量不随 TP 减** → 全模型
复制到每张卡。

**修法**: `TP=1×EP=2` (2 卡) — 专家按 EP 分片才省显存。4 卡 `TP=2×EP=2`。

### 3.7 ESTALE: 训练中改共享 NFS 脚本 (2026-08-24)

**现象**: 训练其实成功, 但收尾 `Stale file handle` → stage-1 误标 failed。

**根因**: 共享 NFS 上运行中的 bash 脚本被并发编辑。

**修法**: 管线 stage-1 已把整个 `scripts/` **整目录拷到 `$WORKSPACE/scripts`** 再执行
(每 run 拷贝取最新)。**手动路径没有这层保护** —— 训练进行中绝不改写
`launch_sft.sh` / pipeline 脚本; 想优化给下一 run, 改源处。

---

## 4. stage-1.5 转换 (torch_dist → HF)

### 4.1 eos 错配 (2026-08-23, 已修复)

**现象**: 同 ckpt serve 后 eval 复读撞 `max_tokens`, stop 不下来。

**根因**: Qwen3 模板训练发 `<|im_end|>` (151645); 转换继承 base 模型
eos=`<|endoftext|>` (151643); **vLLM 只认后者** → 模型发 im_end 不被当终止符 → 复读。

**修法**: stage-1.5 已加 **eos align patch** (转换产物 eos 对齐训练模板)。
旧 ckpt 手动 patch 后同 ckpt 同 prompt 能正常 stop (117 tokens)。

**判断工具**: `python backend/backend/llama_factory/scripts/check_eos_alignment.py
--checkpoint-dir <ckpt>` — tokenizer.eos 在词表 → 训练侧泛化问题 (数据/容量没教会发
stop); 不在词表 / 与 serve 停止符分叉 → infra。

> 注意: 当前 serve 路径 (vllm 0.27 + `--generation-config vllm`) 下模板 eos 分叉
> **无害**, 不用为此改模板。但 2026-08-28 实证: full-SFT **短 prompt 会发 im_end**,
> eval **10-shot few-shot 2223 token 下不发** → 复读撞 max_tokens —— 这是**训练侧
> 泛化问题**, eos 配置正确, 别去改 infra。

### 4.2 add_missing / EP 分片专家 (2026-08-23)

转换 `-a` (add_missing): 只补 ckpt 缺失的权重。EP 分片专家的安全网 —— **已训练
权重优先**。转换落点 `hf_dir` (serve 用), torch_dist ckpt 保留在 `ckpt_dir`。

---

## 5. stage-2 serve

### 5.1 vllm_ready_timeout: 35B MoE 冷启动 >600s (2026-08-28)

**现象**: 35a3b-256k serve 在 600s 死在 CUDA graph 捕获 (权重加载 ~101s +
torch.compile ~53s + graph 捕获)。

**修法**: `vllm_ready_timeout` 默认 `2400` (别降); serve 轮询超时被误杀时先看是不是
还在冷启动。

### 5.2 parser flags (2026-08-28 真机验证)

stage-2 serve 自动追加 `--tool-call-parser <p> --enable-auto-tool-choice
--reasoning-parser qwen3` (qwen3-8b→`hermes`, 35a3b→`qwen3_coder`)。vLLM 0.27.1
**全量接受**三 flag, 35B serve 出分 **0.9149 = 基线** (新 flags 不改 eval 行为)。
想覆盖用 `vllm_tool_call_parser`。

### 5.3 rope 配置分裂 (2026-08-19)

**现象**: 同 ckpt 同 prompt, 新旧 vllm env 出分天差地别 (0.626 vs 0.01)。

**根因**: LLaMA-Factory (transformers 5.8) 写 **rope_parameters 新格式**; 旧 vllm 0.11
env (transformers 4.57) 读不懂 → **rope_theta 静默回落 10000** (差 100×) → 不是同一个
模型。

**修法**: serve env 与训练产物格式匹配。同 vllm 0.27 逐字节复现输出。

### 5.4 vLLM / transformers 版本 (2026-08-05)

vLLM 0.11 与 transformers 5.x **硬不兼容**。serve env 用与训练产物匹配的组合, 别混。

---

## 6. stage-3 eval

### 6.1 404 / vLLM 端口撞车 (2026-08-23)

**现象**: stage-3 全 404 `/v1/chat/completions`。

**根因**: 并行任务撞同一 vLLM 端口 (例: think/nothink 并行同 exp_0001 撞 18819);
guard 的 `/v1/models` 200 可能是**邻 server** 答的。

**排查**: 查 `vllm.log` 有没有 "Uvicorn running" (engine init 完但被 cleanup 杀掉 →
serve 根本没监听); 再查并行任务端口是否互斥。

**修法**: vLLM 端口 `_pick_vllm_port` 已随机化; 同一实验重跑/恢复建议显式传
`eval_all.vllm_port` 保持稳定。

### 6.2 max_tokens 太短 (分数被低估)

**现象**: `finish_reason=length` 主导 → 分数低。**不是模型失败**。

**修法**: `evaluation.eval_all.max_tokens` 别留 0 (0 → benchmark suite 默认 gsm8k=3000,
aime=16000, 对长 CoT 太短)。**起 16k**, 长 CoT / 大模型更高。`serve_max_model_len`
≥ max_tokens + prompt (默认 32k, 长文本 64k)。

### 6.3 eval_all 轨迹 samples (2026-08-23)

eval_all 只拷 `samples.jsonl` 进 run-dir; 若 runner 不写它 → **零轨迹**。
livecodebench 曾踩过, 已补合并写入器 + 回填。

---

## 7. 平台 / 运维

### 7.1 远程 ExpOS backend 重启 (2026-08-23)

**现象**: 改 tools/ExpOS 后, 远程 8990 提交 slime_sft 报 **422 (旧 schema)**。

**根因**: ExpOS FastAPI backend **不自动重载**。

**修法**: 改 tools/ExpOS 后手动 kill + restart `:8990`。quickstart 脚本若跳过重启,
新字段会被旧 schema 拒。

### 7.2 进程清理纪律

- **绝不 `pkill -9 python` / `pkill -9 ray`** — 节点共享, 会杀掉 gpu_backend 服务、
  其它 run 的训练。
- 清孤儿 ray worker: cmdline 含 temp-dir, 精确用 `pkill -f <RAY_TMPDIR>`。
- 停 ray 集群用 `ray stop --force` (只清 ray 自己的进程)。

### 7.3 日志看哪 (按耗时权威排序)

| 想要 | 看 |
|---|---|
| 训练/转换/serve/eval 阶段推进 | `experiments/<id>/pipeline.stage.log` |
| 训练迭代 (loss / step) | `experiments/<id>/train.log` (ray job 日志) |
| serve 启动 / parser / 端口 | `experiments/<id>/vllm.log` |
| 终态分数 | `experiments/<id>/result.json` (`status`/`eval_status`/`score`) |
| 提交往返 / 后端 | 提交侧 backend 日志 (RUN_LOG_DIR) |
