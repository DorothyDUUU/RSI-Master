# 本部署：环境激活与训练启动

> 本文是 **PostTrainBenchRollout** 在本机（共享盘 conda）部署 slime 的**具体操作文档**。
> 区别于官方「Docker 镜像开箱即用」方案，本部署直接在共享盘 conda 环境 `slime` 上运行；
> 框架通用概念与全部参数说明见同目录官方文档（如
> [get_started/usage.md](get_started/usage.md)、[get_started/customization.md](get_started/customization.md)）。

> [!IMPORTANT]
> **只改配置，不碰环境与源码。** 详见 `../README.md` 顶部「使用约定」。

## 1. 训练栈与物理布局

| 层 | 组件 | 位置 | 说明 |
|---|---|---|---|
| 训练 | Megatron-LM | `$SLIME_MEGATRON_LM` | 经 `PYTHONPATH` 引入，仅能从仓库根导入 |
| 框架 | slime | `$SLIME_ROOT` | editable 安装，入口 `slime/train_async.py` |
| 推理/rollout | SGLang | `$SLIME_WORKSPACE/sglang` | SFT 离线模式（`--debug-train-only`）不启动引擎 |
| 环境 | conda `slime` | `$SLIME_ENV` | Python 3.12 / PyTorch 2.11+cu129 |
| 权重 | Qwen3.6-35B-A3B | HF `<repo>/public/models/Qwen/Qwen3.6-35B-A3B`；torch_dist `<repo>/public/models/Qwen/Qwen3.6-35B-A3B_torch_dist` | 训练 `--load` 指向 torch_dist |
| 数据 | `xyz_aquila_sft.parquet` | `$SLIME_WORKSPACE/data/xyz_aquila_sft.parquet` | 3412 条，seq≤81920，含 `messages` 列 |

启动入口脚本（本目录 `../scripts/`）：

| 脚本 | 作用 |
|---|---|
| `activate_env.sh` | 激活 slime 环境（source 即用，非交互 shell 安全） |
| `verify_env.sh` | 验证环境 + GPU + 导入链 |
| `cleanup.sh` | 清理残留 ray / sglang / python（重跑前必做） |
| `convert_hf_to_torch_dist.sh` | HF → Megatron torch_dist（单卡一次，10-30 分钟） |
| `launch_sft.sh` | 一键启动 SFT 训练（参数化，默认=跑通值） |
| `launch_sft_smoke.sh` | 端到端冒烟（1 个真实训练步 + 存 ckpt） |
| `watch_training.sh` | 观测训练日志 / ray dashboard |

## 2. 环境激活

非交互 shell 下最可靠的方式（不依赖 `.bashrc` 的 conda hook）：

```bash
source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate slime
```

等价写法（直接改 PATH，不依赖 conda 函数）：

```bash
export PATH="$SLIME_ENV/bin:$PATH"
```

激活后建议设置（与构建时一致，供 JIT / 原生扩展匹配 H100/H200 架构）：

```bash
export CUDA_HOME="$CONDA_PREFIX"
export TORCH_CUDA_ARCH_LIST=9.0
export NVTE_CUDA_ARCHS=9.0
```

一键版：`bash scripts/activate_env.sh`（source 即用）。

## 3. 环境验证

```bash
# 单行冒烟
python -c "import torch, megatron, slime; print('ok', torch.__version__)"

# 完整验证（导入链 + GPU + NVLink 拓扑）
bash scripts/verify_env.sh
```

## 4. 启动训练

> 所有训练都必须在 **8×H20 GPU 节点**上跑（本机 `ptb-cpu-dev-0` 无 GPU）。
> 机器名示例：`gpu-node-0`（`gpu-node`）。

### 4.1 先冒烟，证明全链路

```bash
# 合成 8 条样本（1 个真实训练步 + 存 ckpt）
python $SLIME_WORKSPACE/training_scripts/prepare_smoke_data.py \
  --dst $SLIME_WORKSPACE/training_scripts/data/smoke_sft.parquet --n 8

bash scripts/launch_sft_smoke.sh
# 通过标志：$CKPT_DIR/SMOKE_PASSED
```

### 4.2 正式 SFT

```bash
bash scripts/launch_sft.sh
```

换数据 / 超参（全部支持环境变量覆盖，默认值 = 之前跑通的值）：

```bash
DATA_FILE=/path/to/xxx.parquet \
NUM_EPOCH=2 \
OUTPUT_DIR=<repo>/backend/backend/slime/output \
bash scripts/launch_sft.sh
```

脚本内部流程：激活环境 → 清理残留 → 检测 NVLink → `ray start --head --num-gpus 8` →
`ray job submit` 提交 `train_async.py`（TP=2 / EP=4 / CP=1、seq 81920、`--loss-type sft_loss`）。

进度观测：

```bash
bash scripts/watch_training.sh              # 只看关键行
tail -f <OUTPUT_DIR>/train.log              # 完整日志
# ray dashboard: http://<GPU节点>:8265
```

### 4.3 断点续训

默认 `--no-save-optim --no-save-rng`（只存模型权重，省空间）。续训需：

1. 去掉 `--no-save-optim --no-save-rng`（保存优化器状态）；
2. `CKPT_DIR` 指向新的输出目录，`--load` 指向上一个保存的 ckpt
   （如 `ckpt/Qwen3.6-35B-A3B_sft/iter_0000100`）。

首次运行保持现状即可。

## 5. 数据与权重准备

### 5.1 torch_dist 权重（首次必做）

`--load` **必须**指向转换产物 `<repo>/public/models/Qwen/Qwen3.6-35B-A3B_torch_dist`，不能直接指 HF 目录
（`hf_to_megatron` 只按 ETP 切专家，256 专家张量塞不进 `[num_local_experts]` 参数；
torch_dist 加载时按训练时 TP/EP/PP 自动重分片）。

```bash
bash scripts/convert_hf_to_torch_dist.sh   # 单卡一次，约 10-30 分钟
```

### 5.2 数据格式（slime SFT 要求）

每条数据是一个 parquet 行，必须含 **`messages` 列**（OpenAI chat 格式）：

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "回答", "reasoning_content": "可选，思考"},
    {"role": "assistant", "content": "调用前的话", "tool_calls": [{"function": {"name": "...", "arguments": {...}}}]},
    {"role": "tool", "content": "工具结果"}
  ]
}
```

- `--input-key messages` 读该列，`--metadata-key metadata` 读可选元数据列。
- **reasoning**：assistant 的 `reasoning_content` 字段，或 content 内嵌 `<think>...</think>`；
  渲染为 `<|im_start|>assistant\n<think>\n...\n</think>\n\n{content}`。**loss mask**：`<think>`
  内容不参与训练，只对 `</think>` 之后的部分计算 loss。
- **tool_call**：assistant 的 `tool_calls` 列表，渲染为 `<tool_call>...`；工具结果用
  `role: "tool"` 消息；工具定义可用 `--tool-key tools` 注入 system `<tools>` 块，或直接写进 system 消息。

## 6. 关键参数速查（launch_sft.sh 默认值）

| 项 | 值 | 说明 |
|---|---|---|
| 并行 | TP=2, EP=4, CP=1, SP | `etp×ep×pp = 1×4×1` 整除 world_size=8；TP=4 破坏 GQA（num-query-groups=2） |
| seq-length | 81920 | 140GB/card 安全上限；98304 实测 OOM |
| `--debug-train-only` | **保留** | 离线 SFT 数据路径：只跳过 SGLang 引擎启动/权重同步，**不跳过**训练步与权重保存 |
| loss mask | `--loss-mask-type qwen3_5` | `<think>` 不参与训练 |
| 优化器 | Adam + CPU offload + distributed optimizer | 内存修复：`--main-grads-dtype bf16`、sft 时 fp32_output=False、`--log-probs-chunk-size 4096` |

## 7. 常见问题

- **`conda activate` 报 "Run 'conda init'..."**：非交互 shell 先 `source .../etc/profile.d/conda.sh`。
- **`ray job submit` 找不到 `train_async.py`**：entrypoint 在 ray head 家目录执行，脚本用绝对路径
  （`import slime` 靠 conda 环境 editable 安装解析），勿改成 `cd X && python3 ...`。
- **`--load` 目录不存在**：首次运行先 `bash scripts/convert_hf_to_torch_dist.sh`。
- **想断点续训**：见 §4.3。
- **换模型**：改 `MODEL_DIR` / `TORCH_DIST_DIR` 环境变量，并确保架构与
  `slime_plugins.models.qwen3_5` spec 匹配（或换成对应 spec）。

## 8. 进一步阅读（官方文档）

- 参数总览与训练/推理资源分配：[get_started/usage.md](get_started/usage.md)
- 自定义生成 / reward / rollout 函数：[get_started/customization.md](get_started/customization.md)
- 各模型示例（含 Qwen3 系列）：[examples/qwen3-30B-A3B.md](examples/qwen3-30B-A3B.md)
- 环境从零构建：本仓库的 [env_construct/](env_construct/README.md)（构建脚本 + 完整版本钉子）
