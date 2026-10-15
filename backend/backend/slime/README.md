# backend/slime — 环境激活与训练启动速查

本目录是 PostTrainBenchRollout 的 **slime 训练入口（demo）**。slime 是 THUDM 的后训练框架
（Megatron-LM 训练 + SGLang rollout），真实工作区与已跑通的配置在共享盘：

| 组件 | 位置 |
|---|---|
| slime 框架源码 | `$SLIME_ROOT`（editable 安装） |
| slime 文档 | 官方文档 `./docs/`；本部署具体操作见 [docs/deployment.md](docs/deployment.md) |
| Megatron-LM | `$SLIME_MEGATRON_LM`（`PYTHONPATH` 指向） |
| SGLang | `$SLIME_WORKSPACE/sglang`（rollout 引擎，SFT 离线模式不启动） |
| conda 环境 | `$SLIME_ENV`（Python 3.12 / PyTorch 2.11+cu129） |
| 已跑通权重 | Qwen3.6-35B-A3B：HF `<repo>/public/models/Qwen/Qwen3.6-35B-A3B`，torch_dist `<repo>/public/models/Qwen/Qwen3.6-35B-A3B_torch_dist` |
| 已跑通数据 | `$SLIME_WORKSPACE/data/xyz_aquila_sft.parquet`（3412 条，seq≤81920） |

> [!IMPORTANT]
> **使用约定（务必遵守）**
>
> - ❌ **不可以：改装环境 / 修改源代码。**
>   不得改动已安装的训练环境（conda env `slime`、各库版本、构建产物），
>   不得修改 slime / Megatron-LM / SGLang 的**源代码**。框架与环境的任何改动
>   都会影响其他使用者与可复现性。
> - ✅ **可以：创建不同的配置脚本。**
>   复制 `scripts/launch_sft.sh` 等模板，换数据 / 换超参 / 换并行度 / 换 ckpt 输出目录，
>   全部通过脚本顶部 CONFIG 区或环境变量覆盖完成——**无需改一行源代码**。
> - 一句话：**只在配置层做改动，不碰安装与源码。**

## 0. 环境从零构建

本仓库**不含** slime / Megatron-LM / SGLang 的源码和 conda 环境（合计约 25 GB）。
需要自建时跑 [`docs/env_construct/`](docs/env_construct/README.md) 里的脚本：

```bash
export WORKSPACE_DIR="$HOME/slime-workspace"   # 放三个 checkout，需要大磁盘
export CONDA_DIR="$HOME/miniconda3"
git clone https://github.com/THUDM/slime.git "$WORKSPACE_DIR/slime"

bash docs/env_construct/build_slime_env.sh        # step 1
bash docs/env_construct/build_slime_env_step2.sh  # step 2
bash docs/env_construct/verify_gpu_env.sh         # GPU 机上验证
```

版本钉子（torch 2.11.0+cu129 / SGLang `0b3bb0cbe` / Megatron-LM `1dcf0dafa` /
TE 2.16.1 / apex `10417ace` / flash-attn 2.8.3）见该 README。建好后把
`$SLIME_ENV` / `$SLIME_MEGATRON_LM` / `$SLIME_CONVERT_TO_HF` 导出给
`backend/backend/submit_pipeline`——未设置时渲染出的 `pipeline.sh` 会在执行前
断言失败并指向该文档，不会静默换解释器。

## 1. 激活环境

```bash
# 非交互 shell 下最可靠的方式（脚本里也是这么做的）：
source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate slime

# 等价写法（不依赖 conda 函数，直接改 PATH）：
export PATH="$SLIME_ENV/bin:$PATH"
```

激活后建议设置（与构建时一致，供 JIT/原生扩展匹配 H100/H200 架构）：

```bash
export CUDA_HOME="$CONDA_PREFIX"
export TORCH_CUDA_ARCH_LIST=9.0
export NVTE_CUDA_ARCHS=9.0
```

验证（单行冒烟）：

```bash
conda activate slime
python -c "import torch, megatron, slime; print('ok', torch.__version__)"
```

> 一键版见 `scripts/activate_env.sh`；更完整的 GPU 验证见 `scripts/verify_env.sh`；
> 环境预检（导入链 + GPU，带退出码）见 `scripts/precheck_env.sh`。
> **一条命令验证 slime 能否跑 SFT → `bash scripts/verify_sft.sh`**（见 §2 方式 B）。
> **完整的分步操作文档见 [docs/deployment.md](docs/deployment.md)**（环境激活、启动训练、数据/权重准备、断点续训、FAQ）。

## 2. 启动训练（三种方式）

### 方式 A：GPU 节点上直接跑（推荐，最快上手）

在 8×H20 GPU 节点上：

```bash
# 一键启动：conda 初始化（幂等）+ 激活 slime + 启动训练 + tee 日志
bash <repo>/backend/backend/slime/scripts/launch_sft.sh

# 换数据 / 换超参用环境变量覆盖（默认值 = 之前跑通的值）：
DATA_FILE=/path/to/xxx.parquet NUM_EPOCH=2 \
  bash .../scripts/launch_sft.sh
```

训练主体就是 `slime/train_async.py`（被 `ray job submit` 提交到单节点 ray cluster）。
日志：脚本内 `OUTPUT_DIR/train.log`（`tail -f` 即可看）。

### 方式 B：一键验证 slime 跑 SFT 可用（推荐先用这个）

```bash
bash <repo>/backend/backend/slime/scripts/verify_sft.sh
```

三段式：**环境预检**（导入链 + GPU，失败即停，不白跑）→ **前置检查**（数据 / torch_dist
缺失会给出修复命令）→ **端到端冒烟**（8 条合成样本 / 1 epoch → 1 个真实训练步 + 存 ckpt）。
退出码：`0`=通过；`1`=环境/前置失败；`2`=冒烟训练失败。
等价于 `precheck_env.sh` + `launch_sft_smoke.sh`。

冒烟默认在 **Qwen3-4B-Base（稠密 4B，单卡）** 上跑，架构参数来自官方
`slime/scripts/models/qwen3-4B.sh`（无 `--spec` / 无 MoE）。首次需先转 torch_dist 权重：

```bash
MODEL_DIR=<repo>/public/models/Qwen/Qwen3-4B-Base \
SAVE_DIR=${MODEL_DIR}_torch_dist \
MODEL_SPEC=$SLIME_ROOT/scripts/models/qwen3-4B.sh \
  bash .../scripts/convert_hf_to_torch_dist.sh
```

> `convert_hf_to_torch_dist.sh` 的默认值已直接改成 Qwen3-4B-Base（`MODEL_DIR`、`SAVE_DIR`、`MODEL_SPEC`），
> 在 GPU 节点上直接 `bash scripts/convert_hf_to_torch_dist.sh` 即可，产物在 `${MODEL_DIR}_torch_dist`。

只跑冒烟（跳过预检），**启动脚本是 `launch_sft_smoke.sh`**：

```bash
NUM_GPUS=1 TP_SIZE=1 EP_SIZE=1 HAS_NVLINK=0 \
  bash <repo>/backend/backend/slime/scripts/launch_sft_smoke.sh
```

上面 4 个并行参数默认值就是这个（稠密 Qwen3-4B 单卡），不传也行。训练结束会打印：
`SMOKE PASSED -> $SLIME_WORKSPACE/training_scripts/ckpt/Qwen3-4B-Base_sft_smoke/SMOKE_PASSED`。

### 并行度 / NVLink 手动设置（GPU 数与集群拓扑相关）

**不做自动检测，全部用环境变量手动覆盖**（默认值不同：冒烟 = 稠密 Qwen3-4B 单卡；正式 = 8×H20 MoE 跑通值）：

| 环境变量 | 冒烟默认 | 正式默认 | 含义 |
|---|---|---|---|
| `NUM_GPUS` | `1` | `8` | ray 分配卡数 / actor 每节点卡数 |
| `TP_SIZE` | `1` | `2` | `--tensor-model-parallel-size`（GQA `num-query-groups=2`，只能 1 或 2） |
| `EP_SIZE` | `1` | `4` | `--expert-model-parallel-size`（稠密模型固定 1；MoE 需整除专家数 256） |
| `HAS_NVLINK` | `0` | `0` | 写入 runtime env 的 `NCCL_NVLS_ENABLE`：NVLink 节点设 `1` 启用集合通信 |

约束：`TP_SIZE × EP_SIZE` 必须等于 `NUM_GPUS`（PP=CP=1）。配错会在清进程**之前**
直接报错退出，不会误杀节点上其他任务；脚本会在顶部打印实际并行配置便于核对。

```bash
# 单张 H20（无 NVLink）
NUM_GPUS=1 TP_SIZE=1 EP_SIZE=1 HAS_NVLINK=0 \
  bash .../scripts/launch_sft_smoke.sh

# 8×H20（默认值即此配置；若该节点有 NVLink，把 HAS_NVLINK 改成 1）
NUM_GPUS=8 TP_SIZE=2 EP_SIZE=4 HAS_NVLINK=0 \
  bash .../scripts/launch_sft.sh
```

## 3. 前置条件（首次运行）

1. **torch_dist 权重**：`--load` 必须指转换产物（单卡转换一次即可，加载时按 TP/EP/PP 自动
   重分片），**不能**直接指 HF 目录。
   - 冒烟（Qwen3-4B-Base）：`${MODEL_DIR}_torch_dist`（即 `public/models/Qwen3-4B-Base_torch_dist`），转换命令见 §2 方式 B。
   - 正式（Qwen3.6-35B-A3B）：`<repo>/public/models/Qwen/Qwen3.6-35B-A3B_torch_dist`，
     `bash scripts/convert_hf_to_torch_dist.sh` 默认即该模型。
2. **数据**：parquet，必须含 `messages` 列（OpenAI chat 格式）。
3. **GPU 节点**：驱动 ≥ 560（支持 CUDA 12.9）。

## 4. 关键参数说明

| 项 | 值 | 说明 |
|---|---|---|
| 并行 | TP=2, EP=4, CP=1, SP | **正式（launch_sft.sh）**：8×H20；EP=4 需满足 `etp×ep×pp` 整除 world_size=8；TP=4 会破坏 GQA（num-query-groups=2）。**冒烟（launch_sft_smoke.sh）** 是稠密 Qwen3-4B，默认 1 卡 TP=1 EP=1。均用 `TP_SIZE`/`EP_SIZE`/`NUM_GPUS` 覆盖，见 §2「并行度 / NVLink 手动设置」 |
| seq-length | 81920 | 8×H20 140GB/card 的安全上限；98304 实测 OOM |
| `--debug-train-only` | **保留** | SFT 离线数据路径：只跳过 SGLang 引擎启动/权重同步，**不跳过**训练步和权重保存 |
| 优化器 | Adam + CPU offload + distributed optimizer | 三处内存修复：`--main-grads-dtype bf16`、sft 时 fp32_output=False、`--log-probs-chunk-size 4096` |
| loss mask | `--loss-mask-type qwen3_5` | `<think>` 内容不参与训练，只对 `</think>` 之后计算 loss |

### 训练启动脚本与 run-local 拷贝 (ESTALE 硬化, 2026-08-24)

`launch_sft.sh` 内部 `cd "$(dirname "$0")"` (第 26 行) 自定位到自身目录, 再相对
`source ./activate_env.sh` (第 106 行) —— **脚本依赖与兄弟脚本同目录**, 拷贝/移动
必须是整目录, 不能只拷单个文件。

- **ExpOS submit 管线 (stage-1)**: 已把整个 `scripts/` 拷到 run 本地
  (`$WORKSPACE/scripts`, 每 run 专属、永不重写) 再 `bash` 执行 —— 执行进程持有
  run-local 文件, 对共享 NFS 上的并发编辑免疫。这是 2026-08-23 平台 4 pod 训练成功
  被 ESTALE 误标 failed 的根因修复 (机制见 `backend/backend/submit_docs/PIPELINE_INTERNALS.md`
  第 4 层)。
- **手动直接跑 (方式 A/B)**: 没有这层保护 —— **训练进行中绝不改写 `launch_sft.sh` /
  其它 scripts 脚本**。bash 惰性分块读脚本, 长训练只在收尾读尾部; NFS 上文件被原子
  替换后, 运行中进程持有的旧 inode 失效 → 收尾 `Stale file handle` → 训练成功被误标
  failed。

## 5. 脚本速查

| 脚本 | 作用 |
|---|---|
| `scripts/activate_env.sh` | 激活 slime conda 环境（source 即用） |
| `scripts/verify_env.sh` | 验证环境 + GPU + 导入链 |
| `scripts/precheck_env.sh` | 环境预检：导入链 + GPU（退出码；`SKIP_GPU_CHECK=1` 可跳过 GPU 检测） |
| `scripts/verify_sft.sh` | **一条命令验证 slime 跑 SFT**：预检 + 前置检查 + 冒烟（推荐） |
| `scripts/cleanup.sh` | 清理残留 ray / sglang / python 进程（重跑前必做） |
| `scripts/convert_hf_to_torch_dist.sh` | HF 权重 → Megatron torch_dist（单卡，约 10-30 分钟） |
| `scripts/launch_sft.sh` | 一键启动 SFT 训练（demo，参数化，默认=跑通值） |
| `scripts/launch_sft_smoke.sh` | 端到端冒烟（稠密 Qwen3-4B，1 个真实训练步 + 存 ckpt） |
| `scripts/watch_training.sh` | 观测训练日志 / ray dashboard |

## 6. 常见问题

- **`conda activate` 报 "Run 'conda init'..."**：非交互 shell 需先 `source .../etc/profile.d/conda.sh`。
- **`ray job submit` 找不到 `train_async.py`**：entrypoint 在 ray head 家目录执行，脚本已用绝对路径
  （`import slime` 靠 conda 环境 editable 安装解析），勿改成 `cd X && python3 ...`。
- **`--load` 目录不存在**：首次运行先做权重转换。
- **想断点续训**：去掉 `--no-save-optim --no-save-rng`，`--load` 指向上一个保存的 ckpt 目录。
- **换模型**：改 `MODEL_DIR` / `TORCH_DIST_DIR` 环境变量，并确保模型架构与
  `slime_plugins.models.qwen3_5` spec 匹配（或换成对应 spec）。
