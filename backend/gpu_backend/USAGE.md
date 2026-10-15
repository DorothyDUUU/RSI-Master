# GPU Backend — 使用手册

## 背景

Agent 运行在 CPU 机器上，训练（LLaMA-Factory）、推理（vLLM）、评测（inspect-eval）跑在
GPU 机器上。GPU Backend 在 GPU 机器上启动 FastAPI 服务，Agent 通过 CLI 远程提交命令和
脚本，由服务端在远程 GPU 的  tmux 会话中执行。

```
CPU 机器 (Agent)                                         GPU 机器 (Backend Server)
┌──────────────────────────────────────────┐   HTTP     ┌──────────────────────────┐
│ python gpu_backend/main.py               │ ────────→  │ FastAPI (:8090)          │
│  exec -c "..."                           │            │  ├─ SessionManager       │
│  submit -p /abs/path/to/xxx.sh           │            │  │   tmux gpu_shell_*    │
│  info                                    │            │  │   tmux gpu_backend_*  │
│  list-sessions                           │            │  └─ GPUMonitor           │
└──────────────────────────────────────────┘            └──────────────────────────┘
```

| 概念 | 说明 |
|------|------|
| **Shell Session** | 持久化 tmux 会话 (`gpu_shell_<hash>`)，命令在其中逐步执行，环境变量在命令间保持 |
| **Script** | bash 脚本，可前台执行（阻塞等结果）或后台执行（tmux + 日志文件） |
| **自动转后台** | 前台脚本运行超过 5 分钟自动转为后台，避免 HTTP 超时 |


## 命令详解

我们为 Agent 提供了一套完善的命令行工具箱，使其可以在本地调用远程的 GPU 计算资源，进行模型训练，评测

### `exec` — 执行 Shell 命令

```bash
# 创建新会话并执行命令
python gpu_backend/main.py exec -c "nvidia-smi"
```

```bash
# 在已有会话中继续执行 — 继承之前的环境变量和工作目录
python gpu_backend/main.py exec -c "echo $CUDA_VISIBLE_DEVICES" -s gpu_shell_a1b2c3d4
```

```bash
# 自定义超时
python gpu_backend/main.py exec -c "sleep 10" -t 5
```

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `-c` / `--command` | ✓ | — | 要执行的 shell 命令 |
| `-s` / `--session-id` | — | 自动创建 | 目标会话 ID |
| `-t` / `--timeout` | — | config.yaml `defaults.exec_timeout` | 超时秒数, 设置为 0 时即为无超时， |


### `submit` — 提交脚本

> **⚠️ 路径必须绝对化(端到端实测踩过的坑)**
>
> `-p` 的脚本路径、脚本内部引用的所有路径,以及 ExpOS `worker submit` 的
> `--data-path` / `--training-config-path`,**都必须传 GPU 机器上存在的绝对路径**。
>
> 原因:脚本是在 GPU 机器上的 tmux 会话里执行的,`-p` 里的相对路径会相对
> **server 进程的 CWD** 解析,而不是你本地终端的 CWD;ExpOS pipeline 里相对路径
> 则会相对 `LLAMA_FACTORY_CWD`(= llama_factory 源码树)解析。结果是 stage-1 直接
> `FileNotFoundError: .../train.yaml` 这类"明明文件在却找不到"的诡异报错——端到端
> 测试中 exp_0001 正是踩在这个坑上,改用绝对路径后 lora + full-sft 全链路通过。
>
> 共享 NFS 挂载下 `$SHARED_ROOT/...` 绝对路径在两台机器上都有效,请一律用这种写法。
> (`WorkerService.submit` 已对 data/config 参数自动 resolve 相对路径,但 agent /
> 脚本仍应主动传绝对路径,避免歧义。)

```bash
# 前台执行 — 阻塞等结果
python gpu_backend/main.py submit -p <repo>/backend/gpu_backend/train.sh
```

```bash
# 后台执行 — 立即返回，日志文件追踪
python gpu_backend/main.py submit -p <repo>/backend/gpu_backend/train.sh --background -t 7200

# -t 0 代表不设置超时时限
```

```bash
# 继承已有 session 的环境变量
python gpu_backend/main.py submit -p <repo>/backend/gpu_backend/scripts/eval.sh -r gpu_shell_xxx -t 600
```

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `-p` / `--script-path` | ✓ | — | bash 脚本路径 |
| `-b` / `--background` | — | false | 后台运行 |
| `-r` / `--resume-session-id` | — | — | 继承该 session 的环境变量 |
| `-t` / `--timeout` | — | config.yaml `defaults.script_timeout` | 超时秒数 |

#### GPU quota 和 排队逻辑

所有 `submit` 的脚本**统一入队**，由后台 VRAM 扫描器调度启动。

**调度规则：**

```
每 5s 扫描 nvidia-smi → 看有多少 GPU 显存满足条件
    ↓
队列最优先任务的需求 ≤ 空闲 GPU 数？
    Yes → pop, 启动 (自动设 CUDA_VISIBLE_DEVICES), 冷却 60s
    No  → 跳过, 等下次扫描
```

- **HIGH** → 直接插队到队列最前面 (**请仅在最高优先级的时候启用!**)
- **NORMAL / LOW** → 追加到队列末尾，按优先级排序
- **不会抢占**：已在运行的任务不会被 kill，高优只是排前面
- **纯显存判断**：不看 GPU 利用率，只看 nvidia-smi 的实际 VRAM 占用

**相关参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-g` / `--gpu-nums` | **必填** | 需要的 GPU 数量 |
| `--priority` | `normal` | 调度优先级：`low` / `normal` / `high` |
| `--max-mem-quota` | config.yaml `defaults.max_mem_quota` | 每张 GPU 至少需要的空闲显存 (MB)，不满足的 GPU 不会被分配 |
| `--need-exclusive-gpu` | `false` | 独占分配到的 GPU 到任务结束；默认关闭时只持有 **20 分钟独占租约**，之后该卡对其他非独占任务开放共享 |
| `-t` / `--timeout` | config.yaml | 超时秒数，`0` = 不限 |

**独占 vs 共享（`--need-exclusive-gpu`，默认关闭）：**

```
默认 (共享):  任务落到卡上 → 独占 20 分钟 → 租约到期 → 该卡对其他非独占任务开放共享
             (卡上还在跑的任务不被打断; 新共享任务自己再拿一个全新 20 分钟租约)
--need-exclusive-gpu: 独占整张卡到任务 tmux 会话结束, 一张卡并发最多一个任务 (旧行为)
```

- **独占任务只会落到"完全干净"的卡上**——没有任何 gpu-backend 任务在跑的卡。
- 非独占任务之间共享同一张卡时, 以**最晚到期的租约**为准: 卡上只要还有任一任务租约未到期, 新任务就不能上卡。
- VRAM 检查仍生效: 共享卡的 `vram_usage_percent < 95%` 且空闲显存 ≥ `--max-mem-quota` 才会被分配, 防止显存爆掉。
- ⚠️ **长挂任务注意**: 默认共享下, `-t 0` 长时间挂着的 vLLM serve 在 20 分钟后其卡会对其他非独占任务开放, 可能争显存。若要整卡跑满整个生命周期, 请显式加 `--need-exclusive-gpu`。

**示例：**

```bash
# 高优抢占队列头，要 2 张 GPU，每张至少 40GB 空闲
python gpu_backend/main.py submit -p <repo>/backend/gpu_backend/train.sh --background -g 2 --priority high --max-mem-quota 40960

# 普通优先级，1 张 GPU，不限显存
python gpu_backend/main.py submit -p <repo>/backend/gpu_backend/eval.sh --background -g 1 --priority normal -t 7200
```

> **重要**：调度器会自动设置 `CUDA_VISIBLE_DEVICES`，**不要在脚本里手动设置这个环境变量**，否则会导致 GPU 分配冲突。脚本只需要正常使用 GPU（如 `device=cuda:0`），调度器已通过 `CUDA_VISIBLE_DEVICES` 隔离了可见设备。

### `script-status` 查看调度提交脚本的状态

一般来说，利用 submit 提交脚本后，会进入排队状态，在一段时间后，服务会被激活执行，可以使用 `script-status` 查看调度提交脚本的状态:

```bash
# * 提交任务，初始显示排队状态
python gpu_backend/main.py submit -p <repo>/backend/backend/eval_gsm8k/eval.sh --background -t 3600 -g 0

2026-07-17 16:28:50,655 - httpx - INFO - HTTP Request: POST http://localhost:8989/scripts/submit "HTTP/1.1 200 OK"
[SCRIPT_ID] gpu_script_04ba2f09
[STATUS] queued
[MESSAGE] Queued — need 0 GPU(s), currently 0 free. Priority: normal. Queue size: 1.

# * 第二次查询，显示正在运行，得到重定向脚本路径
python gpu_backend/main.py script-status -s gpu_script_04ba2f09
2026-07-17 16:29:15,260 - httpx - INFO - HTTP Request: GET http://localhost:8989/scripts/gpu_script_04ba2f09 "HTTP/1.1 200 OK"
{
  "script_id": "gpu_script_04ba2f09",
  "command_path": "<repo>/backend/backend/eval_gsm8k/eval.sh",
  "status": "running",
  "session_id": "gpu_backend_04ba2f09",
  "log_path": "/tmp/gpu_backend_logs/gpu_script_04ba2f09.log",
  "return_code": null,
  "stdout": "",
  "stderr": "",
  "created_at": "2026-07-17T08:28:49.628791Z",
  "started_at": "2026-07-17T08:28:56.683413Z",
  "completed_at": null,
  "gpu_devices": [],
  "priority": "normal",
  "gpu_nums": 0,
  "max_mem_quota": 3000
}
```


### `list-sessions` — 查看会话

```bash
python gpu_backend/main.py list-sessions
# 或简写
python gpu_backend/main.py ls
```

### `kill-session` — 终止会话

```bash
python gpu_backend/main.py kill-session -s gpu_shell_a1b2c3d4
```


### 资源监控

可以通过资源监控查看可用的 GPU 资源: 

```bash
python gpu_backend/main.py info
```

输出：
```
[GPU_DEVICES]
GPU 0  NVIDIA H20-3e         vram  45.2%  util  98.0%  free  78000MB ← gpu_script_32c397e2 (normal)
GPU 1  NVIDIA H20-3e         vram   1.4%  util   0.0%  free 141000MB

[SYSTEM]  CPU: 12.5%  Mem: 45.2% (227.6/503.5 GB)

[QUEUE] (1 waiting)
  gpu_script_abc123  priority=normal  need=2 GPU(s)
```


## 实战使用场景

### 运行后台脚本

```bash
# 1. 提交后台脚本 → 立即返回 script_id (脚本路径必须绝对)
$ python gpu_backend/main.py submit -p <repo>/backend/gpu_backend/train.sh --background
[SCRIPT_ID] gpu_script_a1b2c3d4
[STATUS] running
[LOG_PATH] /tmp/gpu_backend_logs/gpu_script_a1b2c3d4.log

# 2. 过一会查一下跑完了没有
$ python gpu_backend/main.py script-status -s gpu_script_a1b2c3d4
{
"script_id": "gpu_script_a1b2c3d4",
"status": "completed",
"return_code": 0,
...
}

# 完成后或者存在异常，可以直接杀死对应脚本
python gpu_backend/main.py kill-script -s gpu_script_x1y2z3
```

在这个过程中，也可以使用 bash 工具查看重定向的 log 文件的进度，查看具体训练任务的进度。

### `vllm` 部署模型

参考 `backend/vllm/scripts/run_qwen_3-1.8.sh` 的示例代码，正确激活对应的 vllm 环境，在后台部署模型:

```bash
python gpu_backend/main.py submit -p <repo>/backend/backend/vllm/scripts/run_qwen_3-1.8.sh --background -t 0 -g 1

# * -t 0 代表不设置超时实现，在后台挂起

# * output
2026-07-17 11:16:07,083 - httpx - INFO - HTTP Request: POST http://localhost:8989/scripts/submit "HTTP/1.1 200 OK"
[SCRIPT_ID] gpu_script_32c397e2
[STATUS] running
[SESSION_ID] gpu_backend_32c397e2
[LOG_PATH] /tmp/gpu_backend_logs/gpu_script_32c397e2.log
[MESSAGE] Script running in background tmux session: gpu_backend_32c397e2
```

- 根据工具返回的 `[LOG_PATH]` 查看具体的日志，检查服务是否完成或者被异常中断
- 你也可以根据 `[SCRIPT_ID]` 进行查询:

```bash
python gpu_backend/main.py script-status -s gpu_script_32c397e2

# {
#   "script_id": "gpu_script_32c397e2",
#   "command_path": "<repo>/backend/backend/vllm/scripts/run_qwen_3-1.8.sh",
#   "status": "running",
#   "session_id": "gpu_backend_32c397e2",
#   "log_path": "/tmp/gpu_backend_logs/gpu_script_32c397e2.log",
#   "return_code": null,
#   "stdout": "",
#   "stderr": "",
#   "created_at": "2026-07-17T03:16:06.049960Z",
#   "started_at": "2026-07-17T03:16:06.049960Z",
#   "completed_at": null,
#   "gpu_devices": [
#     0
#   ],
#   "priority": "normal",
#   "gpu_nums": 1,
#   "max_mem_quota": null
# }
```

之后，就可以进行模型评测了，模型部署在 GPU 的远程端口上，可以直接在远程启动评测服务，得到模型的分数。评测完成后，你需要终止你部署的 Vllm 服务，你可以使用命令:

```bash
python gpu_backend/main.py kill-script -s gpu_script_32c397e2
```

之后再去检查，就会发现模型服务已经被成功杀死:

```bash
kend# python gpu_backend/main.py script-status -s gpu_script_32c397e2
2026-07-17 11:28:24,737 - httpx - INFO - HTTP Request: GET http://localhost:8989/scripts/gpu_script_32c397e2 "HTTP/1.1 200 OK"
{
  "script_id": "gpu_script_32c397e2",
  "command_path": "<repo>/backend/backend/vllm/scripts/run_qwen_3-1.8.sh",
  "status": "killed",
  "session_id": "gpu_backend_32c397e2",
  "log_path": "/tmp/gpu_backend_logs/gpu_script_32c397e2.log",
  "return_code": -1,
  "stdout": "",
  "stderr": "",
  "created_at": "2026-07-17T03:16:06.049960Z",
  "started_at": "2026-07-17T03:16:06.049960Z",
  "completed_at": "2026-07-17T03:27:45.971497Z",
  "gpu_devices": [
    0
  ],
  "priority": "normal",
  "gpu_nums": 1,
  "max_mem_quota": null
}
```

### `llama-factory` 模型训练

```bash
# * 一个简单的 llama-factory 训练脚本
python gpu_backend/main.py submit -p <repo>/backend/backend/llama_factory/scripts/train.sh --background -t 3600 -g 1
```

可以阅读对应 Llama-factory 的源代码和详细文档，了解更高级的训练用法和数据配置!

### 模型评测

以 GSM8K 模型评测为例:

```bash
python gpu_backend/main.py submit -p <repo>/backend/backend/eval_gsm8k/eval.sh --background -t 3600 -g 0
```

评测脚本有许多开关，可以控制并发数，测试模型等等，例如，在快速迭代的时候，可以只取小批量测试集进行测试.
* 注意，评测脚本往往很长，请使用 tail 或者 head 进行查看