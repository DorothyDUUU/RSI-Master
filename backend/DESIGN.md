# GPU Backend — 设计文档

> 版本 2.0.0 | 2026-07-16

## 1. 概述

### 1.1 背景与动机

在 RSI-Master 的 Agent 工作流中，Agent 运行在 **CPU 机器** 上，但需要频繁调用 vLLM 推理服务、LLaMA-Factory 训练框架、benchmark 评测等 **GPU 后端服务**。原有的 task-based 架构将每次调用建模为一个"一次性任务"，存在以下不足：

- **无状态**：每次任务都是独立的 subprocess，无法复用 shell 环境（环境变量、工作目录等）
- **调度队列不可用**：优先级队列的 `_process_queue()` 实现有 bug，排队任务永远不会真正执行
- **无法交互**：Agent 不能逐步向同一个 shell 发送多条命令
- **缺少 CLI**：没有提供给 Agent 直接调用的命令行工具

### 1.2 设计目标

将后端从 "task abstraction"（一次性任务）重构为 **"shell session abstraction"**（持久化 shell 会话），核心原则：

1. **Session 即 tmux**：每个 shell 会话对应一个 tmux session，天然持久化，Agent 可以逐步交互
2. **脚本提交**：支持前台（阻塞等结果）和后台（tmux + 日志文件）两种模式
3. **纯 CLI 工具**：不耦合 evomaster agent 框架，`gpu-backend` 作为独立 CLI 使用
4. **自动转后台**：运行超过 5 分钟的前台任务自动转为后台，避免工具调用超时
5. **资源保护**：GPU 显存 <95%、利用率 <99% 的硬阈值保护


## 4. API 参考

### 4.1 端点一览

| Method | Path | Request | Response | 说明 |
|--------|------|---------|----------|------|
| `GET` | `/health` | — | `{status, resources, active_sessions}` | 健康检查 |
| `GET` | `/resources` | — | `GPUInfo` | GPU/CPU/内存 |
| `GET` | `/status` | — | `{sessions, gpu_info}` | 聚合状态 |
| `POST` | `/sessions/exec` | `SessionExecRequest` | `SessionExecResponse` | 执行命令 |
| `GET` | `/sessions` | — | `list[ShellSession]` | 列出所有会话 |
| `POST` | `/sessions/{id}/kill` | — | `{success, message}` | 终止会话 |
| `POST` | `/scripts/submit` | `ScriptSubmitRequest` | `ScriptSubmitResponse` | 提交脚本 |
| `GET` | `/scripts/{id}` | — | `ScriptInfo` | 脚本状态 |
| `GET` | `/scripts/{id}/log` | — | `ScriptLog` | 脚本日志 |
| `POST` | `/scripts/{id}/kill` | — | `{success, message}` | 终止脚本 |

### 4.2 数据模型

#### SessionExecRequest

```json
{
  "command": "nvidia-smi && echo done",
  "timeout": 300,
  "session_id": null
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `command` | `str` | ✓ | 要执行的 shell 命令 |
| `timeout` | `int` | — | 超时秒数 (default: 300) |
| `session_id` | `str \| null` | — | 目标会话 ID；为 null 时创建新会话 |

#### SessionExecResponse

```json
{
  "session_id": "gpu_shell_a1b2c3d4",
  "return_code": 0,
  "stdout": "hello\n/workspace",
  "stderr": "",
  "cwd": "/workspace",
  "is_new": true
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `session_id` | `str` | 命令执行的会话 ID |
| `return_code` | `int` | 命令退出码 |
| `stdout` | `str` | 标准输出 |
| `stderr` | `str` | 标准错误（当前总是空，tmux 合并输出） |
| `cwd` | `str \| null` | 工作目录（仅新建会话时返回） |
| `is_new` | `bool` | 是否创建了新会话 |

#### ScriptSubmitRequest

```json
{
  "command_path": "/workspace/scripts/train.sh",
  "timeout": 3600,
  "resume_session_id": "gpu_shell_abc123",
  "run_in_background": true
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `command_path` | `str` | ✓ | bash 脚本的绝对路径 |
| `timeout` | `int` | — | 超时秒数 (default: 300) |
| `resume_session_id` | `str \| null` | — | 继承该会话的环境变量 |
| `run_in_background` | `bool` | — | 是否后台运行 |

#### ScriptSubmitResponse

```json
{
  "script_id": "gpu_script_a1b2c3d4",
  "session_id": "gpu_backend_a1b2c3d4",
  "log_path": "/tmp/gpu_backend_logs/gpu_script_a1b2c3d4.log",
  "status": "running",
  "return_code": null,
  "stdout": "",
  "stderr": "",
  "message": "Script running in background tmux session: gpu_backend_a1b2c3d4"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `script_id` | `str` | 唯一脚本 ID |
| `session_id` | `str \| null` | 后台 tmux 会话名 (仅 background) |
| `log_path` | `str \| null` | 日志文件路径 (仅 background) |
| `status` | `ScriptStatus` | `pending` \| `running` \| `completed` \| `failed` \| `killed` |
| `return_code` | `int \| null` | 退出码 (complete/failed 时有值) |
| `stdout` | `str` | 标准输出 (foreground 完成时) |
| `stderr` | `str` | 标准错误 |
| `message` | `str` | 人类可读的状态描述 |

#### ShellSession

```json
{
  "session_id": "gpu_shell_a1b2c3d4",
  "status": "active",
  "cwd": "/workspace",
  "created_at": "2026-07-16T10:30:00Z",
  "is_background": false,
  "log_path": null
}
```

#### GPUInfo

```json
{
  "devices": [
    {
      "device_id": 0,
      "name": "NVIDIA H20-3e",
      "total_memory_mb": 143771,
      "used_memory_mb": 132407,
      "free_memory_mb": 10761,
      "gpu_utilization": 29.0,
      "memory_utilization": 92.1,
      "temperature": 37.0,
      "processes": []
    }
  ],
  "total_devices": 1,
  "available_devices": [0],
  "cpu_percent": 12.5,
  "memory_percent": 45.2,
  "memory_total_gb": 503.5,
  "memory_used_gb": 227.6
}
```

---

## 5. CLI 参考

### 5.1 全局配置

```bash
export BACKEND_URL=http://gpu-server:8989   # 默认 http://localhost:8989
```

### 5.2 命令速查

```bash
# Shell 会话管理
gpu-backend exec -c "nvidia-smi"                        # 新会话执行命令
gpu-backend exec -c "echo hi" -s gpu_shell_xxx          # 已有会话执行
gpu-backend exec -c "ls -la" -t 60                      # 自定义超时
gpu-backend list-sessions                                # 列出所有会话
gpu-backend ls                                           # 同上 (别名)
gpu-backend kill-session -s gpu_shell_xxx                # 终止会话

# 脚本提交
gpu-backend submit -p ./train.sh                         # 前台执行 (阻塞等结果)
gpu-backend submit -p ./train.sh -b -t 7200              # 后台执行 + 2h 超时
gpu-backend submit -p ./eval.sh -r gpu_shell_xxx         # 继承 session 环境变量
gpu-backend script-status -s gpu_script_xxx              # 查看脚本状态
gpu-backend script-log -s gpu_script_xxx                 # 查看脚本日志
gpu-backend kill-script -s gpu_script_xxx                # 终止脚本

# 资源监控
gpu-backend info                                         # GPU/CPU/内存
gpu-backend status                                       # 会话 + 资源 (JSON)
gpu-backend health                                       # 健康检查

# 服务管理
gpu-backend server start                                 # 启动服务 (前台)
gpu-backend server start --port 9999 --reload            # 自定义端口 + 热重载
```

### 5.3 Python Client 等价调用

```python
from backend.gpu_backend import BackendClient

client = BackendClient("http://gpu-server:8989")

# 创建会话并执行命令
resp = client.exec_command("nvidia-smi", timeout=30)
print(f"session={resp.session_id}, stdout={resp.stdout}")

# 在已有会话中执行
resp2 = client.exec_command("echo $CUDA_VISIBLE_DEVICES",
                            session_id=resp.session_id)

# 后台提交训练脚本
resp = client.submit_script("/workspace/scripts/train.sh",
                            run_in_background=True,
                            timeout=7200)
print(f"log: {resp.log_path}")

# 查看 GPU 状态
gpu = client.get_resources()
for d in gpu.devices:
    print(f"GPU {d.device_id}: {d.memory_utilization:.1f}% mem")
```

---

## 6. 工作流示例

### 6.1 典型训练流程

```bash
#!/bin/bash
# 1. 启动 vLLM 推理服务（后台）
gpu-backend submit -p scripts/start_vllm.sh --background

# 2. 等待服务就绪
sleep 30

# 3. 创建持久化 shell 会话
SID=$(gpu-backend exec -c "echo ready" | jq -r '.session_id')

# 4. 在会话中设置环境变量
gpu-backend exec -c "export MODEL_PATH=/models/qwen-1.8b" -s $SID

# 5. 提交训练（前台，等结果）
gpu-backend submit -p scripts/train_sft.sh -r $SID -t 3600

# 6. 提交评测（继承训练的环境变量）
gpu-backend submit -p scripts/eval.sh -r $SID

# 7. 清理
gpu-backend kill-session -s $SID
```

### 6.2 Agent 工具调用流程

```
Agent                            GPU Backend Server
  │                                      │
  │── gpu-backend exec -c "nvidia-smi" ─→│  检查 GPU 资源
  │←── {gpu_info} ──────────────────────│
  │                                      │
  │── gpu-backend submit -p train.sh ───→│  提交训练脚本
  │     --background -t 7200             │  tmux new-session -d
  │←── {script_id, log_path} ───────────│  "bash train.sh > log"
  │                                      │
  │  (Agent 继续其他工作)                 │  (训练在 GPU 上运行)
  │                                      │
  │── gpu-backend script-log ───────────→│  查看进度
  │     -s gpu_script_xxx                │
  │←── {stdout: "epoch 3/10..."} ───────│
  │                                      │
  │  ... (poll 直到完成)                  │  ... (tmux session 退出)
  │                                      │
  │── gpu-backend script-status ────────→│  确认完成
  │     -s gpu_script_xxx                │
  │←── {status: "completed", rc: 0} ────│
```

---

## 7. GPU 资源监控

### 7.1 数据采集

```
┌─────────────────────────────────────┐
│           GPUMonitor                 │
├─────────────────────────────────────┤
│                                     │
│  主路径: nvidia-smi                  │
│  ┌─────────────────────────────┐    │
│  │ nvidia-smi --query-gpu=     │    │
│  │   index,name,                │    │
│  │   memory.total,memory.used,  │    │
│  │   memory.free,               │    │
│  │   utilization.gpu,           │    │
│  │   utilization.memory,        │    │
│  │   temperature.gpu            │    │
│  │ --format=csv,noheader,nounits│    │
│  └─────────────────────────────┘    │
│                                     │
│  Fallback: nvitop (pip install)      │
│  ┌─────────────────────────────┐    │
│  │ from nvitop import Device   │    │
│  │ for dev in Device.all():    │    │
│  │   dev.memory_stats()        │    │
│  │   dev.utilization()         │    │
│  │   dev.temperature()         │    │
│  └─────────────────────────────┘    │
│                                     │
│  系统指标: psutil                     │
│  ┌─────────────────────────────┐    │
│  │ psutil.cpu_percent()        │    │
│  │ psutil.virtual_memory()     │    │
│  └─────────────────────────────┘    │
│                                     │
│  硬阈值:                             │
│  MAX_MEMORY_UTILIZATION = 95.0%     │
│  MAX_GPU_UTILIZATION    = 99.0%     │
└─────────────────────────────────────┘
```

### 7.2 可用性判定

GPU 设备被视为"可用"当且仅当：
- `memory_utilization < 95.0%`
- `gpu_utilization < 99.0%`

可用设备按 `free_memory_mb` 降序排列。

---

## 8. 文件结构

```
backend/
├── DESIGN.md                    ← 本文档
├── README.md                    ← 原始重构需求
├── start_server.sh              ← 服务启动脚本
│
├── server/                      ← GPU 侧 FastAPI 服务
│   ├── __init__.py
│   ├── main.py                  ← FastAPI app + 9 个端点
│   ├── session_manager.py       ← tmux 会话管理核心
│   ├── gpu_monitor.py           ← GPU/CPU/内存监控
│   └── models.py                ← Pydantic 数据模型
│
├── gpu_backend/                 ← CLI + HTTP Client + Config
│   ├── __init__.py
│   ├── client.py                ← BackendClient (httpx)
│   ├── cli.py                   ← gpu-backend CLI (argparse)
│   ├── config.py                ← config.yaml loader
│   └── config.yaml              ← 默认配置文件
│
├── tools/                       ← (未改动 — 留给 evomaster 集成)
│   ├── __init__.py
│   └── gpu_tools.py
│
├── tests/                       ← 测试
│   ├── test_server.py           ← 服务端集成测试 (12 tests)
│   ├── test_client.py           ← 客户端集成测试 (11 tests)
│   └── test_cli.py              ← CLI 参数解析测试 (15 tests)
│
└── backend/                     ← Agent 参考脚本 (未改动)
    ├── vllm/
    ├── llama_factory/
    ├── ptb_aime/
    └── ptb_gsm8k/
```

---

## 9. 配置

### 9.1 服务端环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `BACKEND_HOST` | `0.0.0.0` | FastAPI 绑定地址 |
| `BACKEND_PORT` | `8989` | FastAPI 绑定端口 |
| `BACKEND_LOG_LEVEL` | `info` | uvicorn 日志级别 |
| `BACKEND_LOG_DIR` | `/tmp/gpu_backend_logs` | 任务日志目录 |
| `BACKEND_RELOAD` | — | 设为任意值启用 uvicorn --reload |

### 9.2 客户端环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `BACKEND_URL` | `http://localhost:8989` | 后端服务 URL |

### 9.3 可调常量 (session_manager.py)

| 常量 | 默认值 | 说明 |
|------|--------|------|
| `AUTO_BACKGROUND_SECONDS` | `300` | 前台任务自动转后台阈值 |
| `LOG_DIR` | `/tmp/gpu_backend_logs` | 日志输出目录 |
| `SHELL_SESSION_PREFIX` | `gpu_shell_` | 交互式 tmux 会话名前缀 |
| `BACKEND_SESSION_PREFIX` | `gpu_backend_` | 脚本执行 tmux 会话名前缀 |

### 9.4 资源阈值 (gpu_monitor.py)

| 常量 | 默认值 | 说明 |
|------|--------|------|
| `MAX_MEMORY_UTILIZATION` | `95.0` | GPU 显存使用率上限 (%) |
| `MAX_GPU_UTILIZATION` | `99.0` | GPU 核心使用率上限 (%) |

---

## 10. 设计决策记录

### 10.1 为什么用 tmux 而不是 subprocess？

- **持久化**：tmux session 在命令完成后继续存在，Agent 可以逐步交互
- **环境继承**：后续命令自动继承之前设置的环境变量和工作目录
- **后台运行**：`tmux new-session -d` 天然支持后台执行
- **日志捕获**：`tmux capture-pane` 可随时获取会话输出
- **进程组管理**：`tmux kill-session` 自动清理所有子进程

### 10.2 为什么用 tmp 文件而不是 capture-pane 收集命令输出？

`tmux capture-pane` + sentinel marker 方案存在 parser 歧义：发送到 tmux 的命令文本本身包含 marker 字符串（在 `echo __MARKER__` 中），导致难以区分「命令文本行」和「输出结果行」。

tmp 文件方案：`(command) > /tmp/xxx.out 2>&1; echo $? > /tmp/xxx.rc`，逻辑简单可靠，不依赖任何字符串解析。

### 10.3 为什么用 argparse 而不是 click/typer？

项目不依赖 click 或 typer，使用标准库 argparse 零额外依赖，且 `run.py` 等已有脚本全部使用 argparse，保持一致性。

### 10.4 旧 scheduler.py 为什么被完全替换？

`server/scheduler.py` 中的队列处理 (`_process_queue`) 有 bug：排队任务只会被标记为 RUNNING 但从不实际执行脚本。与其修复一个有状态丢失问题的旧架构，不如基于 session 模型重写。

### 10.5 为什么不耦合 evomaster agent 框架？

`tools/gpu_tools.py` 中的旧 agent 工具硬编码了 `<repo>` 路径且依赖 evomaster 内部模块。新设计将 `gpu-backend` 作为独立 CLI 工具，agent 可以直接通过 bash 调用，降低耦合度。未来如需 agent tool 集成，只需在工具中封装 CLI 调用即可。
