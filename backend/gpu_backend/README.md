---
BACKEND_DESCRIPTIONS
Remote GPU execution backend — a FastAPI server on the GPU machine runs submitted
bash scripts/commands in persistent tmux sessions, called by CPU-side agents via
`python gpu_backend/main.py exec -c <cmd>` / `submit -p <abs .sh> [--background] [--gpu-nums N]`.
The scheduler picks the free card and injects CUDA_VISIBLE_DEVICES — never set it
yourself; use absolute NFS paths only. See USAGE.md for the full CLI.
---

# GPU Backend — Remote GPU Execution

Agent 跑在 CPU 机器上, 训练 (LLaMA-Factory) / 推理 (vLLM) / 评测 跑在 GPU 机器上。
GPU Backend 在 **GPU 机器**上起 FastAPI 服务, agent 通过 CLI 远程提交命令和脚本,
服务端在远程 GPU 的 **tmux 会话**里执行。

```
CPU 机器 (Agent)                                        GPU 机器 (Backend Server)
┌────────────────────────────────────────┐   HTTP      ┌──────────────────────────┐
│ python gpu_backend/main.py             │ ─────────→  │ FastAPI (:8090)          │
│  exec -c "..."                         │             │  ├─ SessionManager       │
│  submit -p /abs/path/to/xxx.sh         │             │  │   tmux gpu_shell_*    │
│  info / list-sessions                  │             │  └─ GPUMonitor / scheduler│
└────────────────────────────────────────┘             └──────────────────────────┘
```

完整手册见 `USAGE.md` (命令详解 / 调度逻辑 / 踩坑记录)。

---

## 核心命令

| 命令 | 用途 |
|---|---|
| `python gpu_backend/main.py exec -c "<cmd>" [-s <session_id>] [-t <timeout>]` | 在 (新建或已有) shell 会话里执行命令, 环境变量跨命令保持 |
| `python gpu_backend/main.py submit -p /abs/path/to/xxx.sh [--background] [-g <gpu_nums>] [--priority low|normal|high] [-t <timeout>]` | 提交 bash 脚本; 前台超过 5 分钟自动转后台 |
| `python gpu_backend/main.py info` | 查看 server 状态 / GPU 负载 |
| `python gpu_backend/main.py list-sessions` | 列出持久化 tmux 会话 |

## 调度规则 (submit 统一入队)

- 每 5s 扫 `nvidia-smi` VRAM → 队列最前任务的 `--gpu-nums` 需求满足 → pop 启动,
  由调度器注入 `CUDA_VISIBLE_DEVICES` (worker **不要**自己设)。
- 已运行任务不被抢占; `high` 只是插队。纯显存判断, 不看利用率。
- 默认非独占: 任务对分配卡持有 **20 分钟独占租约**, 到期后该卡对其他非独占任务开放共享;
  `--need-exclusive-gpu` 则独占整卡到任务 tmux 会话结束 (长挂 vLLM serve 用这个)。
- 单卡 (`--gpu-nums 1`) 是当前唯一经过端到端验证的形态; 多卡会分配显存但 pipeline
  仍是单进程训练。

## 关键注意

- **路径必须绝对化**: 脚本在 GPU 机器的 tmux 里执行, 相对路径会相对 server 进程的
  CWD 解析。共享 NFS 挂载下 `$SHARED_ROOT/...` 绝对路径两边都有效, 一律用绝对路径。
- worker_submit 已对 `--data-path` / `--training-config-path` 自动 resolve, 但脚本内部
  引用的路径仍需你自己写成绝对路径。
- 这就是 `worker_submit` 的底层通道: ExpOS FastAPI 把 `pipeline.sh` 交给 gpu_backend
  在 GPU 机上跑完 train → vLLM → eval 三阶段, `result.json` 里的 score 作为实验分。
