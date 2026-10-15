# cc_rollout — 单 CC 教学式 Rollout Playground

cc_rollout 是 RSI-Master 的一个 **教学式（baseline）post-training playground**：
不搞多 agent 编排，而是**直接拉起一个 `claude -p` CLI subprocess**（一个 CC），
让它以**单一路径、连续、不 branching** 的方式跑完一整个研究循环。参考实现是
`playground/cc_branching`（CC 的启动本质就是启动一个 CLI subprocess）。

config 的 `mode:` 字段选择两种 rollout 方式：

| mode | workspace 注入 | agent 怎么执行 | 提交训练 |
|---|---|---|---|
| `cc_naive` | 只做 backend copy + README/BENCHMARK/GLOBAL_MEMORY 注入，**无** experiment_os/SUBMIT_GUIDE，**也无** submit_pipeline/pipeline-submit launcher（手搓基线） | 一个 CC 一路 Rollout 到底，self-plan/self-execute/self-audit | 手写训练/serve/eval 脚本用 `public/gpu-backend` 驱动 |
| `cc_with_expos` | 同 naive + experiment_os + SUBMIT_GUIDE.md + `./expos` CLI launcher | 一个 CC **兼任 MainAgent/Worker/Reviewer 三角色**，通过 `./expos` CLI 调多角色工具 | `./expos worker submit`（submit 训练管线） |

## 文档导航

- **[`cc_naive.md`](cc_naive.md)** — cc_naive 模式下 agent 的执行逻辑（研究循环 / 两条提交路径 / 退出语义）。
- **[`cc_with_expos.md`](cc_with_expos.md)** — cc_with_expos 模式下 agent 的执行逻辑（三角色框架 / `./expos` 工具接口 / worker submit 全链路 / reviewer 闭环）。
- **[`expos_cli.md`](expos_cli.md)** — CLI 形式 ExpOS 的内部机制（launcher / `SubmitPipelineBackend` / 与 v2 ExpOS 的差异）。

## 如何运行

```bash
# cc_naive（默认 mode）
python run.py --agent cc_rollout \
  --config configs/cc_rollout/config.yaml \
  --task "Prepare a PostTrainBench attempt for aime2025 with clean final answers." \
  --set run.single_agent_running_time_seconds=3600

# cc_with_expos
python run.py --agent cc_rollout \
  --config configs/cc_rollout/aime2025_35b_with_expos_0901.yaml \
  --task "..." --set run.single_agent_running_time_seconds=3600
```

> **config 约定** (仿 `configs/rsi_master`): 每个 config 只关注一个实验,
> 独立后缀命名 `<bench>_<size>_<mode>_<date>.yaml` (如
> `aime2025_35b_with_expos_0901.yaml` / `aime2025_35b_naive_0901.yaml`)。
> `config.yaml`(naive) / `config_cc_with_expos.yaml` 只是模板/默认入口,
> **真实实验用 per-experiment 快照, 不要改模板**。新实验: 复制对应模板,
> 改 `run.base_model` + `eval.bench_id` + `run.run_id` + 文件后缀。

也支持直接跑 rollout CLI（不经 run.py 的 adapter）：

```bash
python -m playground.cc_rollout.core.rollout run --config configs/cc_rollout/config.yaml
python -m playground.cc_rollout.core.rollout health   # 检查 gpu_backend :8989 健康
```

## run 目录布局（单 agent，无 branches）

run.py 的 run_dir（`runs/cc_rollout_<ts>/`，RSI_MASTER_RUN_ROOT）下面再嵌套一层
orchestrator 建的 `<run_id>_<timestamp>/`（见 `orchestrator.run`；与 cc_branching 同款）：

```
runs/<run.py_ts>/                        # run.py run_dir = RSI_MASTER_RUN_ROOT
  config.override.yaml                   # (经 run.py --set 时) 覆盖后快照
  <run_id>_<timestamp>/                  # orchestrator 的 run_root
    config.yaml                          # 输入 config 快照（adapter 注入 run.task 后）
    workspace/                           # agent cwd（唯一可见面）
      README.md                          # 组合后的协议 prompt（= claude 的 --append-system-prompt）
      BENCHMARK.md                       # eval 权威 spec（benchmark root format/<bench_id>/ 拷贝）
      GLOBAL_MEMORY.md                   # 跨上下文刷新持久记忆 stub
      AGENT.md                           # agent 自己写的 living handoff（运行中产生）
      FINISH                             # agent 丝滑退出标记：echo <experiment_id> > FINISH
      public/                            # backend 只读拷贝 + gpu-backend launcher (pipeline-submit 仅 with_expos seed 时出现)
      public/cc_rollout_expos/           # (仅 cc_with_expos) 顶层独立包副本（./expos 执行的就是它）
      experiment_os/                     # (仅 cc_with_expos) ExpOS DAG：state/worker/experiments/reviews/data_pool
      SUBMIT_GUIDE.md                    # (仅 cc_with_expos) ./expos 的 submit 工作流指南
      expos                              # (仅 cc_with_expos) expos CLI launcher（755）
    configs/                             # claude 不可见（CLAUDE_CONFIG_DIR 指这里，transcript 存这）
      settings.json                      # 权限白名单（--settings 传入）
      run.json                           # {mode, session_id, task, ...}
      .claude-config/                    # CC 会话 transcript
    logs/agent.log                       # claude -p stdout/stderr tee
    summary.md                           # orchestrator 写的收尾报告
```

## Host 侧启动机制（两模式共享）

`playground/cc_rollout/core/rollout/runner.py::run_agent` 是核心：

1. **命令构造**：`claude -p --session-id <uuid> --append-system-prompt <workspace/README.md> [--dangerously-skip-permissions|--permission-mode <m>] --settings <configs/settings.json> [--add-dir <public_dirs...>]`。
   - `bypassPermissions` 模式用 `--dangerously-skip-permissions`；配合 settings.json 里的 **deny 规则** 作为唯一生效的权限层（见 `permissions.py`）。
   - `--add-dir` 是 `nargs='+'`，会吞掉尾随 positional，所以 **prompt 走 stdin**，不挂在命令行。
2. **进程**：`cwd = workspace`，`start_new_session=True`（独立进程组，`killpg` 可整树回收）；stdout/stderr tee 到 `logs/agent.log`；stdin 喂 task prompt。
3. **env**：`CLAUDE_CONFIG_DIR=configs/.claude-config`（transcript 放 agent 够不到的地方）；root + bypass 需 `IS_SANDBOX=1`；`ANTHROPIC_BASE_URL/AUTH_TOKEN/MODEL` 从 `config.claude` 注入（sidecar LLM）。
4. **退出四态**（每 1s 轮询）：
   - agent 写 `workspace/FINISH` → `done_marker`（先等最多 3s 让 `echo` 重定向内容落全）；
   - `claude -p` 自然退出 → `completed`；
   - `single_agent_running_time_seconds` 预算耗尽 → `timeout`；
   - `max_running_time_seconds` 全局 deadline 触发（watchdog 线程置 `stop_event`）→ `cancelled`。
   非 `completed` 一律 `_terminate_tree`：SIGTERM → 15s 宽限 → SIGKILL（killpg）。
5. **收尾**：`orchestrator._write_summary` 写 `summary.md`（mode / stop_reason / returncode / elapsed / FINISH content / session_id / agent log 路径）。因为 submit 是 async 的，summary 不记 GPU score，只记指针。

## 白名单（继承 cc_branching，逐字移植）

`permissions.py::write_settings` 生成 `configs/settings.json`：

- `additionalDirectories` = `run.public_dirs`（benchmark root / public / .venv / conda env）——CC 能读这些宿主目录；
- `deny`（65 条）——`Read/Edit(../*)`、`Read/Edit(../**)`、`Read/Edit(../../*)`…（相对 workspace 向上逃逸），`Bash(*<configs>*)*`（configs/ 不可达），`_OS_ROOTS`（/root、/proc 等系统根）。

在 `--dangerously-skip-permissions` 下，**deny 规则是唯一绑定生效的权限层**，这是隔离 headless agent 的安全边界，不要绕过。

## 复用清单（为什么能快速搭起来）

runner / permissions / _paths / layout / backend_readmes / promptlib 全部**逐字移植或小改**自 `playground/cc_branching`；`expos.cli.main` 和 `backend/backend/submit_pipeline` 原样复用（后者经 `SubmitPipelineBackend` 注入，见 `expos_cli.md`）。新写部分集中在 `seed.py` 的 with_expos 分支、两份协议 prompt、GLOBAL_MEMORY.md stub、SUBMIT_GUIDE.md、顶层独立包 `tools/cc_rollout_expos/`（`expos_cli.py` + `expos_cli_backend.py`，seed 复制进 workspace）和两个 config。
