# cc_naive — agent 执行逻辑

> config `mode: cc_naive`。workspace 只注入 backend copy + README/BENCHMARK/GLOBAL_MEMORY，
> **不注入** experiment_os / SUBMIT_GUIDE.md / expos launcher。一个 CC 从开场到 FINISH 是
> **一条连续弧线**：self-plan（main）→ self-execute（worker）→ self-audit（reviewer），
> 全部在同一会话里完成，没有分支、没有别人接手你的笔记。

## 一、开场（agent 启动时看到什么）

Host 侧 `runner.run_agent` 拉起 `claude -p` 后，agent 的可见世界是：

- **cwd** = `workspace/`（唯一可见面；`configs/` 是隐藏的平级目录，白名单 deny `Bash(*<configs>*)*`）。
- **system prompt**（`--append-system-prompt`）= `README.md`，由 `seed.promptlib.compose("protocol_naive.md", backend_desc, eval_contract, submit_driver_naive)` 拼成：
  - `protocol_naive.md` —— 角色协议：单次连续 run 的研究方法论 + naive 的 workspace 布局；
  - `shared_context.md` —— 环境/工具/规则（backend 模块描述、训练落地建议、泄漏边界、时间纪律、eval contract、精确 pkill）。
  - naive 的 `{submit_driver}` 填的是**手搓管线**说明（自写训练/serve/eval 脚本用 `public/gpu-backend` 驱动，**无 `pipeline-submit` launcher / `submit_pipeline` 模块**），**没有任何 `./expos` / experiment_os 字样**（`test_workspace_seed.py` 断言了这点）。
- **stdin prompt** = `build_seed_task(config, mode)` 或 `run.task`：run 参数（benchmark / base_model / budget）+ “Begin: read README.md, BENCHMARK.md, public/README.md, GLOBAL_MEMORY.md …”。

## 二、agent 的执行循环（协议规定的 8 步）

`protocol_naive.md` 规定一个连续研究循环，每一轮都必须改变理解，否则收敛收尾：

1. **Frame the hypothesis** — 一句话假设，写进 `AGENT.md`；先读 `BENCHMARK.md`（grader 解析的答案格式）。
2. **Design one clean experiment** — 一次只改一个有意义的变量；预算内估训练时间。
3. **Data quality gate** — 先对 5–10 行做 train-inference 一致性检查（输入侧 prompt 模板一致、输出侧能被 grader 判对，如 aime2025 要 `ANSWER: <int>` 独占一行）；筛脏数据；泄漏 → 记 Warnings 一行并丢弃。
4. **Run it** — 提交训练后**不守着**：去准备下一个实验 / 读已完成 eval 的轨迹。两条提交路径（见下）。
5. **Diagnose** — 读 `samples.jsonl` / `metrics.json` / `run_eval.log`，判断失败是“没学到”还是“格式漂移”，把诊断**喂回数据**。
6. **Converge on the bottleneck** — 自审 `data → config → score → trajectories → bottleneck`，只追一个最值得修的阻塞点；每个结论必须由真实 artifact 支撑。
7. **Keep notes fresh** — `AGENT.md` 五段维护 + append-only `## Sessions` 尾段；跨刷新经验进 `GLOBAL_MEMORY.md`。
8. **Finish** — `echo <experiment_id> > FINISH` 是**最后一个动作**，不是提前逃跑。

## 三、训练/评估——手搓管线（无现成 launcher）

cc_naive **没有现成的 submit-train launcher**（不 seed `submit_pipeline` 模块 / `pipeline-submit`），
agent 自写训练/serve/eval 三步并用 `public/gpu-backend` 驱动（`protocol_naive.md` §4）：

```bash
# 1) 训练 —— 手写 bash 脚本（读 public/backend/slime/ 的 README + launch_sft.sh），然后：
public/gpu-backend submit -p train.sh       # --gpu-nums / priority 经 flags
# 2) serve checkpoint（public/backend/vllm/） 3) eval（public/backend/eval_all/）
# 4) 读 result.json + eval 轨迹（samples.jsonl / metrics.json / run_eval.log）
```

`public/backend/*` 下 seed 了框架模块（`slime/` / `eval_all/` / `vllm/` / `search_cli/`）
供 agent 读文档 + 调用，但**驱动管线的是 agent 自己**——这正是 naive 基线的教学点：
与 with_expos 版（`./expos worker submit` 一键管线）形成"有无结构化编排"的对照。
每个实验的产物放 `workspace/experiments/exp_####/`：`work_order.json` / `result.json` /
`metrics.json` / `samples.jsonl` / 各阶段日志。eval contract 的 flags（`--bench-id` /
`--limit` / `--serve-util` / `--max-model-len`）必须原样传（来自 README 的 Run Contract 段）。

训练/评测都异步打 gpu_backend（:8989）；agent 用 `public/gpu-backend script-status` /
`script-log` 轮询（协议明确禁止 sleep-poll 循环）。

## 四、退出语义

| 条件 | stop_reason | host 行为 |
|---|---|---|
| `echo <id> > FINISH`（内容=提交的实验/ckpt） | `done_marker` | 等 ≤3s 读内容 → killpg 回收 |
| `claude -p` 自然退出 | `completed` | 直接收尾 |
| `single_agent_running_time_seconds` 预算耗尽 | `timeout` | SIGTERM → 15s → SIGKILL |
| `max_running_time_seconds` 全局 deadline | `cancelled` | 同上 |

非 `completed` 一律整树回收；`summary.md` 记录 stop_reason / returncode / elapsed / FINISH content / session_id / log 路径。score 因为提交是 async 的，落在 `workspace/experiments/<id>/result.json`，summary 只记指针。

## 五、agent 侧硬约束（协议 + shared context）

- **禁止 sleep-poll 循环**、**禁止相同重提交**、**禁止编造 eval 分数**、**禁止单实验追多假设**、**禁止 eval/test 数据进训练**（泄漏）、**禁止盲目重试**、**禁止提前写 FINISH**、**禁止只看分数不读轨迹就下结论**、**禁止 `pkill -9 python` 无差别杀进程**。
- **禁止设 `CUDA_VISIBLE_DEVICES`** —— gpu_backend scheduler 自己选卡注入，agent 只管 `--gpu-nums` / `public/gpu-backend submit -g <n>`。
- **必须**：先读 BENCHMARK/README/public/README；开工前把假设写进 `AGENT.md`；数据质量门过完才烧 GPU；每个 checkpoint 出分前先分析轨迹。
