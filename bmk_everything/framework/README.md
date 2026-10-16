# framework/ 评测框架

BmkEverything 的评测框架：**核心运行时**（`core/`）+ **自动化脚本**（`automation/`）+ **单元测试**（`tests/`）。
候选开源包包含论文展示结果的 24 个评测定义；数据、外部评测器和其环境需单独准备。
请先阅读 [安装与路径配置](../SETUP.md) 和 `../paper_manifest.json`，准备所选评测的数据和 `framework/.venv`。
`BENCHMARK_ROOT` 控制评测定义及通用 runner 的数据根目录；未设置时使用本目录的父目录。

> 使用教程（统一接口怎么用、benchmark 怎么接入/校验、新 bench 怎么用）见 **[`framework/USAGE.md`](./USAGE.md)**。

## 目录结构

```
framework/
├── core/                # 运行时库
│   ├── provider.py      #   模型调用：OpenAI 兼容 / Anthropic，从环境变量解析端点与凭据
│   ├── scoring.py       #   确定性打分器（exact_match / multiple_choice / numeric / ...）
│   ├── adapters.py      #   数据适配：bigbench / hf_* / legacy_runner
│   ├── generic_runner.py#   通用 runner：加载本地数据 → 并发调模型 → 判分 → 写 samples/reward
│   ├── catalog.py       #   format/ 目录发现与清单
│   ├── config.py        #   YAML 配置加载 / deep_merge / 凭据脱敏
│   └── legacy_bridge.py #   3 个手写 legacy（gsm8k/gpqa/aime2025）桥接
├── automation/          # 生成 / 下载 / 校验 / 批跑 / 修复 / 汇总 / 报告脚本（见下表）
├── tests/               # 单元测试
└── .venv/               # 共享 venv（datasets / pyyaml / requests 等）
```

`framework/run_eval.py` 是**统一评测入口**（从仓库根目录运行），把上面所有环节串起来；单个 benchmark 目录里的 `run_benchmark.py` 是薄包装，把请求转给 `core.generic_runner`。

## 快速开始：单点跑一个 benchmark

凭据**只能通过环境变量注入**（不得写入 JSON/YAML/日志），先导出：

```bash
export OPENAI_API_BASE="https://your-router.example.com/v1"
export OPENAI_API_TOKEN="dummy-noauth"        # 免鉴权端点；带鉴权端点填真实 token
export MODEL_NAME="Qwen3.6-35B-A3B"
```

然后在**仓库根目录**运行（`framework/run_eval.py` 从自身位置推导 `format/`，不依赖 cwd）：

```bash
framework/.venv/bin/python framework/run_eval.py <benchmark-id> --limit 500 --trial-name my-trial
```

| 参数 | 说明 |
|---|---|
| `<benchmark-id>` | 位置参数，也支持 `--benchmark-id` / `BENCHMARK_ID` 环境变量 |
| `--limit N` | 要跑的题数。**薄 runner 默认只跑 3 题，务必显式传** |
| `--trial-name` | 输出目录名（默认随机 `default_xxxx`） |
| `--split` | 覆盖数据分片（默认 `default` 或 spec 声明的第一个） |
| `--sample-indices "0,5,12"` | 只重跑指定 index（修复场景用） |
| `--config-path` | 额外 YAML，deep_merge 覆盖该 bench 的 config |
| `--model` | 覆盖模型名（也可用 `MODEL_NAME`） |
| `--openai-api-base/--openai-api-token/--anthropic-*` | 覆盖端点/凭据（优先于环境变量） |
| `--dry-run` | 只解析、不调用模型；加 `--validate-data` 再流式读一条本地数据验证 schema |

先 `--dry-run` 看解析结果再真跑：

```bash
framework/.venv/bin/python framework/run_eval.py bigbench-arithmetic --dry-run --validate-data
```

3 个手写 legacy（gsm8k / gpqa / aime2025）也走同一入口（adapter=`legacy_runner`），用各自的 `run_benchmark.py`：

```bash
framework/.venv/bin/python framework/run_eval.py gsm8k --limit 100 --trial-name gsm8k-smoke
```

### 产物

```
data/output/<benchmark-id>/<trial-name>/
├── samples.jsonl   # 逐题轨迹：response / reasoning_content / finish_reason / usage /
│                   #   complete / incomplete_reason / targets / correct / provider
├── reward.json     # 标量分数 + meta_data（attempted / completed / incomplete /
│                   #   correct / observed_reward / failures ...）
└── config.yaml     # 本次实际生效配置（已脱敏）
```

### 列出 / 筛选 benchmark

```bash
framework/.venv/bin/python framework/run_eval.py --list                     # 全部
framework/.venv/bin/python framework/run_eval.py --list --status runnable   # 只列可跑
framework/.venv/bin/python framework/run_eval.py --list --domain math_reasoning
```

## 批量评测（全量 244 个）

并行批跑用 `automation/batch_run.py`：按 `max_concurrency // workers_per_bench` 个并发子进程（每个内部 `max_workers` 线程）控制总并发，断点续跑，产物写到 `data/output/<id>/<trial>/`：

```bash
framework/.venv/bin/python framework/automation/batch_run.py \
  --limit 500 --max-concurrency 150 --workers-per-bench 10 \
  --trial qwen36-20260808 \
  --openai-api-base "https://.../v1" --openai-api-token dummy-noauth \
  --model Qwen3.6-35B-A3B \
  --config-override data/config-overrides/qwen36.yaml \
  --timeout-hours 6.0
```

- `--limit N`：数据集 ≤N 行全量跑，>N 行抽前 N 题
- 断点续跑：状态在 `data/output/qwen36-batch/<trial>/state.json`，同样命令重跑即续
- `--benchmarks id1,id2`：只跑子集（单点也能用它）
- 3 个 legacy 被 batch_run 跳过（无 spec.json），用 `automation/run_legacy_batch.py` 单独跑

## 完整评测流水线（以 Qwen3.6 全量 244-bench 为例）

```
1. 冒烟       run_model_smoke.py / model_smoke_plan.py
2. 全量批跑    batch_run.py + run_legacy_batch.py
3. 空答修复    repair_qwen36_batch.py --repair-override data/config-overrides/qwen36-repair3.yaml
              repair_legacy_batch.py   （只重跑空答 index，merge 回原 trial）
4. 空答判分    apply_failure_scoring.py  --dry-run 先预览
   # 规则：空答（finish_reason=length，content 空）= 模型失败 = 判 0 分计入分母；
   #       request_error（传输故障）保持 incomplete、不进分母
5. 汇总        summarize_batch.py --trial qwen36-20260808 --format-dir format \
                  --out data/output/qwen36-batch/qwen36-20260808/analysis-final.json
6. 报告        build_report_html.py --analysis <analysis-final.json> --format-dir format \
                  --out data/output/qwen36-batch/qwen36-20260808/benchmarks_report.html
```

修复覆盖配置：`data/config-overrides/<name>.yaml` 会 `deep_merge` 到每个 benchmark 的 `config.yaml`。
`qwen36-repair.yaml` = max_tokens 32768（reasoning 模型 thinking 与答案共享预算，硬题会耗尽）；
`qwen36-repair3.yaml` = max_tokens 65536（前两轮仍被 chess-state-tracking 等"预算黑洞"耗尽）。

## 打分器

确定性打分器全部在 `core/scoring.py`，注册表在 `generic_runner.py:26`。打分前统一做
`normalize()`（NFKC / 小写 / 空白折叠）+ `final_text()`（取 `final answer:` 之后的内容）。

| 打分器 | 逻辑 | 本轮用量 |
|---|---|---|
| `multiple_choice` | 字母（A-Z）落候选集，或全文与候选相等 | 135 bench |
| `exact_match` | normalized 全文与任一 target 精确相等 | 37 |
| `unscored` | 恒 None，不出 reward（需 bleu/rouge/语义等官方指标） | 15 |
| `numeric` | 取最后一个数字，容差 1e-6 匹配 | 8 |
| `squad` | SQuAD 规范化 + unanswerable 分支 | 3 |
| `boolean` | yes/no/true/false/1/0 别名折叠 | 1 |
| `label` | 词干前缀匹配 | 1 |
| `json` | 解析 JSON 结构相等（已注册，暂无任务用） | 0 |

分布与均分详见 `BENCHMARK_ANALYSIS.md` §2.2 / §7.4。

## 数据与 spec

- 数据：`data/datasets/<id>/<split>.jsonl`，一行一个样本 dict（`inputs` / `targets` / `multiple_choice_targets` 等）
- `format/<id>/spec.json` 声明：`adapter`、`fields`（prompt/target/choices 字段映射）、`scorer`、
  `dataset`（本地路径 + 原始 HF 元数据）、`license`、`domain`、`local_rows`
- `format/<id>/config/config.yaml` 是该 bench 生效配置；运行时可用 `--config-path` / `config-override` 覆盖

数据下载（241 个自动接入）与重建：

```bash
framework/.venv/bin/python framework/automation/build_registry.py \
  --inputs data/benchmark-survey/registry/selected_200_open_verified.jsonl \
           data/benchmark-survey/registry/revalidated_open.jsonl \
  --output framework/automation/benchmark_specs.jsonl --target 200

framework/.venv/bin/python framework/automation/generate_integrations.py \
  --registry framework/automation/benchmark_specs.jsonl --format-dir format \
  --replace-generated --total 200

framework/.venv/bin/python framework/automation/download_datasets.py \
  --format-dir format --data-dir data/datasets --hf-cache data/cache
```

> 数量口径：`format/` 现有 241 个自动接入（165 个 BIG-bench + 76 个独立 HF），加 4 个 legacy = 245 个可跑
> （`run_eval.py --list` 核实）。`benchmark_specs.jsonl` 只有 200 条，生成器重建只覆盖 registry 内；
> 重建全量前先扩充 registry，`--total` = 自动接入数 + 4。
> 校验目录数用 `validate_integrations.py --expected 245`。

## 常用脚本速查

| 脚本 | 作用 |
|---|---|
| `batch_run.py` | 多进程并行批跑（自动接入 241 个） |
| `run_legacy_batch.py` | 批跑 3 个手写 legacy |
| `repair_qwen36_batch.py` / `repair_legacy_batch.py` | 只重跑空答 index（更高 max_tokens）并 merge 重算 |
| `apply_failure_scoring.py` | 空答重分类为模型失败（`correct=false` + `scored_as_failure=true`），幂等，`--dry-run` 预览 |
| `summarize_batch.py` | 聚合 → JSON + markdown 片段（分领域/打分器/Top-Bottom） |
| `build_report_html.py` | 生成单文件 HTML 报告（每 bench 卡片 + BENCHMARK.md） |
| `download_datasets.py` | 下载 HF 数据集到本地 JSONL 并重写 spec.json dataset 段 |
| `validate_integrations.py` | 校验 245 个目录与注册表（`--expected 245`） |
| `validate_local_datasets.py` | 校验本地数据行数与 schema |
| `verify_benchmark_md.py` | 校验各 bench 的 BENCHMARK.md 结构 |
| `model_smoke_plan.py` / `run_model_smoke.py` | 付费冒烟计划 / 执行 |
| `preflight_kimi.py` | Kimi 凭据预检 |

## 凭据安全

- 端点/凭据/模型名只从**环境变量或命令行参数**注入：`OPENAI_API_BASE`、`OPENAI_API_TOKEN`、
  `ANTHROPIC_API_BASE`、`ANTHROPIC_API_TOKEN`、`MODEL_NAME`
- `config.yaml` 落盘前统一 `redact()` 脱敏，凭据绝不写入 JSON/YAML/日志
- 免鉴权端点 token 用 `dummy-noauth` 占位

## 测试

```bash
framework/.venv/bin/python -m unittest discover -s framework/tests
```
