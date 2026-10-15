# 统一评测接口使用指南

> 回答三个问题：
> **① 统一评测接口怎么用？** ② 怎么把一个 benchmark 接入并校验？ ③ 怎么用一个新的 benchmark？
>
> 配套文档：[`framework/README.md`](./README.md)（框架结构与全量脚本清单）、
> [`format/ADD_BENCHMARK.md`](../format/ADD_BENCHMARK.md)（手写接入步骤）、
> [`format/TEST_BENCHMARK.md`](../format/TEST_BENCHMARK.md)（评测用法）。

---

## 0. 三条命令先记住

```bash
# ① 用统一接口跑一个 bench（仓库根目录执行）
framework/.venv/bin/python framework/run_eval.py <benchmark-id> --limit 500 --trial-name my-trial

# ② 不调模型，校验解析 + 数据 schema
framework/.venv/bin/python framework/run_eval.py <benchmark-id> --dry-run --validate-data

# ③ 列出所有已接入的 bench
framework/.venv/bin/python framework/run_eval.py --list
```

凭据**只能通过环境变量注入**（不得写入 JSON/YAML/日志）：

```bash
export OPENAI_API_BASE="https://<endpoint>/v1"
export OPENAI_API_TOKEN="dummy-noauth"     # 免鉴权端点用占位符；带鉴权端点填真实 token
export MODEL_NAME="<model-id>"
# 也可用 Anthropic 兼容端点：ANTHROPIC_API_BASE / ANTHROPIC_API_TOKEN
```

---

## 1. 统一评测接口如何使用

### 1.1 入口与运行环境

| 项 | 值 |
|---|---|
| 统一入口 | `framework/run_eval.py`（**在仓库根目录运行**；它从自身位置推导 `format/`，不依赖 cwd） |
| 解释器 | `framework/.venv/bin/python`（共享 venv；3 个手写 legacy gsm8k/gpqa/aime2025 用各自的 `.venv`） |
| 参数来源 | 环境变量 → 命令行参数 → 该 bench 的 `config/config.yaml` |
| 凭据注入 | 只走环境变量或 `--openai-api-* / --anthropic-api-*` 参数，从不落盘 |

### 1.2 跑一轮评测

```bash
framework/.venv/bin/python framework/run_eval.py <benchmark-id> \
  --limit 500 --trial-name my-trial
```

| 参数 | 说明 |
|---|---|
| `<benchmark-id>` | 位置参数；也支持 `--benchmark-id` 或环境变量 `BENCHMARK_ID` |
| `--limit N` | 跑多少题。**自动接入的薄 runner 默认只跑 3 题，务必显式传** |
| `--trial-name` | 输出目录名（默认随机 `default_xxxx`） |
| `--split` | 覆盖数据分片（默认 `default` 或 spec 声明的第一个） |
| `--sample-indices "0,5,12"` | 只重跑指定 index（修复 / 定向复查场景） |
| `--config-path` | 额外 YAML，`deep_merge` 覆盖该 bench 的 config |
| `--model` | 覆盖模型名（等价 `MODEL_NAME`） |
| `--openai-api-base/--openai-api-token/--anthropic-*` | 覆盖端点 / 凭据（优先于环境变量） |
| `--dry-run` | 只解析、不写产物、不调用模型 |
| `--validate-data` | 配合 `--dry-run`：流式读一条本地数据，验证 prompt/target/choices 映射 |

3 个手写 legacy 走同一入口（adapter=`legacy_runner`）：

```bash
framework/.venv/bin/python framework/run_eval.py gsm8k --limit 100 --trial-name gsm8k-smoke
```

### 1.3 先 `--dry-run` 再真跑

```bash
framework/.venv/bin/python framework/run_eval.py bigbench-arithmetic --dry-run --validate-data
```

- `--dry-run`：解析出该 bench 的 adapter / scorer / 数据集 / 生效配置，确认一切可解析。
- 加 `--validate-data`：再从本地数据流式读一行，跑一遍字段映射，确认 schema 正确。
- 都通过后再真跑，避免空跑浪费配额。

### 1.4 列出 / 筛选 benchmark

```bash
framework/.venv/bin/python framework/run_eval.py --list                     # 全部
framework/.venv/bin/python framework/run_eval.py --list --status runnable   # 只列可跑的
framework/.venv/bin/python framework/run_eval.py --list --status catalogued # 已入库未接入的
framework/.venv/bin/python framework/run_eval.py --list --domain math_reasoning
framework/.venv/bin/python framework/run_eval.py --list --eval-method exact_match
```

`--status` 取值：`catalogued`（已登记）→ `runnable`（已接入可跑）→ `verified`（已验证）。

### 1.5 产物

```
data/output/<benchmark-id>/<trial-name>/
├── samples.jsonl   # 逐题轨迹：response / reasoning_content / finish_reason / usage /
│                   #   complete / incomplete_reason / targets / correct / provider
├── reward.json     # 标量分数 + meta_data（attempted / completed / incomplete /
│                   #   correct / observed_reward / failures ...）
└── config.yaml     # 本次实际生效配置（已脱敏，凭据不落盘）
```

### 1.6 批量评测

```bash
framework/.venv/bin/python framework/automation/batch_run.py \
  --limit 500 --max-concurrency 150 --workers-per-bench 10 \
  --trial qwen36-20260808 \
  --openai-api-base "https://<endpoint>/v1" --openai-api-token dummy-noauth \
  --model Qwen3.6-35B-A3B \
  --config-override data/config-overrides/qwen36.yaml \
  --timeout-hours 6.0
```

- 自动接入的 241 个用 `batch_run.py`；3 个 legacy 用 `automation/run_legacy_batch.py`。
- 断点续跑：状态在 `data/output/<trial>/state.json`，同样命令重跑即续。
- 完整流水线（冒烟 → 批跑 → 空答修复 → 判分 → 汇总 → HTML 报告）见 `framework/README.md` §“完整评测流水线”。

---

## 2. 如何接入并校验一个 bench

接入前先决定走哪条路：

| 路径 | 适用 | 产物来源 |
|---|---|---|
| **手写接入** | 特殊格式 / 自定义打分 / 数据不在 HF（如 gsm8k/gpqa/aime2025） | 自己按规范写 |
| **自动接入** | HF 数据集、标准字段（prompt/target/choices）、确定性打分 | `framework/automation/` 生成器生成 |

两种路径产出的目录结构一致（见下），只是文件的“作者”不同。**数据必须本地化**：
数据落到 `data/datasets/<benchmark-id>/<split>.jsonl`（一行一个样本 dict），运行时优先读本地，
HF 原始信息完整保留在 `spec.json` 的 `dataset.hf` 段。

```
format/<benchmark-id>/
├── config.json            # 清单：benchmark-id / benchmark-name / source / split / judge_quality_score
├── config/config.yaml     # 生效配置：general 段由框架注入，custom 段放自定义参数
├── spec.json              # 声明式数据契约（adapter / fields / scorer / dataset），运行时唯一事实来源
├── reward/reward.json     # 输出契约样例（runner 在 output_dir 写出同形状 reward.json）
├── run_benchmark.py       # 薄包装（自动接入的转给 core.generic_runner）
└── BENCHMARK.md           # 5 段式说明文档
```

### 2.1 路径 A：自动接入（推荐，HF 数据集）

```bash
# 1) 建注册表（把候选 bench 的声明式 spec 汇成一行一条的 registry）
framework/.venv/bin/python framework/automation/build_registry.py \
  --inputs <registry.jsonl...> --output framework/automation/benchmark_specs.jsonl --target 200

# 2) 生成 format/<id>/ 目录（spec.json / config.yaml / run_benchmark.py / config.json）
framework/.venv/bin/python framework/automation/generate_integrations.py \
  --registry framework/automation/benchmark_specs.jsonl --format-dir format \
  --replace-generated --total 200

# 3) 下载本地数据并重写 spec.json 的 dataset 段（保留 hf 元数据）
framework/.venv/bin/python framework/automation/download_datasets.py \
  --format-dir format --data-dir data/datasets --hf-cache data/cache
```

> **数量口径**：当前 `format/` 下自动接入共 **241 个**（165 个 BIG-bench + 76 个独立 HF 数据集），
> 加 4 个手写 legacy = **245 个可跑**（`run_eval.py --list` 核实）。
> 而 `benchmark_specs.jsonl` 目前只有 200 条，生成器重建只覆盖 `registry` 里的部分；
> 要重建全量，先扩充 registry 再让 `--total` = 自动接入数 + 4（与 `--list` 总数一致）。

`spec.json` 是运行时唯一事实来源，benchmark 专属逻辑应写在这里而不是改核心代码：

```json
{
  "adapter": "bigbench",
  "fields": { "prompt": "inputs", "target": "targets", "choices": "multiple_choice_targets" },
  "scorer": "multiple_choice",
  "dataset": {
    "path": "data/datasets/bigbench-arithmetic",
    "split": "train",
    "local_rows": 12019,
    "hf": { "path": "tasksource/bigbench", "name": "arithmetic" }
  },
  "license": "...", "domain": "..."
}
```

### 2.2 路径 B：手写接入（特殊 benchmark）

按 [`format/ADD_BENCHMARK.md`](../format/ADD_BENCHMARK.md) 的 5 步写：
目录 → `config.json` → `config/config.yaml` → `spec.json` → `run_benchmark.py` + `reward/reward.json` → `BENCHMARK.md`。
可照抄 `format/benchmark_example/` 模板，gsm8k/gpqa/aime2025 是手写范例。

### 2.3 校验门（接入完成后逐道过）

```bash
# 门 1：解析校验（不调模型）
framework/.venv/bin/python framework/run_eval.py <benchmark-id> --dry-run

# 门 2：数据 schema 校验（流式读一条本地数据）
framework/.venv/bin/python framework/run_eval.py <benchmark-id> --dry-run --validate-data

# 门 3：端到端冒烟（真跑 3 题，必须能产出 reward.json）
framework/.venv/bin/python framework/run_eval.py <benchmark-id> --limit 3 --trial-name smoke

# 门 4：目录与注册表一致性（全量；--expected 与 `run_eval.py --list` 总数一致）
framework/.venv/bin/python framework/automation/validate_integrations.py --format-dir format --expected 245

# 门 5：本地数据行数与 schema（全量）
framework/.venv/bin/python framework/automation/validate_local_datasets.py --format-dir format --data-dir data/datasets

# 门 6：BENCHMARK.md 结构（5 段式、无占位符）
framework/.venv/bin/python framework/automation/verify_benchmark_md.py --format-dir format --ids <benchmark-id>

# 门 7：框架单元测试
framework/.venv/bin/python -m unittest discover -s framework/tests
```

`validate_datasets.py`（流式读每条生成接入、校验字段映射）作为在线数据侧的补充校验。

---

## 3. 如何使用一个新的 bench

一条完整旅程：**发现候选 → 决定接入方式 → 本地化数据 → 过校验门 → 跑一轮评测 → 看结果**。

### 3.1 候选从哪来

- **已在 `format/` 里**：直接跳到 3.4，不用接入。
- **发现流水线的产物**：`swe_autobenchmark_finding/stage_2/data/<benchmark-id>/` 下有
  `config.json`（manifest）+ `BENCHMARK.md`（5 段式说明）+ `raw/`（已下载的原始数据），
  是**已核实但尚未接入**的候选库（721 个 verified）。
- **其他 HF 数据集**：手动挑选后走 2.1 自动接入。

### 3.2 决定接入方式

看候选的 `BENCHMARK.md` / `config.json`：

- 数据在 HF、字段标准、可用确定性打分 → **自动接入**（§2.1）。
- 需要特殊 prompt / 自定义打分 / 隐藏测试执行 → **手写接入**（§2.2）。
- 打分器清单（`core/scoring.py`）：`multiple_choice` / `exact_match` / `numeric` / `boolean` /
  `squad` / `label` / `json` / `unscored`（unscored 需另行提供 bleu/rouge 等官方指标）。

### 3.3 本地化数据（硬性要求）

```bash
framework/.venv/bin/python framework/automation/download_datasets.py \
  --format-dir format --data-dir data/datasets --hf-cache data/cache --ids <benchmark-id>
```

产出 `data/datasets/<benchmark-id>/<split>.jsonl`，并在 `spec.json` 的 `dataset` 段指向本地路径。
漏掉这一步，运行时可能退回到在线流式加载，网络抖动 / 限流会拖垮评测。

### 3.4 跑一轮评测

```bash
export OPENAI_API_BASE="https://<endpoint>/v1"
export OPENAI_API_TOKEN="dummy-noauth"
export MODEL_NAME="<model-id>"

# 冒烟 3 题
framework/.venv/bin/python framework/run_eval.py <benchmark-id> --limit 3 --trial-name smoke

# 正式跑（显式 limit 才有全量）
framework/.venv/bin/python framework/run_eval.py <benchmark-id> --limit 500 --trial-name run-1
```

### 3.5 看结果

```bash
# 单点结果
ls data/output/<benchmark-id>/run-1/
cat data/output/<benchmark-id>/run-1/reward.json

# 批量汇总 + HTML 报告
framework/.venv/bin/python framework/automation/summarize_batch.py --trial run-1 \
  --format-dir format --out data/output/<benchmark-id>/run-1/analysis.json
framework/.venv/bin/python framework/automation/build_report_html.py \
  --analysis data/output/<benchmark-id>/run-1/analysis.json \
  --format-dir format --out data/output/<benchmark-id>/run-1/report.html
```

---

## 4. 路径速查

| 想做 | 命令 |
|---|---|
| 列出所有 bench | `framework/.venv/bin/python framework/run_eval.py --list` |
| 跑一个 bench | `framework/.venv/bin/python framework/run_eval.py <id> --limit N --trial-name T` |
| 只校验不动模型 | `framework/.venv/bin/python framework/run_eval.py <id> --dry-run --validate-data` |
| 批量跑 | `framework/.venv/bin/python framework/automation/batch_run.py ...` |
| 自动接入 | `build_registry.py` → `generate_integrations.py` → `download_datasets.py` |
| 手写接入 | 按 `format/ADD_BENCHMARK.md` |
| 接入后校验 | §2.3 的 7 道门 |
| 数据 | `data/datasets/<id>/<split>.jsonl` |
| 结果 | `data/output/<id>/<trial>/{samples.jsonl, reward.json, config.yaml}` |
