---
BACKEND_DESCRIPTIONS
QA 数据蒸馏流水线 (qa_distill) — 并发单轮 QA 蒸馏。配置 config && 写 prompt && 写输出解析器 → 批量并发生成 JSONL 问答训练数据。
Invoke: `python examples/example_qa_distill.py` (先 `cp config/config.example.yaml config/config.yaml` 填入 API)。
核心: `qa_distill/` 包 (LLMClient + LLMPipeline); 输出 `<output_dir>/<experiment_name>/result.jsonl`。
---

# qa_distill — 大批量 QA 数据蒸馏工具

> **一句话用法**: 配一个 YAML (模型 / 并发 / 输出), 写两个 prompt 模板 (system + user),
> 写一个输出解析器, 就能把任意输入**并发**蒸馏成 QA 训练数据 (JSONL)。

```
config && prompt && parser → LLMPipeline.run(data_pool, concurrency, extractor) → result.jsonl
```

本目录是 `thirdparty/agent-code-base` 中「数据蒸馏」核心链路的迁移: 只保留
`LLMClient` + `LLMPipeline` (及其依赖的 config / logging / feishu), 去掉
`draw` 等与本场景无关的组件。

---

## 目录结构

```
backend/backend/qa_distill/          # ← 本身就是可导入的包 (含 __init__.py)
├── README.md                  # 本文档
├── __init__.py                # 顶层导出: LLMPipeline / LLMClient / APITester / ...
├── pipeline.py                # ★ LLMPipeline: 并发蒸馏流水线 (主入口)
├── llm_client.py              # ★ LLMClient: 异步限速+重试的 API 客户端
├── apitest.py                 # (可选) APITester: 大批量前 API 连通性预检
├── utils/
│   ├── config.py              # Config / get_config: YAML dot-notation 配置
│   ├── logging.py             # get_logger: 统一日志
│   └── feishu.py              # (可选) FeishuBot: 长任务完成/失败飞书告警
├── config/
│   └── config.example.yaml    # 配置模板 (复制为 config.yaml 后填写)
├── prompts/
│   ├── system/qa_generator.md # 示例 system prompt (QA 生成, 含 {占位符})
│   └── user/qa_generator.md   # 示例 user prompt (含 {topic} / {difficulty} / {num})
└── examples/
    ├── example_qa_distill.py  # 最小可跑示例 (支持 --dry-run 不联网演示)
    └── data_pool_sample.jsonl # 示例输入数据
```

> 运行依赖不靠 requirements.txt —— 见下方「运行环境 (public/distill_env)」。

**两种导入方式** (包内全部用相对导入, 名字无关):

```python
# 方式 A: 仓库根在 PYTHONPATH 时
from backend.backend.qa_distill import LLMPipeline

# 方式 B: backend/backend 在 PYTHONPATH 时 (如直接 cd 进去跑)
from qa_distill import LLMPipeline
```

---

## 快速开始 (3 步 + 1 跑)

```bash
cd backend/backend/qa_distill

# 0) 激活运行环境 (见下方「运行环境」; 不激活就直接用 public/distill_env/bin/python)
source <repo>/public/distill_env/bin/activate

# 1) 配置: 复制模板并填入 api_key / base_url (或留空, 用环境变量 / .env)
# 推荐直接用外部 teacher API (见下方「本 run 推荐配置」); 不要本地起小模型当 teacher
cp config/config.example.yaml config/config.yaml

# 2) 写 prompt: 编辑 prompts/{system,user}/qa_generator.md (已有示例可直接用)

# 3) 写解析器: 见 examples/example_qa_distill.py 里的 make_extract_qa (按需改)

# 跑: 不联网也能用 --dry-run 把整条流水线跑一遍 (返回占位响应, 无需 API key)
python examples/example_qa_distill.py --dry-run

# 真实跑: 换成真实 API 后去掉 --dry-run
python examples/example_qa_distill.py --concurrency 10
```

默认输出到 `<qa_distill>/output/qa_distill_demo/result.jsonl`。

---

## 执行流程

```
data_pool 条目
   │  (system_prompt_kwargs / user_prompt_kwargs)
   ▼
system prompt 模板 ── .format(**kwargs) ──► 渲染后的 system message ─┐
user    prompt 模板 ── .format(**kwargs) ──► 渲染后的 user message ──┼─► LLMClient
                                                                      │   (全局限速 + 指数退避重试 + 超时保护)
   ▲                                                              ▼
   │                                                    模型响应 (response)
   └──────────────────────── 解析器 extract_function(response)
                                 │  (如提取 <question>/<answer> 标签)
                                 ▼
                  result.jsonl: {input, response, naive_response, extracted, timestamp}
```

- **单轮**: 每个条目 = 一次 system+user → response, 不做多轮对话/自反思 (多轮可各自写成
  独立条目, 或在外面包一层你自己的编排)。
- **并发**: `concurrency_limit` 控制同时进行中的请求数; 全部并发协程**共享同一个
  LLMClient**, 因此 `rate_limit` (每分钟请求数) 是**全局生效**的。
- **实时落盘**: 每完成一条就追加写入 `result.jsonl`, 中途中断不丢已完成结果。

---

## 配置文件参考 (`config/config.example.yaml`)

| 键 | 说明 | 默认 |
|---|---|---|
| `qa_distill.model.provider` | 服务商标识 (仅备注) | openai |
| `qa_distill.model.api_key` | API key; **留空回退到** `OPENAI_API_KEY` | "" |
| `qa_distill.model.base_url` | base URL; **留空回退到** `OPENAI_BASE_URL` | "" |
| `qa_distill.model.model` | 模型名 | glm-5.2 (the teacher router; 亦可 kimi-k2.7-code / deepseek-v4-flash / deepseek-v4-pro) |
| `qa_distill.model.rate_limit` | **全局**每分钟请求数 (所有并发共享) | 20 |
| `qa_distill.model.max_tokens` | 单次响应最大 token | 2048 |
| `qa_distill.model.temperature` | 采样温度 | 0.7 |
| `qa_distill.prompts.system_prompt_path` | system prompt 模板路径 | ../prompts/system/qa_generator.md |
| `qa_distill.prompts.user_prompt_path` | user prompt 模板路径 | ../prompts/user/qa_generator.md |
| `qa_distill.output_data.output_dir` | 输出根目录 | ../output |
| `qa_distill.output_data.experiment_name` | 实验名 (输出子目录) | qa_distill_demo |
| `qa_distill.output_data.need_time_stamp` | 是否在实验名下再加时间戳子目录 | false |
| `feishu.url` / `feishu.keyword` | 飞书 Webhook (可选告警) | "" |
| `logging.file` | 日志落盘路径 (留空只打 stdout) | "" |

**路径约定**: 配置里的**相对路径** (prompt 文件 / output_dir) 一律按
**配置文件所在目录**解析, 与你在哪个目录执行命令无关 (找不到时回退 CWD)。

**默认配置查找**: `config_path` 为空时, 依次尝试环境变量 `QA_DISTILL_CONFIG` →
`<包根>/config/config.yaml` → `<包根>/config/config.example.yaml` (模板兜底)。
也就是说只配好 `OPENAI_API_KEY` / `OPENAI_BASE_URL`, 不建任何 config 文件也能跑。

**API key 优先级**: `qa_distill.model.api_key` > 环境变量 `OPENAI_API_KEY`
(已 export 的环境变量优先) > `.env` 里的 `OPENAI_API_KEY` (只填空, 不覆盖环境变量)。

### 本 run 推荐配置: 外部 teacher API (the teacher router, 强模型优先)

RSI-Master 的数据合成 / 蒸馏 **必须优先走强模型的外部 API, 不要本地起小模型 (32B) 当
teacher**。teacher 是 the teacher router 的 **distill-api** 端点 (managed by the hosting team), 暴露
**4 个模型** (`glm-5.2` / `kimi-k2.7-code` / `deepseek-v4-flash` / `deepseek-v4-pro`),
每个 `tpm=2_000_000`、`rpm=200`。本 run 直接把下面这段写进 `<qa_distill>/config/config.yaml`:

```yaml
qa_distill:
  model:
    provider: openai
    base_url: "https://your-teacher-router.example.com/v1"
    api_key: "sk-open-proxy-no-auth"   # proxy 未开鉴权, 任意值可; 占位值防 .env 回退
    model: "glm-5.2"
```

- `base_url` / `api_key` / `model` 三项一起填, 不要只靠 `.env` —— `.env` 的
  `OPENAI_BASE_URL` 指向另一个 sidecar (`your-sidecar.example.com:8000/v1`),
  不是这个端点。显式 config 里的值优先级最高, 会正确覆盖环境变量 / `.env`。
- **严格限制并发**: 每模型 `rpm=200`、`tpm=2M`, 设 `rate_limit` ≤ 150、
  `concurrency_limit` ≤ 32 (全局令牌桶, 取更紧者), 别撞 proxy 限流。
- **task 合成 && 轨迹蒸馏都可以; 轨迹蒸馏优先 `glm-5.2`** (推理轨迹质量最好),
  大批量 QA 改写可用便宜的 `deepseek-v4-flash`。轨迹蒸馏 `max_tokens` ≥ 16384。
- `model: "glm-5.2"` (轨迹蒸馏首选); 想跑大批量 QA 改写再换 `deepseek-v4-flash`。

---

## 输入: `data_pool` 格式

`data_pool` 是一个 list, 每个元素是一个 dict, 可选键:

| 键 | 说明 |
|---|---|
| `user_prompt_kwargs` | dict, 填充 **user** prompt 模板的 `{占位符}` |
| `system_prompt_kwargs` | dict, 填充 **system** prompt 模板的 `{占位符}` |
| `input` | 任意元数据 (来源 / id / 原始材料), 原样写进结果, 用于溯源 |

```python
data_pool = [
    {
        "input": {"source": "线性代数", "id": 1},
        "system_prompt_kwargs": {"language": "中文"},
        "user_prompt_kwargs": {"topic": "矩阵乘法的计算复杂度", "difficulty": "基础", "num": 2},
    },
    # ... 可以是几千上万条
]
```

> 提示: 大批量时把 data_pool 存成 JSONL 文件再读入即可 (见
> `examples/example_qa_distill.py::load_data_pool`)。

---

## Prompt 编写指南

两个模板都是普通文本文件, 用 Python `str.format()` 渲染 `{占位符}`:

```markdown
<!-- prompts/system/qa_generator.md -->
你是高质量问答数据蒸馏专家。生成 {language} 问答对。
=== 输出格式 (严格, 不要输出其它内容) ===
<question>
...
</question>

<answer>
...
</answer>
```

```markdown
<!-- prompts/user/qa_generator.md -->
主题: {topic}
难度: {difficulty}
请生成 {num} 条互不重复的问答数据。
```

要点:

1. **占位符与 `*_prompt_kwargs` 一一对应**; 渲染时缺 key 会打 warning 并保留原样。
2. **把输出格式写死在 prompt 里** (XML 标签 / JSON / 编号列表), 解析器才好解析。
3. 一个 prompt 对可以同时服务多种输入 —— 输入差异全部走 `{占位符}`, 模板不用改。
4. 想换任务 (比如改成"生成解题过程 / 生成选择题 / 打分") 就新写一份 prompt 对 +
   对应解析器, 同一套流水线照跑。

---

## 解析器 (输出解析) 编写指南

解析器就是 `fn(response: str) -> 结构化结果`, 返回值会作为 `extracted` 写进
result.jsonl。三种写法:

### 1) 内置默认提取器 —— 提一个标签

```python
extractor = pipeline.make_default_extractor("question")
# extract(response) → List[str]; 传 to_list=False 取第一个; 没匹配到返回 []/None
```

### 2) 自定义 —— 提多个标签 (推荐, 示例即此)

```python
def extract_qa(response: str):
    questions = pipeline.extract_all("question", response)
    answers = pipeline.extract_all("answer", response)
    if not questions or not answers:
        return None                      # 解析失败 → extracted=None, 便于事后过滤
    return [{"question": q, "answer": a} for q, a in zip(questions, answers)]
```

### 3) 自定义 —— 任意解析 (JSON / 正则 / 后处理)

解析器只是普通函数, 你可以 `json.loads`、正则、调用子模型做二次校验都行。
解析抛异常会被捕获并落 `extracted=None`, 不会中断整个批次。

> 解析失败条目的处理: 结果仍会落盘但 `extracted=None`; 大批量跑完后用
> `jq 'select(.extracted == null)' result.jsonl` 之类过滤出来重跑或人工处理。

---

## 输出: `result.jsonl` 格式

每行一条, 字段:

| 字段 | 说明 |
|---|---|
| `input` | 对应 data_pool 条目**整体** (含 `input` / `*_prompt_kwargs`) |
| `response` | 模型原始响应 |
| `naive_response` | 调用元数据 (耗时 / token 用量 / 重试次数 / finish_reason) |
| `extracted` | 解析器输出 (None 表示解析失败) |
| `timestamp` | 完成时间 ISO 格式 |
| `error` | (仅失败条目) 错误信息; 有该字段表示该条未成功生成 |

一条成功样例:

```json
{"input": {"input": {"source": "线性代数"}, "user_prompt_kwargs": {"topic": "..."}},
 "response": "<question>...</question>\n\n<answer>...</answer>",
 "naive_response": {"duration_ms": 812.4, "retry_count": 0, "usage": {"total_tokens": 312}, "model": "gpt-4o-mini", "finish_reason": "stop"},
 "extracted": [{"question": "矩阵乘法的计算复杂度是多少？", "answer": "..."}],
 "timestamp": "2026-08-16T21:40:01.123456"}
```

---

## 并发 / 限速 / 重试语义

- `concurrency_limit` 是**同时进行中的请求数**; `rate_limit` 是**全局**每分钟请求
  数, 由所有并发请求共享 (令牌桶)。两者取更紧的那个生效。
  → 经验: 不限速的 API 可以只调大 `concurrency_limit`; 会限速的 API 调小
  `rate_limit` (如 20~60), 让 `concurrency_limit` 略大于它即可。
- 重试: 超时 / 429 限速 / 5xx 服务端错误会自动重试, 指数退避
  (`max_retries` / `retry_delay` / `backoff_factor`, 见 `RetryConfig`)。
- 超时保护: 单次请求默认 3600s 超时 (`safe_chat_completion`), 挂死不会拖住整个批次。
- API 全挂时的行为: 该条结果落 `error` 字段, 批次继续; 跑完看错误统计即可。

---

## 断点续跑 (resume)

`pipeline.run(..., resume=True)` 会读取已有 `result.jsonl`, 跳过 `input` 相同的条目:

```bash
python examples/example_qa_distill.py --resume        # 上次中断后继续
```

适合大批量 + 不稳定 API 的场景: 断了重跑不重复花钱。

---

## 进阶用法

### API 预检 (APITester)

大批量前先确认 endpoint / 模型可用 (含流式/非流式、定时心跳、失败飞书告警):

```python
from qa_distill import APITester, StreamType

tester = APITester()                    # 自动从 config / .env / 环境变量读 key
results = tester.run_test(models=["gpt-4o-mini"], stream_type="all", enable_feishu_alert=True)
```

### 飞书告警 (FeishuBot)

```python
from qa_distill import send_feishu_message

send_feishu_message("qa_distill 蒸馏完成: 500/500 条")     # 需要 config 里 feishu.url
```

### 覆盖 config 而不用改文件

```python
pipeline = LLMPipeline(config_path="config.yaml",
                       model="gpt-4o",          # 覆盖模型
                       rate_limit=60,           # 覆盖限速
                       concurrency_limit=10,    # 覆盖并发 (注意: run() 里也可传)
                       )
```

### 用自定义 model 配置 (不走 config 文件)

```python
from qa_distill import LLMClient, ModelConfig
client = LLMClient(config=ModelConfig(model_name="qwen3", api_key="sk-...", base_url="...", rate_limit=50))
```

---

## 运行环境 (public/distill_env)

运行时只依赖 **5 个包**: `openai / pyyaml / requests / tqdm / python-dotenv`
(迁移时已去掉 `paramiko / matplotlib` 等无关依赖)。不再维护 requirements.txt,
已建好专用 uv 虚拟环境 (Python 3.12), 激活即可用:

```bash
# 每次新终端激活一次, 之后 python / pip 都指向该环境
source <repo>/public/distill_env/bin/activate
python examples/example_qa_distill.py --dry-run
```

也可以不 activate, 直接用该环境的解释器:

```bash
<repo>/public/distill_env/bin/python examples/example_qa_distill.py --dry-run
```

**环境重建** (环境被删 / 想升级依赖):

```bash
# uv 不在默认 PATH 上 (位于 $CONDA_ROOT/bin/uv), 先加进来
export PATH=$CONDA_ROOT/bin:$PATH
cd <repo>
uv venv --python 3.12 public/distill_env
uv pip install --python public/distill_env/bin/python openai pyyaml requests tqdm python-dotenv
```

> 只装了运行必需的最小集合; 不需要 `paramiko / matplotlib` (迁移时已去掉)。

---

## 测试

```bash
# 仓库根下 (需 .venv, 无网络):
.venv/bin/python -m pytest test/test_qa_distill.py -x -s
```

覆盖: 包导入 / config dot-notation 与 reload / `<标签>` 提取与默认提取器 /
全流水线 dry-run (monkeypatch 假响应, 不联网) / resume 续跑跳过已完成。

> distill_env 只装运行依赖, 不含 pytest; 跑测试用仓库根的 `.venv` (自带 pytest)。

---

## 迁移说明 (相对 thirdparty/agent-code-base)

保留的组件: `LLMClient`、`LLMPipeline`、`Config`、`get_logger`、`FeishuBot`、
`APITester`。去掉: `draw` (画图)、`paramiko`/`matplotlib` 等无关依赖。

迁移时做的修正 (与上游行为差异):

1. **全局限速**: 原实现每个并发任务 `new LLMClient()`, 各自独立的令牌桶,
   `rate_limit` 形同虚设; 现改为**所有并发 worker 共享一个 client**,
   `rate_limit` 全局生效。
2. **修复 None 解包崩溃**: `safe_chat_completion` 返回 `None` 时原实现会抛
   `cannot unpack non-iterable` 并掩盖真实错误; 现落一条带 `error` 字段的结果。
3. **路径解析**: config 里的相对路径 (prompt / output_dir) 按 **config 文件目录**
   解析, 不再依赖 CWD。
4. **prompt 文件缓存**: 按路径缓存模板原文, 大批量避免重复磁盘 IO。
5. **断点续跑**: 新增 `run(..., resume=True)`。
6. **env 兜底**: `api_key / base_url` 支持 `.env` / 环境变量
   `OPENAI_API_KEY / OPENAI_BASE_URL` (与仓库 `llm.openai` 约定一致)。
7. **配置节更名**: `llm_pipeline.*` → `qa_distill.*`。
8. 包内全部改为**相对导入**, 使包在 `qa_distill` 或 `backend.backend.qa_distill`
   两种名字下均可导入。

---

## FAQ

- **Q: 结果里为什么 `input` 套了一层?** A: 结果里 `input` 保存的是 data_pool
  条目整体 (`{input, user_prompt_kwargs, ...}`), 便于完整溯源; 读 `input.input`
  就是你的原始元数据。
- **Q: 一条数据要跑两个模型 (生成 + 校验)?** A: 分成两个 data_pool 各跑一次
  pipeline (各自配 prompt / 解析器 / 模型), 再用 `input.id` 关联。
- **Q: 输出想合并成 train.jsonl (LLaMA-Factory 格式)?** A: 解析器直接返回
  `{"conversations": [...]}` 或 `{"messages": [...]}`, 或跑完后用脚本把
  `extracted` 转成目标 schema。
- **Q: 不想存 `naive_response` (太占地方)?** A: 那个是元数据字段; 可以后续用脚本
  从 jsonl 里 `del naive_response` 再重写, 或跑完后按需处理。
