---
BACKEND_DESCRIPTIONS
Unified search backend: HuggingFace Hub (search/inspect/configs/splits/readme/
sample/download/materialize) + web (Serper) + GitHub + scholar (arXiv/GScholar/DBLP).
Invoke: `python search_cli/main.py <verb> ...` — the agent hf_*/search_* tools are
thin wrappers over these ops. Client proxies to the server on localhost:8991 when
reachable, else runs locally. Discovery verbs are cheap; download/materialize are heavy.
Search first: workers/main should discover & vet public sources before building data.
---

# search_cli — Unified Search Backend

`search_cli` 是 agent 数据发现工具 (`hf_*` / `search_*_...`) 底下的统一搜索后端:
一个入口 (`python search_cli/main.py <verb> ...`) 覆盖四类外部数据源 ——
HuggingFace Hub、web (Serper)、GitHub、scholar (arXiv / Google Scholar / DBLP)。

> 该模块由 `rsi_master.public_seeds.backend_modules.search_cli` 开关, seed 到
> `public/backend/search_cli/`。config 里 `search_tools.enabled: true` 时 agent
> 直接调 `hf_*` / `search_*_...` 工具即可, 不必手写本 CLI —— 工具的底层就是这里的 op。

---

## 为什么多用搜索 (worker / main-agent 必读)

拿到 objective 后 **第一步是搜, 不是造**。好数据源大概率已经在 HF Hub / GitHub /
web 上公开了 —— 用本工具族花几个**廉价**调用就能发现 + 验证; 凭记忆手搓数据、或从
repo 抄一段来"补能力", 反而烧 GPU、还可能引入脏数据。规则:

- **在写任何训练行之前, 先问一句「哪个公开源已经覆盖这个能力?」然后去搜。**
- 多换关键词搜几轮, 别用一次搜索就下结论。
- **搜索工具没有 leakage 豁免** —— 采纳前仍要核对与目标 eval 的重叠。

## Server / client 拓扑

| 组件 | 说明 |
|---|---|
| `ops/` | 实际搜索逻辑 (纯函数, 每个 op 可独立调用) |
| `cli.py` | CLI 入口, 每个 verb → 一个 op |
| `server.py` | FastAPI 壳 (`POST /search` 收 `{op, params}`), 默认 `:8991` |
| `client.py` | 客户端: 探测 server, 可达 → 代理, 不可达 → 本地执行 |
| `config.yaml` | host/port/timeouts/默认参数 (env 可覆盖) |

client 自动处理可达性 —— 你不用关心 server 在不在; 同一 op 从 bash 或 HTTP 走
完全一致。`health` 可随时探测。

## CLI verbs

**HuggingFace 数据集** (`hf_*` 工具对应):

| verb | 用途 | 对应工具 |
|---|---|---|
| `search <query> [--limit] [--author]` | 按关键词搜数据集 | `hf_search_datasets` |
| `inspect <dataset_id> [--config]` | 完整元数据 | `hf_inspect_dataset` |
| `configs <dataset_id>` | 列 configs | `hf_dataset_configs` |
| `splits <dataset_id> [--config]` | 列 splits | `hf_dataset_splits` |
| `readme <dataset_id>` | README/描述 | `hf_dataset_readme` |
| `sample <dataset_id> [--config] [--split] [--num-samples]` | 看真实行 | `hf_dataset_sample` |
| `download <dataset_id> [--output-dir]` | 下载原始文件 (慢) | `hf_download_dataset` |
| `materialize <dataset_id> [--config] [--split] [--max-rows] [--output-dir]` | 产出缓存的 train.jsonl | `hf_materialize_dataset` |

**Web / GitHub / scholar**:

- `web-search <query>` / `web-parse <url>` —— Serper 通用 web 搜索 + 抓页面。
- `github-repos` / `github-code` / `github-issues` / `github-prs` / `github-users` / `github-readme <owner> <repo>` —— GitHub API。
- `scholar-arxiv <query> [--by content|author]` / `scholar-google <query>` / `scholar-dblp <query> [--kind papers|authors|venues]` —— 论文 / scholar / DBLP。

**运维**:

```bash
python search_cli/main.py health                          # 探测 server
python search_cli/main.py server start --port 8991        # 起 FastAPI server
```

## 详细操作指南

### 标准采纳流程 (每次都用)

1. **发现候选** —— `search <query>` (HF) 起步; objective 需要 web / GitHub / 论文
   证据时用 `web-search` / `github-*` / `scholar-*`。**多换关键词搜几轮**。
2. **采纳前审查** —— `inspect` + `configs` + `splits` + `sample --num-samples N`:
   查格式、split 规模、行形状。**只看名字就采纳是最贵的错误** —— 先看真实行。
3. **核可用性** —— 候选是否已缓存 (`~/data/`, HF cache)? 物化会不会超时? 先确认再承诺。
4. **物化** —— `materialize <dataset_id> --split train --max-rows <N>` 产出缓存的
   JSONL (会自动缓存, 重复调用复用)。
5. **注册 + 提交** —— 抽查后交给 `worker_submit` (或先 `worker_add_data` 进 pool,
   带 `descriptions=`)。

### HF Hub 示例 (hf_* 工具 ↔ CLI verb 1:1)

```bash
# 1) 发现: 关键词搜数据集
python search_cli/main.py search "math word problems" --limit 20

# 2) 审查: 元数据 + 真实行
python search_cli/main.py inspect openai/gsm8k
python search_cli/main.py sample openai/gsm8k --config main --split train --num-samples 5

# 3) 物化: 产出缓存的 train.jsonl
python search_cli/main.py materialize openai/gsm8k --config main --split train --max-rows 2048
```

### web / GitHub / scholar 示例

```bash
python search_cli/main.py web-search "LLM post-training best practices"   # 通用 web
python search_cli/main.py web-parse https://example.com/blog               # 抓页面正文
python search_cli/main.py github-repos "llm post-training" --language python
python search_cli/main.py github-code "def evaluate_gsm8k" --repo owner/repo
python search_cli/main.py github-readme owner repo
python search_cli/main.py scholar-arxiv "RLHF" --by content --max-results 5
python search_cli/main.py scholar-arxiv "Yoshua Bengio" --by author
python search_cli/main.py scholar-google "direct preference optimization"
python search_cli/main.py scholar-dblp "post-training" --kind papers
```

### 成本与性能

- **发现类便宜**: `search` / `inspect` / `sample` / `web-*` / `github-*` / `scholar-*`
  —— 随便用。每次调用 stdout 是单个 JSON 对象 (`{"ok": true, "result": {...}}`)。
- **重动词要省**: `download` / `materialize` 慢且有缓存 —— 只调一次, 复用缓存;
  迭代时优先用便宜的发现类动词。`materialize` 有 `--max-rows` 上限, 别整库拉。
- **默认参数**: `search` 默认 `max_results=100`; `sample` 默认 `num_samples=5`
  (上限 100); `materialize` 默认 `max_rows=2048`。

### 故障排查

- **`error` 载荷**: `{"ok": false, "error": "..."}` —— 读消息调整, 别闷头重试。
- **server 不可达**: `health` 返回 `unreachable`, client 自动落到本地执行 —— 结果一致,
  只是更慢。想在常驻机器上提性能就 `server start`。
- **搜索超时**: 多为 `download` / `materialize` 大源。先 `inspect` 看规模再定
  `--max-rows` 与 split; 必要时换更小的候选源。
- **API key 缺失**: web (Serper) 要 `SERPER_API_KEY`, GitHub 要 `GITHUB_TOKEN`,
  HF 要 `HF_TOKEN` —— 都在 `.env`, 由 `evomaster/env_loader.py` 注入进程环境。

## Leakage 边界

搜索工具不是 leakage 豁免: 对搜到的一切, 采纳前先核对与目标 eval 的重叠。发现污染 →
丢弃该源, 并在 `GLOBAL_MEMORY.md → Warnings` 记一笔。
