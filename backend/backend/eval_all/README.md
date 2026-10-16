---
BACKEND_DESCRIPTIONS
Unified benchmark suite evaluation backend — delegates scoring to benchmark suite
`format/<bench_id>/`; target any bench via evaluation.eval_all.bench_id.
Invoke: `python eval_all.py --model <serve_dir> --base-url <vllm_url> --bench-id <id> --run-dir <out>`.
Output `metrics.json` (accuracy = experiment score); partial-tolerant — a numeric
metric over completed samples still exits 0.
---

# eval_all — Unified benchmark suite Evaluation Backend

`eval_all` 是 v2 line 的**统一评测 backend**:它不实现任何评分逻辑,而是把评测委托给
统一评测框架 benchmark suite (`$BENCHMARK_ROOT`),
通过一个 **`bench_id`** 评测 `format/<bench_id>/` 下任意一个 bench
(约 249 个,含 aime2025 / gsm8k / gpqa / arc-challenge / ... 等)。

> 相比 `eval_gsm8k` / `eval_aime2025` 等分立模块:评测一个新 bench **不需要**再写一个
> `eval_<bench>/` 模块,只需要在 config 里注入对应的 `bench_id`。

---

## Files

| File | Purpose |
|---|---|
| `eval_all.py` | **Core** — 包装 `framework/run_eval.py`;纯函数 (`resolve_bmk_root` / `build_override_yaml` / `build_run_eval_command` / `map_reward_to_metrics`) + `main()`. |
| `eval.sh` | **Manual launcher** — 起一个 bench 的评测,自动建时间戳输出目录. |
| `eval.yaml` | **Metadata** — 记录委托方式 / 评分来源 / 部分完成策略. Not read at runtime. |

## CLI contract (对齐 pipeline stage-3)

```bash
python eval_all.py \
    --model <serve_dir> --base-url <vllm_url> --api-key sk-123 \
    --max-tokens N --max-workers N --timeout 600 --limit N \
    --run-dir <eval_dir> \
    --bench-id <bmk_bench_id> [--trial-name X] [--split S] [--bmk-root <root>]
```

- `--bench-id` 必填;合法字符同 `run_eval.py` 的 `SAFE_NAME`
  (`^[A-Za-z0-9][A-Za-z0-9_.-]*$`)。校验 `format/<id>/run_benchmark.py` 存在,尽早失败。
- `--bmk-root` 默认 `$BENCHMARK_ROOT`,可用
  `BENCHMARK_ROOT` 环境变量覆盖。
- `--max-tokens` / `--max-workers` / `--timeout` 传 **0** = 用 benchmark suite 该 bench
  自身 `config/config.yaml` 里的值(只把正值写进 `--config-path` 的 override)。
- `--limit` 必须是正数;`<=0` 回退到 **500**。**绝不传 `-1`**——generic runner 的
  `_load_rows` 会把 `len(rows) >= -1` 当成终止条件,导致一个样本都不跑。

## Output

写入 `--run-dir/`:

- **`metrics.json`** — pipeline 从这里提取 `accuracy` 作为实验 score。
- `reward.json` / `samples.jsonl` / `config.yaml` / `bmk_override.yaml` — benchmark suite
  产出与 override 配置的拷贝,agent 可直接检查。
- `run_eval.log` — 子进程 stdout/stderr。

`metrics.json` 结构:

```json
{
  "benchmark": "aime2025",
  "trial_name": "exp_0001",
  "metric_key": "accuracy",
  "accuracy": 0.42,
  "raw_reward": 0.42,
  "attempted": 500, "completed": 498, "incomplete": 2, "correct": 209,
  "scoring_supported": true,
  "status": "partial",
  "output_dir": ".../data/output/aime2025/exp_0001",
  "log": ".../run_eval.log",
  "limit": 500
}
```

`status` 取值:`completed`(全样本完成)/ `partial`(有数值指标但存在 incomplete)/
`failed`(无数值指标,如 unscored bench 或全部超时)。exit code:数值指标在手 → 0,
否则 → 1。

## Partial-tolerant 语义

benchmark suite 对任何 incomplete 样本都会**退出非 0** 且把 `reward` 置 `None`
(legacy gsm8k runner 与 generic runner 都如此)。`eval_all` 不把「部分超时」当实验
失败:只要 `meta_data` 里有可观测指标(legacy `accuracy` 或 generic `observed_reward`,
即 completed 样本上的正确率),就写 `status="partial"` 并 **exit 0**。只有完全拿不到
数值指标(unscoored bench / 全部 incomplete / 传输全挂)才失败。这与
`eval_gsm8k/gsm8k_eval.py` 对部分完成的宽容一致,避免一次 vLLM 超时毁掉整个实验。

## When to use which

- **pipeline stage-3 (默认)**:config 里 `evaluation.backend: "eval_all"` 且
  `evaluation.eval_all.bench_id` 有值时,`pipeline_llama_factory_gsm8k.py` 自动把
  stage-3 派发到 `eval_all.py`。
- **手动**:`bash eval.sh <bench_id> --limit 100`。

## Setup and portable paths

See [BmkEverything setup](../../../bmk_everything/SETUP.md). The project supports 24 benchmark integrations; datasets and external evaluator environments are prepared separately. Set `BENCHMARK_ROOT` to a complete tree on the execution host, or pass `--bmk-root`; keep ExpOS `eval_all.bmk_root` consistent. A healthy GPU backend does not verify benchmark assets.
