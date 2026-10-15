---
BACKEND_DESCRIPTIONS
Submit-train pipeline CLI (ExpOS port) — end-to-end 提交即训练: decontam → train →
torch_dist→HF convert → vLLM serve → eval_all scoring → result.json, 全部在
gpu_backend 上异步跑。Invoke: `pipeline-submit submit --data <raw> --train-config
<yaml> --mode llama_factory_sft|slime_sft --bench-id <id> --limit <n>`.
实验在 workspace `experiments/exp_####/`;回执里的 experiment_id 就是 FINISH 契约值。
---

# submit_pipeline — 提交即训练管线 (CLI backend)

`submit_pipeline` 是 v2 line 的**迭代数据管线 backend**:agent 只提交 **原始训练数据 +
训练配置**,管线端到端完成 去污 → 预检 → 训练 → (slime: torch_dist→HF 转换) →
vLLM serve → eval_all 评测,把 score 写进 `result.json`。不再有独立的"提交 ckpt 再单独
serve+eval"两个步骤 —— 评测归属管线内部,根治了 round-end serve 失败那一类故障。

> 移植自 ExpOS (`<repo>/tools/ExpOS/expos/backends/pipelines/`),改成
> **serverless CLI**:不跑 HTTP server,直接渲染 `pipeline.sh` 提交 gpu_backend 异步调度。

## Files

| File | Purpose |
|---|---|
| `cli.py` + `__main__.py` | **CLI** — `python -m backend.backend.submit_pipeline <submit\|status\|list\|health>`. |
| `driver.py` | `run_trial(request)` — 渲染 pipeline.sh → 提交 gpu_backend → 返回异步回执. |
| `render.py` | `build_pipeline_script(request, workspace)` → (pipeline.sh, key_paths). |
| `fragments.py` | `compose_llama_factory` / `compose_slime` — 组装 bash 片段为单个 Template. |
| `models.py` | `SubmitRequest` / `SubmitResponse` (pydantic; lora/qlora 拒绝校验). |
| `constants.py` | 路径常量 (`_LLAMA_FACTORY_ENV` 等跨仓库绝对路径) + `_BENCHMARK_EVAL`(只含 eval_all). |
| `config.py` + `config.yaml` | 运行期配置 (gpu_backend url / 轮询超时). |
| `decontaminate.py` | 去污逻辑 (从 `expos/service/decontamination.py` 移植). |
| `decontam_stage.py` | **stage-0 运行时入口** — GPU 节点 heavy python 去污 + 建数据集注册表. |
| `stages/` | 每 stage 一段 bash fragment (`preflight` / `train_llama_factory` / `train_slime` / `serve_eval`). |
| `configs/` | 示例训练配置 (LF full / slime single / multi gpu). |
| `README.md` | 本文档. |

## CLI contract

```bash
# 提交一次训练 (llama_factory): 渲染 + 提交 gpu_backend, 打印回执 (experiment_id = FINISH 值)
pipeline-submit submit \
    --data train.jsonl --train-config train.yaml \
    --mode llama_factory_sft --bench-id gpqa --limit 100

# slime 路径 (model_dir/torch_dist_dir 等走 slime 配置 YAML)
pipeline-submit submit \
    --data train.parquet --train-config slime.yaml \
    --mode slime_sft --bench-id gpqa

# 查实验状态 / 列表 / gpu_backend 健康
pipeline-submit status exp_0005
pipeline-submit list
pipeline-submit health
```

`submit` 输出 JSON 回执,含 `experiment_id` / `gpu_script_id` / `result_json_path` /
`checkpoint_dir` / `eval_dir`。实验 id 自动分配 (`exp_%04d`, 扫描 experiments-dir 取下一个)。

## Pipeline stages (pipeline.sh)

| Stage | llama_factory | slime |
|---|---|---|
| stage-0 | 数据预检 + **去污** (heavy python 读 raw data → clean jsonl + `dataset_info.json` 注册表, `DATASET_DIR=workspace/llama_factory_data`) | 数据预检 + **去污** (→ `artifacts/decontaminated_train.parquet`, `DATA_FILE` 指向它) |
| stage-1 | full-param SFT (`llama_factory_env`) | SFT (`launch_sft.sh` env 覆盖) |
| stage-1.5 | — | torch_dist→HF 转换 (`convert_torch_dist_to_hf.py`) |
| stage-2 | vLLM serve (rope 对齐守卫 + 就绪轮询) | 同左 |
| stage-3 | `eval_all --bench-id --trial-name <exp_id> --limit` → `metrics.json` | 同左 |

EXIT trap 无论成败都把终态写进 `result.json` (`status` / `eval_status` / `score` /
`failure_reason`);round-end 评测直接读这个文件,不再单独部署 FINISH ckpt。

## Experiment workspace layout

```
experiments/exp_0005/
├── work_order.json      # 提交时的原始参数
├── config.json          # 运行期状态 (status/score)
├── pipeline.sh          # 渲染产物 (GPU 节点执行)
├── pipeline.stage.log   # stage 日志
├── artifacts/           # decontamination_report.json, decontaminated_train.parquet ...
├── llama_factory_data/  # LF: decontaminated_train.jsonl + dataset_info.json
├── output/              # checkpoint / tokenizer (LF) 或 torch_dist 产物
├── hf_model_dir/        # slime stage-1.5 转换后的 HF 模型 (serve/eval 入口)
├── vllm.log / eval/     # serve 日志 / eval_all 输出 (metrics.json)
└── result.json          # EXIT trap 写终态; round-end 评测读这个
```

## 设计要点

- **去污在管线内** (stage-0, GPU 节点 heavy python): `.venv` 保持轻 (env 隔离铁律),
  渲染期不 staging 数据 —— 管线自洽, agent 交原始数据即可。
- **bench_id 是评测契约**: run 级 eval 参数 (bench_id/limit/serve 参数) 由 orchestrator
  写入 shared_context,agent 提交时透传,杜绝同 run 内评测参数漂移。
- **ESTALE 防护**: 执行前把 `scripts/` 拷到 workspace 再跑 (ExpOS 模式)。
- **回退**: gpu_backend 不可达 → 提交失败且 `result.json` 保持占位 (`status=pending`),
  round-end 可识别为 "实验存在、在跑/未出分"。
