# LegalBench（DataPrep-Bench 协议）

本接入复现截图所用 DataPrep-Bench 法律评测子集，而非声称覆盖 LegalBench 全部 162 个任务。固定数据包含 413 条，来自 `abercrombie`、`contract_qa`、`hearsay`、`personal_jurisdiction`、`ucc_v_common_law` 五个任务。

先运行 `python format/legalbench/bootstrap.py`，再运行 `framework/.venv/bin/python framework/run_eval.py legalbench --split all --dry-run --validate-data`。正式评测需要候选模型端点，以及独立的 `JUDGE_OPENAI_API_BASE`、`JUDGE_OPENAI_API_TOKEN`、`JUDGE_MODEL`；凭据只从环境变量读取。

主分数为官方五任务 balanced accuracy 按样本数加权的结果。候选模型达到固定生成预算（`finish_reason=length/max_tokens`）时按错误计入分母，并在 `meta_data.truncated_as_incorrect` 中单独报告；judge、传输或服务故障仍属于评测基础设施失败，会令最终 `reward` 为 `null`。逐任务指标、生成完成数、计分数和基础设施失败数均写入 `meta_data`。
