# MedCaseReasoning（DataPrep-Bench 协议）

固定 897 条官方 test case。候选模型生成 `<think>` 临床推理和 `<answer>` 诊断；独立 judge 判断诊断等价性，并逐条判断 clinician reasoning point 的覆盖情况。`reward` 是诊断准确率，推理召回写入 `meta_data.reasoning_recall`。

运行 `python format/medcasereasoning/bootstrap.py` 后执行 dry-run。正式评测必须通过环境变量提供独立的 `JUDGE_OPENAI_API_BASE`、`JUDGE_OPENAI_API_TOKEN`、`JUDGE_MODEL`。

生成与评分分别控制并发：`custom.max_workers` 默认为 8，`custom.judge_workers` 默认为 16（必须为正数，设为 1 可串行）。评分按病例并行，每个病例内部按原顺序判断诊断和推理要点，因此同时在途的裁判请求不超过 judge_workers。旧配置省略 judge_workers 时也使用 16。

裁判通过显式 endpoint/model 调用，不修改候选模型的全局环境变量。诊断准确率、推理召回、截断计错和裁判失败判 incomplete 的规则保持不变。日志及 reward.meta_data 新增 generation_elapsed_s、scoring_elapsed_s 和实际 judge_workers，便于分别检查生成与评分耗时。默认并发值尚需结合裁判服务限流和实际吞吐调优。
