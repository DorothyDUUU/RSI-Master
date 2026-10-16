# CKQA / CPA-KQA

截图列名 `CKQA` 在当前 DataPrep-Bench 文档中对应 `c`，统一数据里的名字是 `FinCDM-CPA-KQA`。本接入保留 `ckqa` ID 和三个别名，固定 311 条四选一题，使用官方 accuracy。

运行 `python format/ckqa/bootstrap.py`，再执行 `framework/.venv/bin/python framework/run_eval.py ckqa --split all --dry-run --validate-data`。评分是确定性的 A–D 解析，不需要 judge。
