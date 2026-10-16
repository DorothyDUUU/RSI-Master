# LiveCodeBench Code Generation Lite v6

## Benchmark

This integration pins the official LiveCodeBench repository and the `code_generation_lite` `release_v6` dataset. The dataset contains 1,055 Python programming problems published from May 2023 through April 2025 on LeetCode, AtCoder, and Codeforces. `recent_2025` contains the 182 problems published from 2025-01-01 through 2025-04-30; `full_v6` contains all 1,055.

## Capability and metric

The benchmark measures whether a model can synthesize a complete Python solution from a contest problem statement. The scalar reward is official execution-based `pass@1` in the range 0 to 1. A solution passes only when it succeeds on all public and private tests. This does not use an LLM judge.

## Reference

Use the official LiveCodeBench leaderboard for model comparisons. Scores are comparable only when the release, date window, sampling count, temperature, prompt style, and execution timeout match. This integration defaults to one deterministic sample (`n=1`, temperature 0) for economical model comparison; official upstream CLI defaults to `n=10`, temperature 0.2 and can therefore differ.

## Reproduction

The runner uses the official generic chat system message and question template, requests Python code in fenced backticks, applies the official last-fenced-block extractor, then invokes the official APPS-derived execution checker. Defaults: 16,384 output tokens, 16 concurrent generation requests, 16 execution workers, and 6 seconds per test. Qwen thinking is enabled by default and is configurable via `custom.enable_thinking`.

```bash
framework/.venv/bin/python framework/run_eval.py livecodebench \
  --split recent_2025 --model /path/to/model \
  --openai-api-base http://127.0.0.1:8000/v1 --openai-api-token sk-local
```

## Integrity

Official source commit: `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`. Dataset: `livecodebench/code_generation_lite`, revision `0fe84c3912ea0c4d4a78037083943e8f0c4dd505`, `release_v6`, 1,055 rows, local fingerprint `3bf27f1e2df313d6`. The runner refuses a different official checkout commit. Benchmark quality is 100/100 because scoring uses hidden executable tests and the upstream evaluator rather than subjective judging.
