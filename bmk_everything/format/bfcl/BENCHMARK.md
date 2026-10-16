# BFCL v3 Full

> **Benchmark ID**: `bfcl` · **Domain**: `tool-use` · **Eval method**: `bfcl_call_sequence_match`

## 1. What this benchmark is

BFCL v3 Full is the public Berkeley Function Calling Leaderboard corpus merged into one local split. It covers single-turn function calling, executable tool calls, live relevance/irrelevance detection, multi-turn function calling, and a small REST-only auxiliary slice. The local `default` split contains 5,251 samples from the official HF release; 5,181 are scoreable and the 70 REST rows are kept for inspection but excluded from the scalar reward because the public release does not ship a fixed answer key for that slice.

The data is stored in `data/datasets/bfcl/default.jsonl`. It is a merged copy of the official BFCL v3 files, including the AST-style subsets, the executable subsets, the multi-turn subsets, and the relevance/no-call subsets.

In one sentence: BFCL measures whether a model can choose the right tool, fill the right arguments, stay silent when no tool is needed, and keep doing that across multiple turns.

## 2. Capabilities measured

This benchmark measures structured function calling rather than free-form QA.

It checks:

- exact tool selection;
- argument filling and value formatting;
- parallel tool-call sets;
- no-call / irrelevance detection;
- relevance detection when at least one tool is usable;
- turn-by-turn tool-call sequences in multi-turn tasks.

It does not measure open-ended reasoning quality directly. A pretty explanation with the wrong tool call still scores wrong. For `live_relevance`, any valid tool call to one of the provided tools counts; for `irrelevance`, `live_irrelevance`, and `chatable`, the correct behavior is to emit no tool call.

## 3. Reference metrics

The local full-split calibration results below use the required MetaMaster models and the deterministic BFCL scorer.

| Model | split | Score | Conditions | Source |
|---|---:|---:|---|---|
| deepseek-v4-flash | default | 0.6163 | Full 5,251 rows, 5,181 scoreable, no truncation | Local calibration, 2026-08-29 |
| Qwen3-8B | default | 0.6595 | Full 5,251 rows; adaptive maximum-compatible output budget; 158 final truncations scored zero | Local calibration, 2026-08-30 |
| Qwen3.5-35B-A3B | default | 0.6709 | Full 5,251 rows; 7 repeated max-budget timeouts scored zero | Local calibration, 2026-08-29 |

## 4. How the model is evaluated

Run from the BmkEverything repository root:

```bash
framework/.venv/bin/python framework/run_eval.py bfcl
```

The runner sends OpenAI-compatible chat completions with `tools` whenever tools exist. Function names containing `.` are sanitized to `_` in the request, then mapped back during scoring.

Scoring rules:

- `simple`, `multiple`, `parallel`, `parallel_multiple`, `live_simple`, `live_multiple`, `live_parallel`, `live_parallel_multiple`, `sql`, `java`, and `javascript` use released BFCL answer sets and structural tool-call matching.
- Parallel subsets ignore tool-call order within the response.
- `exec_*` subsets compare against the released ground-truth call strings after parsing and canonicalizing positional arguments.
- `multi_turn_*` subsets are evaluated turn by turn against the released call sequences; order matters within a turn.
- `irrelevance`, `live_irrelevance`, and `chatable` require no tool call.
- `live_relevance` requires at least one valid tool call.
- `rest` is tracked in samples but excluded from `reward`.
- A token/context-truncated scoreable response receives zero credit even if a partial tool call can be parsed; truncations remain in the denominator and are counted in `meta_data.truncated_total`.

Default runtime settings:

- `temperature = 0.0`
- `max_tokens = 65536`
- `timeout = 600`
- `max_workers = 6`

For models with a combined input/output context limit, `custom.fallback_max_tokens` may list descending output budgets. The runner only moves to the next budget when the provider explicitly reports that input plus requested output exceeds the model context window, and uses the highest accepted budget. Other HTTP or transport failures do not trigger this fallback. Each sample or multi-turn request records `requested_max_tokens`; a response that still ends with `finish_reason=length` at the highest accepted budget receives zero credit.

The runner writes `samples.jsonl` with the prompt, parsed tool calls, expected target, actual requested output budget, truncation flag, and per-sample correctness.

## 5. Config alignment

`config.json` declares one split: `default`.

`default` maps to the merged BFCL v3 corpus:

- total rows: 5,251
- scoreable rows: 5,181
- unscored rows: 70 (`rest`)

The official local mirror contains two distinct rows with the same ID, `live_relevance_3-3-0`. Resume and merge operations must preserve dataset order and validate both row position and ID; using an ID-only dictionary would drop one row.

`judge_quality_score` is 100 because this is the official BFCL corpus and scoring is deterministic.
