# AIME 2026

> **Benchmark ID**: `aime2026` · **Domain**: `math_reasoning` · **Scorer**: `integer exact-match`

## 1. What this benchmark is

AIME (American Invitational Mathematics Examination) is the American Invitational Mathematics Examination, a high-difficulty contest that sits between the AMC and the USA Mathematical Olympiad. This AIME 2026 set is packaged by [MathArena](https://matharena.ai) — the 30 competition questions "extracted, converted to LaTeX and verified" for the MathArena Leaderboard, sourced from [MathArena/aime_2026](https://huggingface.co/datasets/MathArena/aime_2026) (license `cc-by-nc-sa-4.0`, non-commercial). It contains **30 contest-level math problems**, each with an integer answer in the range 0–999.

The problems span the competitive-math branches — number theory, combinatorics, algebra, and geometry — and are substantially harder than word-problem sets such as GSM8K. Solving them requires strong multi-step mathematical reasoning rather than simple arithmetic.

- Domain: competitive mathematics
- Size: 30 problems (local `test.jsonl`, drawn from `MathArena/aime_2026`)
- Answer form: integer (0–999)
- Scoring convention: in the original contest each problem is worth 1 point for a total of 15 per exam; here the reported metric is plain accuracy (proportion correct)

Real sample (first row of the local data):

> Patrick started walking at a constant rate along a straight road from school to the park. One hour after Patrick left, Tanya started running along the same road from school to the park. One hour after Tanya left, Jose started bicycling along the same road from school to the park. Tanya ran at a constant rate of $2$ miles per hour faster than Patrick walked, Jose bicycled at a constant rate of $7$ miles per hour faster than Tanya ran, and all three arrived at the park at the same time. The distance from the school to the park is $\frac{m}{n}$ miles, where $m$ and $n$ are relatively prime positive integers. Find $m + n$.

The correct answer is `277`. In one sentence: the model must produce the exact integer answer to a multi-step olympiad-style problem and emit it in a fixed `ANSWER: <number>` format.

## 2. Capabilities measured

- **Competitive-level mathematical reasoning**: each problem needs several steps of insight — rate-and-distance algebra, geometry, combinatorics, or number-theoretic arguments — far beyond a single known operation. There is no step-by-step scaffolding; the model must decide the approach itself.
- **Numeric exactness**: the target is a specific integer in 0–999. There are no choices to fall back on and no approximate credit — a near miss (e.g. `278` for `277`) is simply wrong.
- **Output-format coupling**: the prompt forces the response to end with a line `ANSWER: $ANSWER`. The extraction logic is strict but tolerant of position: the last `ANSWER: <number>`, else the last `\boxed{<number>}`, else the last standalone number in the text is taken. A correct computation buried in prose without a parseable `ANSWER:` line can still be recovered by the fallback, but a response with no standalone number at all fails.
- **Distractor structure**: unlike multiple-choice, there are no planted decoys; the challenge is purely that a single arithmetic slip anywhere in the chain produces a wrong final integer.

**What this does NOT measure (boundary)**: no world knowledge, no retrieval, no code execution, no open-ended generation, and no follow-up interaction. The set is small (30 items), and individual-problem variance is high, so fast smoke tests should pin a fixed split rather than subsample at random.

**Scoring strictness / failure modes**: a sample fails when (a) no number is extractable from the response, (b) the extracted value differs from the target integer, or (c) the value is a non-integer that does not numerically equal the target. The comparison is `float(predicted) == float(target)` after string cleanup (commas / `$` / `%` stripped), so `277`, `277.0`, and `$277$` all hit `277`, while `278` fails.

## 3. Reference metrics

Public reference on the **MathArena Leaderboard** (`matharena.ai`, accessed 2026-08-19), which ranks models on exactly this 30-problem set:

| 模型 | 分数 | 条件 | 来源 |
|---|---|---|---|
| Inkling | ~97.1 | AIME 2026, MathArena leaderboard | matharena.ai |
| A.X-K2 | ~97.1 | AIME 2026, MathArena leaderboard | matharena.ai |

These are the leaderboard's top entries; the exact evaluation protocol (sampling, temperature, answer extraction) is MathArena's own and may differ from the runner here, so treat them as an upper-bound sanity reference rather than an apples-to-apples baseline. No human baseline is fabricated; the evaluation here is a clean integer-match accuracy.

## 4. How the model is evaluated

### Run command

From the repository root, with model credentials exported (see `framework/README.md`):

```bash
framework/.venv/bin/python framework/run_eval.py aime2026 \
  --limit 30 --trial-name my-trial
```

- `--limit N`: caps the number of problems evaluated (the full set is 30, so `--limit 30` runs everything; pass a smaller value for a quick smoke test).
- `--trial-name`: results are written to `data/output/aime2026/<trial>/` (`reward.json` plus per-sample `samples.jsonl`).
- Dry-run first to confirm resolution without calling the model: `... aime2026 --dry-run --validate-data` prints the resolved configuration and exits.

This benchmark is a hand-written, self-contained runner (`format/aime2026/run_benchmark.py`) invoked through the unified entrypoint; it does not use a `spec.json` or the generic bigbench/HF adapters.

### Input

The prompt is built from `AIME_PROMPT_TEMPLATE`, a "reason step by step, then close with `ANSWER: $ANSWER`" template with no few-shot examples:

```
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{problem}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \boxed command.
```

### Output and scoring

The model should end with `ANSWER: <integer>`. `extract_answer()` pulls the value by priority: (1) the last `ANSWER: <number>` match, (2) the last `\boxed{<number>}`, (3) the last standalone number found scanning the response from the bottom. Commas, `$`, and `%` are stripped. Scoring then applies `float(predicted) == float(target)` — numeric exact-match on the target integer. Responses whose API call completed (`complete=true`) determine the denominator; `correct` over `completed` samples is the accuracy reported in `reward.json`.

### Runtime conditions

From `config/config.yaml` `custom`: `temperature=0.0` (greedy decoding), `max_tokens=16000` (generous budget for long chain-of-thought), `timeout=600s` per request, `max_workers=10` concurrent requests. The `default` split runs all 30 problems; `sample_10` and `sample_100` are the quick and capped variants.

### Edge cases

A response with no extractable number is counted wrong. A truncated response that cuts off before any number is also wrong. Transport/request failures are recorded as incomplete (`complete=false`) and excluded from the accuracy denominator — they are not treated as model errors.

## 5. Config alignment

- **split** (from `config.json`): `default` (all 30 problems), `sample_10` (first 10, for a fast smoke test), and `sample_100` (capped at 100; the local set has only 30, so this is equivalent to the full set). The local data is `data/datasets/aime2026/test.jsonl`, **30 rows**, with `id` carrying the source `problem_idx` (1–30). A copy lives at `format/aime2026/test.jsonl` for self-containment.
- **eval-method**: the benchmark is scored by integer exact-match (`float(predicted) == float(target)`) computed inside the runner — deterministic, no LLM judge.
- **judge_quality_score**: `100` — a curation-time self-assessment of data quality (max 100), present in `config.json`. The evaluation itself never consults an LLM judge.
- **license**: `cc-by-nc-sa-4.0` (non-commercial), per the source dataset `MathArena/aime_2026`.
