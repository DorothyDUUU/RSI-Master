# AIME 2025

> **Benchmark ID**: `aime2025` · **Domain**: `math_reasoning` · **Scorer**: `integer exact-match`

## 1. What this benchmark is

AIME (American Invitational Mathematics Examination) is the American Invitational Mathematics Examination, a high-difficulty contest that sits between the AMC and the USA Mathematical Olympiad. The AIME 2025 validation set packaged here comes from the [AI-MO/aimo-validation-aime](https://huggingface.co/datasets/AI-MO/aimo-validation-aime) repository and contains **90 contest-level math problems**, each with an integer answer in the range 0–999.

The problems span the competitive-math branches — number theory, combinatorics, algebra, and geometry — and are substantially harder than word-problem sets such as GSM8K. Solving them requires strong multi-step mathematical reasoning rather than simple arithmetic.

- Domain: competitive mathematics
- Size: 90 problems (local `test.jsonl`, drawn from the aimo-validation-aime validation set)
- Answer form: integer (0–999)
- Scoring convention: in the original contest each problem is worth 1 point for a total of 15 per exam; here the reported metric is plain accuracy (proportion correct)

Real sample (first row of the local data):

> Quadratic polynomials $P(x)$ and $Q(x)$ have leading coefficients $2$ and $-2,$ respectively. The graphs of both polynomials pass through the two points $(16,54)$ and $(20,53).$ Find $P(0) + Q(0).$

The correct answer is `116`. In one sentence: the model must produce the exact integer answer to a multi-step olympiad-style problem and emit it in a fixed `ANSWER: <number>` format.

## 2. Capabilities measured

- **Competitive-level mathematical reasoning**: each problem needs several steps of insight — interpolation of quadratics, sphere-tangency geometry, set-sum combinatorics, or number-theoretic arguments — far beyond a single known operation. There is no step-by-step scaffolding; the model must decide the approach itself.
- **Numeric exactness**: the target is a specific integer in 0–999. There are no choices to fall back on and no approximate credit — a near miss (e.g. `117` for `116`) is simply wrong.
- **Output-format coupling**: the prompt forces the response to end with a line `ANSWER: $ANSWER`. The extraction logic is strict but tolerant of position: the last `ANSWER: <number>`, else the last `\boxed{<number>}`, else the last standalone number in the text is taken. A correct computation buried in prose without a parseable `ANSWER:` line can still be recovered by the fallback, but a response with no standalone number at all fails.
- **Distractor structure**: unlike multiple-choice, there are no planted decoys; the challenge is purely that a single arithmetic slip anywhere in the chain produces a wrong final integer.

**What this does NOT measure (boundary)**: no world knowledge, no retrieval, no code execution, no open-ended generation, and no follow-up interaction. The set is moderate in size (90 items), and individual-problem variance is high, so fast smoke tests should pin a fixed split rather than subsample at random.

**Scoring strictness / failure modes**: a sample fails when (a) no number is extractable from the response, (b) the extracted value differs from the target integer, or (c) the decimal is rounded to the wrong integer. Note the comparison deliberately rounds a decimal prediction with `int(round(float(predicted)))` before matching, so `116.4` and `116` both hit `116`, while `116.6` would round to `117` and fail.

## 3. Reference metrics

No stable public reference score is available for this AIME 2025 validation subset (the problems are recent and model gaps are large), so:

> No public reference score yet; to be filled after first evaluation.

No human baseline is fabricated; the evaluation here is a clean integer-match accuracy, not a contest-perfect comparison.

## 4. How the model is evaluated

### Run command

From the repository root, with model credentials exported (see `framework/README.md`):

```bash
framework/.venv/bin/python framework/run_eval.py aime2025 \
  --limit 500 --trial-name my-trial
```

- `--limit N`: caps the number of problems evaluated (the full set is 90, so `--limit 500` runs everything; pass a smaller value for a quick smoke test).
- `--trial-name`: results are written to `data/output/aime2025/<trial>/` (`reward.json` plus per-sample `samples.jsonl`).
- Dry-run first to confirm resolution without calling the model: `... aime2025 --dry-run --validate-data` prints the resolved configuration and exits.

This benchmark is a hand-written, self-contained runner (`format/aime2025/run_benchmark.py`) invoked through the unified entrypoint; it does not use a `spec.json` or the generic bigbench/HF adapters.

### Input

The prompt is built from `AIME_PROMPT_TEMPLATE`, the same "reason step by step, then close with `ANSWER: $ANSWER`" template used for GSM8K, with no few-shot examples:

```
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{problem}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \boxed command.

Reasoning:
```

### Output and scoring

The model should end with `ANSWER: <integer>`. `extract_answer()` pulls the value by priority: (1) the last `ANSWER: <number>` match, (2) the last `\boxed{<number>}`, (3) the last standalone number found scanning the response from the bottom. Commas, `$`, and `%` are stripped. Scoring then applies `int(round(float(predicted))) == int(target)` — an integer exact match after rounding. The response that is `complete` (a successful API call) determines the denominator; `correct` count over `completed` samples is the accuracy reported in `reward.json`.

### Runtime conditions

From `config/config.yaml` `custom`: `temperature=0.0` (greedy decoding), `max_tokens=16000` (generous budget for long chain-of-thought), `timeout=600s` per request, `max_workers=10` concurrent requests. The `default` split runs all 90 problems; `sample_10` and `sample_100` are the quick and capped variants.

### Edge cases

A response with no extractable number is counted wrong. A truncated response that cuts off before any number is also wrong. Transport/request failures are recorded as incomplete (`complete=false`) and excluded from the accuracy denominator — they are not treated as model errors.

## 5. Config alignment

- **split** (from `config.json`): `default` (all 90 problems), `sample_10` (first 10, for a fast smoke test), and `sample_100` (capped at 100; the local set has only 90, so this is equivalent to the full set). The local data is `data/datasets/aime2025/test.jsonl`, **90 rows** (a symlink to `PostTrainBench/public/datasets/aime2025/test.jsonl`).
- **eval-method**: the benchmark is scored by integer exact-match (`int(round(float(pred))) == int(target)`) computed inside the runner — deterministic, no LLM judge.
- **judge_quality_score**: `100` — a curation-time self-assessment of data quality (max 100), present in `config.json`. The evaluation itself never consults an LLM judge.
