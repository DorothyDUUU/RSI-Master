# GSM8K

> **Benchmark ID**: `gsm8k` · **Domain**: `math_reasoning` · **Scorer**: `numeric` (exact float equality)

## 1. What this benchmark is

GSM8K (Grade School Math 8K) is a dataset of grade-school-level math word problems released by OpenAI and described in the paper [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168). Each problem is a short natural-language story that must be solved with several arithmetic steps, and the answer is a single number (integer or decimal). The data source is the Hugging Face dataset [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) (config `main`, revision `cc7b047b6e5bb11b4f1af84efc572db110a51b3c`).

This is a hand-written integration. Its local mirror lives at `data/datasets/gsm8k/main/` as the HF parquet files `train-00000-of-00001.parquet` and `test-00000-of-00001.parquet`; the evaluation runs the **test** split with **1319 rows** (`default`). The `sample_100` split evaluates the first 100 test questions for quick smoke runs. All problems are in English and target the arithmetic competence expected at grade school level.

Real sample (first row of the local test data):

> Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?

with gold answer `18` (the final `####` value in the `answer` field, whose preceding text is the step-by-step reasoning: "Janet sells 16 - 3 - 4 = 9 duck eggs a day. She makes 9 * 2 = $18 every day at the farmer's market."). In one sentence: the model must parse a natural-language word problem, carry out the multi-step arithmetic, and emit a single numeric answer.

## 2. Capabilities measured

- **Multi-step arithmetic reasoning**: decomposing a story problem into sequential operations (subtracting daily consumption from egg production, then multiplying the remainder by the unit price) and computing an exact numeric result. Correctness depends on the final value, not on the working shown.
- **Problem-to-equation parsing**: mapping the narrative entities and quantities (eggs per day, eggs eaten, eggs baked, price per egg) onto the right operations and avoiding the irrelevant details that are present but not needed.
- **10-shot chain-of-thought format following**: by default the prompt prepends 10 few-shot exemplars (sampled from the train split with `seed=42`) that demonstrate step-by-step reasoning ending in `ANSWER: <number>`, and the answer directive explicitly requires the response to close with `ANSWER: $ANSWER`. The runner extracts only the final number, so a response that reaches the right value but never states it in a parsable way fails.

**Distractor structure**: word problems embed real-world units (`$`, `%`, thousands separators) and extra quantities that must be ignored or combined; the difficulty is in selecting the correct operation chain, not in reading a single equation.

**What this does NOT measure (boundary)**: no world knowledge, no code execution, no long-context retrieval, and no open-ended or explanatory grading — the scorer only checks the numeric answer. There is no partial credit for a correct method with an arithmetic slip.

**Scoring strictness / failure modes**: the extracted answer must be numerically equal to the gold value (float equality; the extractor strips commas, `$` and `%`, and the comparison tolerates integer/decimal equivalence). A sample fails if the wrong number is produced, the format is illegal, or no number can be extracted from the response. Answers that are textually close but numerically wrong, or formatted as prose without a final number, are also wrong.

## 3. Reference metrics

The GSM8K paper reports a human performance of roughly 88% on the test set, and a 5-shot chain-of-thought GPT-4 evaluation on GSM8K is commonly cited around 92% (approximate public values, not registry figures). These are public references only:

| Model | Split | Score | Condition | Source |
|---|---|---|---|---|
| GPT-4 | default | ~92 | 5-shot, chain-of-thought | Common public evaluation value |
| Human | default | ~88 | – | GSM8K paper (arXiv:2110.14168) |

The registry carries no first-party numeric score for this benchmark; the table above is cited from public sources, and the actual figure is to be confirmed by first evaluation in this repo. No human baseline is fabricated beyond the paper's reported value.

## 4. How the model is evaluated

### Run command

From the repository root, with model credentials exported (see `framework/README.md`):

```bash
framework/.venv/bin/python framework/run_eval.py gsm8k \
  --limit 500 --trial-name my-trial
```

- `--limit N`: rows per benchmark (datasets ≤ N run in full; the thin runner defaults to 3 — pass it explicitly).
- `--trial-name`: output goes to `data/output/gsm8k/<trial>/`.
- Dry-run first to verify resolution without calling the model: `... gsm8k --dry-run --validate-data`.

### Input

The runner wraps the question in a fixed `MATH_PROMPT_TEMPLATE` that instructs step-by-step reasoning ending in `ANSWER: $ANSWER`, and (when `fewshot > 0`) prepends the 10-shot block. Each few-shot exemplar is formatted as `{question}\n\nReasoning:\n{reasoning}\n\nANSWER: {target}`. The user message is:

```
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

<question>

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \boxed command.

Reasoning:
```

### Output and scoring

The model should reason step by step and end with `ANSWER: <number>`. Extraction (in `run_benchmark.py::extract_answer`) takes, in priority order: (1) the last `ANSWER: <number>` match, (2) the last `\boxed{<number>}`, (3) the last standalone number found in the response. The extracted value is compared to the gold value with `score_one`, which uses float equality after stripping commas, `$` and `%` (falling back to string equality if conversion fails). This is a deterministic numeric exact-match scorer; no LLM judge is involved.

### Runtime conditions

From `config/config.yaml`: `fewshot=10` (sampled with `fewshot_seed=42`), `temperature=0.0` (greedy decoding), `max_tokens=3000`, `timeout=600s`, `max_workers=30`, split from `general.split` (default `default`, i.e. all 1319 test rows; `sample_100` for the first 100). If a hard reasoning model is used via a `data/config-overrides/` override, `max_tokens` may need raising.

### Edge cases

If no number can be extracted from the response, or the extracted number is not float-equal to the gold value, the sample is wrong. Request failures or timeouts are recorded as `incomplete` (`request_error`) and excluded from the accuracy denominator — they are not counted as model failures.

## 5. Config alignment

- **split**:
  - `default`: the full test set, **1319 rows** (matches the 1319-row `test-00000-of-00001.parquet` local mirror and the HF `openai/gsm8k` `main` config test split).
  - `sample_100`: the first 100 test rows, for fast smoke runs.
- **eval-method**: numeric exact-match — deterministic `float` equality scoring, no LLM judge.
- **judge_quality_score**: `95` (config.json) — a data-quality self-assessment out of 100 for this benchmark.
- The scorer is not `unscored`, so no `required-metric` is needed.
