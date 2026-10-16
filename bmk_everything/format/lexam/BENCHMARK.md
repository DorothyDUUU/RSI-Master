# LEXam

> **Benchmark ID**: `lexam` · **Domain**: `domain_knowledge` · **Scorer**: `multiple_choice`

## 1. What this benchmark is

LEXam is a multilingual legal-exam benchmark released in 2025 (arXiv:2505.12864). This integration pins the `mcq_16_choices` configuration of `LEXam-Benchmark/LEXam`, revision `31e57ae395f92ed7284a4c28d278c55da892b898`, test split, under CC-BY-4.0. The local file has 1,028 rows: 390 English and 638 German questions.

Each item presents four legal propositions and 16 choices representing their possible truth-value subsets. The local `choices` value is serialized as a Python-list-literal string; the runner parses it into the original 16 ordered options. For an item where only propositions i, ii and iii are correct, the corresponding option may be:

> M. i, ii, and iii

with gold index `12`, which maps to the answer letter `M`.

## 2. Capabilities measured

- Proposition-level judgment in Swiss, German and US law.
- Exact subset selection across 16 near-neighbour choices.
- Handling negation and explicit “none of the statements” choices.
- Legal comprehension in both English and German.
- Returning letters beyond A–D without confusing them with proposition labels.

LEXam does not measure legal drafting, citation retrieval, open-ended argumentation, or professional advice. Only the selected option letter is scored and there is no partial credit.

## 3. Reference metrics

The local Qwen3.6-35B-A3B run dated 2026-08-10 completed 484 of 500 sampled rows and obtained observed accuracy `0.3182` on completed rows. This is an internal sampled result with the repository's zero-shot prompt.

No additional numeric paper baseline has been verified in this repository.

## 4. How the model is evaluated

```bash
framework/.venv/bin/python framework/run_eval.py lexam \
  --split default --limit 500 --trial-name my-trial
```

The `hf_multiple_choice` adapter parses the `choices` string using `ast.literal_eval`, verifies an ordered list, and renders all 16 entries from `A.` through `P.` beneath `question`. It then asks for `ANSWER: X`.

The integer `gold` value is a zero-based option index. `_targets()` converts it to a letter (`0 → A`, `12 → M`, `15 → P`). `multiple_choice_match()` extracts a standalone final letter case-insensitively and compares it with that target. Returning the option text instead of its letter does not match.

The default config uses temperature `0.0`, `max_tokens: 4096`, timeout `180s`, and `max_workers: 8`. Reasoning models that consume their output budget internally may require an explicit higher-token override for full evaluations.

## 5. Config alignment

- `default` maps to the complete 1,028-row `mcq_16_choices` test split.
- Adapter and scorer are both multiple-choice oriented; all rows have exactly 16 parsed choices and targets in A–P.
- License is CC-BY-4.0, `resource-class` is `light`, and no LLM judge is used.
