# Pre-Flight

> **Benchmark ID**: `pre-flight` · **Domain**: `domain_knowledge` · **Scorer**: `multiple_choice`

## 1. What this benchmark is

Pre-Flight is an English aviation-domain multiple-choice dataset from `AirsideLabs/pre-flight-06`, pinned at revision `fdb1e431033db1b9fc7a2e87bdb62478eb0738e1` under the MIT license. The local test file contains 300 questions.

The corpus is broader than one ground-operations manual: it includes airport ground safety and stand scheduling, FAA operational rules, and ICAO annex knowledge. Rows 0–151 generally contain five choices, while rows 152–299 generally contain four; the runner preserves the option count and order stored in each row. Example:

> What should towing crews do at lighted stop bars and red stop-bar lights?
>
> A. Proceed with caution.
> B. Increase speed.
> C. Stop.
> D. Contact ground control.
> E. no suitable option
>
> Gold: `C`

## 2. Capabilities measured

- Aviation ground-operations procedures and safety rules.
- FAA/ICAO regulatory and operational knowledge.
- Applied airport scheduling and conflict-resolution questions.
- Discrimination between plausible but unsafe or non-compliant actions.

The task does not test flight simulation, live operational decisions, tool use, or long-form safety justification. It is a text-only knowledge benchmark; only the final option letter is scored.

## 3. Reference metrics

The local Qwen3.6-35B-A3B run dated 2026-08-10 completed 295 of 300 rows and obtained observed accuracy `0.7322` on completed rows. A later Kimi audit of the first 10 rows (`kimi27-audit10-a1`) completed 10/10 and scored `1.0`; that small audit checks integration behavior and is not a representative full-set capability estimate.

No official public model baseline is recorded in this repository.

## 4. How the model is evaluated

```bash
framework/.venv/bin/python framework/run_eval.py pre-flight \
  --split default --trial-name my-trial
```

The `hf_multiple_choice` adapter reads `input`, renders the row's `choices` in their original order as `A.` onward, and appends the standard `ANSWER: X` directive. The `target` field already contains the correct letter.

`multiple_choice_match()` extracts the final standalone letter case-insensitively and compares it with the target. The scorer supports the row-level four/five-choice variation because validity is determined by the actual rendered choices. Wrong letters or completed prose without a parseable answer letter score zero; empty responses are incomplete.

The default config uses temperature `0.0`, `max_tokens: 4096`, timeout `180s`, and `max_workers: 8`.

## 5. Config alignment

- `default` maps to all 300 rows of the pinned test split.
- Adapter is `hf_multiple_choice`, scorer is `multiple_choice`, and fields map `input`, `choices`, and `target` directly.
- The stored option count is heterogeneous by design; documentation and validation must not assume every row has five choices.
- License is MIT, `resource-class` is `light`, and no LLM judge is used.
