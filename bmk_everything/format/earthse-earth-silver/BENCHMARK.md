# EarthSE Earth-Silver

> **Benchmark ID**: `earthse-earth-silver` · **Domain**: `domain_knowledge` · **Scorer**: `multiple_choice`

## 1. What this benchmark is

Earth-Silver is the harder multiple-choice subset of EarthSE, an earth-science benchmark released with the EarthSE study (arXiv:2505.17139). This integration pins `ai-earth/Earth-Silver` revision `bb2f9cc31911a9d95cca9cedf075fa17ba549b08`, config `default`, split `multiple_choice`, under the MIT license. The local evaluation file is `data/datasets/earthse-earth-silver/multiple_choice.jsonl` and contains 250 English questions.

Each row contains a question with four inline options, a gold answer letter, a reference reasoning chain, and subject metadata. The reasoning chain is retained in the local data for auditing but is never shown to the evaluated model. Example:

> How does the APC matrix play a role in correcting measurements taken by the SMAP microwave radiometer?
>
> A) It is used solely for atmospheric corrections.
> B) It refines antenna-sidelobe corrections and removes cross-coupled external signals.
> C) It only performs internal sensor calibration.
> D) It adjusts internal thermal-temperature changes.
>
> Gold: `B`

The task is to select the correct option for a specialist earth-science question and return its letter.

## 2. Capabilities measured

- Specialist knowledge across the lithosphere, hydrosphere, atmosphere and cryosphere.
- Discrimination between technically plausible mechanisms, instruments and observations.
- Reading options embedded in the original question text.
- Constrained single-letter instruction following.

The benchmark does not score explanations, does not expose the stored `reasoning_chain`, and does not evaluate the separate free-form Earth-Silver task. Each item receives binary accuracy with no partial credit.

## 3. Reference metrics

The local Qwen3.6-35B-A3B run dated 2026-08-10 completed 234 of 250 attempted rows and obtained an observed accuracy of `0.6239` on completed rows (`qwen36-new46-20260810`, temperature 0, thinking enabled). Request failures were excluded from that observed-accuracy denominator. This is an internal calibration result, not an official EarthSE leaderboard score.

No additional numeric paper baseline has been verified in this repository.

## 4. How the model is evaluated

Run from the repository root:

```bash
framework/.venv/bin/python framework/run_eval.py earthse-earth-silver \
  --split default --trial-name my-trial
```

`fields.prompt` reads `question`. Because the four `A:`/`A)` through `D:`/`D)` options are already embedded in that field, the runner does not synthesize a second option block. It appends one instruction:

```text
{question_with_inline_options}

Return the single correct option letter as 'ANSWER: X'.
```

`fields.target` reads the bare `answer` letter. `multiple_choice_match()` takes the text after the final `ANSWER:` marker, extracts a standalone letter case-insensitively, and compares it with the normalized target. `ANSWER: B`, `answer: b`, and bare `B` are accepted; option text without a letter is not. Empty responses are incomplete, while a completed response with the wrong or missing letter scores zero.

The default config uses temperature `0.0`, `max_tokens: 4096`, timeout `180s`, and `max_workers: 8`.

## 5. Config alignment

- `default` maps to the pinned HF `multiple_choice` split and the complete 250-row local JSONL.
- Both manifest and runtime spec use `hf_multiple_choice`; runtime scoring is `multiple_choice`.
- License is MIT and `resource-class` is `light`.
- Evaluation is deterministic and uses no LLM judge.
