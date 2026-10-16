# EconLogicQA

> **Benchmark ID**: `econlogicqa` · **Domain**: `domain_knowledge` · **Scorer**: `exact_match`

## 1. What this benchmark is

EconLogicQA is an economic sequential-reasoning benchmark from Findings of EMNLP 2024 (arXiv:2405.07938). Each question describes a business, policy, technology or geopolitical scenario and supplies four events that must be placed in their logical or temporal order. This integration pins `yinzhu-quan/econ_logic_qa` revision `28de9728244fd33913a64188d526187ac8e3bdae`, test split, under CC-BY-NC-SA-4.0. The local test file contains 130 English rows.

Example:

> In the context of disinformation from Russian government accounts, arrange the events in logical order.
>
> A. Platforms label government accounts.
> B. Platforms come under pressure to crack down.
> C. Platforms remove posts that violate rules.
> D. Government accounts spread disinformation.
>
> Gold: `D, A, C, B`

The integration renders all four event descriptions; the model is never asked to order unseen labels.

## 2. Capabilities measured

- Ordering four interdependent economic or business events.
- Cause/effect and before/after inference in real-world narratives.
- Coordinating a complete permutation rather than identifying only the first event.
- Following an exact comma-separated output format.

The task does not require numerical economics, document retrieval, symbolic proof, or a written justification. A single misplaced event makes the complete sequence incorrect; no partial credit is awarded.

## 3. Reference metrics

The paper reports test-set accuracy for prompting configurations including GPT-4-Turbo `56.92%` at 1-shot and `56.15%` at 5-shot, and GPT-4 `55.38%` at 1-shot. These numbers use the paper's prompting protocol and are contextual references, not directly interchangeable with this repository's zero-shot template.

The local Qwen3.6-35B-A3B run dated 2026-08-10 completed 122 of 130 rows and obtained observed accuracy `0.4754` on completed rows.

## 4. How the model is evaluated

```bash
framework/.venv/bin/python framework/run_eval.py econlogicqa \
  --split default --trial-name my-trial
```

`choice_fields` renders columns `A`, `B`, `C`, and `D`. The `event_sequence` transform produces:

```text
{Question}

A. {event A}
B. {event B}
C. {event C}
D. {event D}

Return the ordered sequence of option letters (comma-separated) as 'ANSWER: A, B, C, D'.
```

The target is the `Answer` string. `exact_match()` compares the normalized text after the final `ANSWER:` marker with the complete target sequence. Letter case and repeated whitespace are normalized; punctuation and ordering are not. Thus `ANSWER: d, a, c, b` matches `D, A, C, B`, while `D A C B`, extra prose after the marker, or any different permutation fails.

The default config uses temperature `0.0`, `max_tokens: 4096`, timeout `180s`, and `max_workers: 8`.

## 5. Config alignment

- `default` maps to the complete, pinned 130-row test split.
- Adapter is `hf_exact_match`; scorer is `exact_match` because the answer is a four-letter sequence rather than one MCQ letter.
- `choice_fields` and `prompt-transform: event_sequence` are required parts of the runtime contract.
- License is CC-BY-NC-SA-4.0, `resource-class` is `light`, and no LLM judge is used.
