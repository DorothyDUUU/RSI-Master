# CTI-MCQ

> **Benchmark ID**: `cti-mcq` · **Domain**: `domain_knowledge` · **Scorer**: `multiple_choice`

## 1. What this benchmark is

CTI-MCQ is the multiple-choice component of CTIBench, a cyber-threat-intelligence benchmark published at NeurIPS 2024 Datasets and Benchmarks (arXiv:2406.07599). This integration pins `AI4Sec/cti-bench`, config `cti-mcq`, revision `9237e1636ee3e168fbe5ebdcc1c571de0525e568`, test split, under CC-BY-NC-SA-4.0. The local file `data/datasets/cti-mcq/test.jsonl` contains 2,500 English questions.

Every source row provides a URL, a question, four option columns, a dataset-authored composite prompt, and a gold letter. The integration intentionally reconstructs the prompt from the structured question and option columns instead of nesting the dataset-authored instruction inside another instruction. Example:

> Which mitigation prevents applications from running when they were not downloaded from legitimate repositories?
>
> A. Audit
> B. Execution Prevention
> C. Operating System Configuration
> D. User Account Control
>
> Gold: `B`

## 2. Capabilities measured

- Precise knowledge of ATT&CK techniques, mitigations, tactics and recommended data sources.
- Discrimination between closely related security controls.
- Mapping a technical question to one of four structured answers.
- Compliance with a concise single-letter response contract.

It does not test live threat feeds, browsing, malware execution, incident-response writing, retrieval, or CTIBench's separate relation-classification tasks. Accuracy is binary and explanations receive no credit.

## 3. Reference metrics

The local Qwen3.6-35B-A3B run dated 2026-08-10 completed 498 of 500 sampled rows and obtained observed accuracy `0.7149` on completed rows (`qwen36-new46-20260810`, temperature 0, thinking enabled). This is an internal sampled calibration result rather than an official CTIBench score.

No additional numeric paper baseline has been verified in this repository.

## 4. How the model is evaluated

```bash
framework/.venv/bin/python framework/run_eval.py cti-mcq \
  --split default --limit 500 --trial-name my-trial
```

The `hf_multiple_choice` adapter reads `Question`, renders `Option A` through `Option D` once, and appends the standard answer directive:

```text
{Question}

A. {Option A}
B. {Option B}
C. {Option C}
D. {Option D}

Return the single correct option letter as 'ANSWER: X'.
```

The target is `GT`. For structured choice fields, `_targets()` records both the normalized gold letter and its corresponding option text. `multiple_choice_match()` therefore accepts either the final standalone answer letter or an exact normalized rendering of the correct option text; the one lowercase target present in the local data remains valid. A wrong choice or an unparseable completed answer scores zero. Empty responses are incomplete.

The default config uses temperature `0.0`, `max_tokens: 4096`, timeout `180s`, and `max_workers: 8`.

## 5. Config alignment

- The pinned local test file has 2,500 physical rows. Source index `2235` is excluded declaratively because option C is missing while `GT=C`, making that row unscorable; `default` therefore evaluates 2,499 rows. Four other rows have a missing trailing D but valid A–C gold answers and are rendered as three-choice questions without relabeling.
- Manifest and spec use `hf_multiple_choice` with `multiple_choice` scoring.
- Structured fields are `Question`, `Option A`–`Option D`, and `GT`; the redundant source `Prompt` field is not sent.
- License is CC-BY-NC-SA-4.0, `resource-class` is `light`, and no LLM judge is used.
