# MedXpertQA

> **Benchmark ID**: `medxpertqa` · **Domain**: `domain_knowledge` · **Scorer**: `multiple_choice`

## 1. What this benchmark is

MedXpertQA is a large, expert-level medical question-answering benchmark released by Tsinghua C3I in 2025 (paper arXiv:2501.18362) and distributed on Hugging Face as `TsinghuaC3I/MedXpertQA` (config `MM`, revision `7e7c465a68eb2b866926bfa59c8c9d17a8daba65`). The suite is widely used as a leading baseline for 2025 medical-LLM evaluation; it ships both a text-only split and a multimodal `MM` split, and every row carries `medical_task` and `body_system` tags so specialty subsets can be extracted. The benchmark is MIT-licensed.

The local data mirrors the `test` split of the `MM` config at `data/datasets/medxpertqa/test.jsonl` with **2,000 rows**. Questions are written in English at the level of board-style / residency medical exams and require clinical domain knowledge; each row also carries an `images` field (attached radiology figures), but the text adapter evaluates the text question only.

Real sample (first row of the local data):

> Q: A 26-year-old man falls from a ladder, landing on his outstretched right hand. He is evaluated in the emergency department and diagnosed with a closed elbow injury without neurovascular compromise. Radiographs are obtained and shown in Figures A and B. During surgery, a sequential approach is used to address each aspect of the injury. Which surgical step is considered to contribute the most to rotatory stability?
> A. Lateral collateral ligament complex repair or reconstruction
> B. Capsular plication
> C. Radial head replacement
> D. Radial head ORIF
> E. Medial collateral ligament complex reconstruction
> A:

The correct answer is `A`. In one sentence: the model must read a clinical vignette, reason about diagnosis/treatment/mechanism, and select the single best of five lettered options.

## 2. Capabilities measured

- **Clinical knowledge recall and application**: each item is a realistic patient vignette (trauma, imaging workup, surgical decision) and requires the model to retrieve and apply board-level medical knowledge — anatomy, imaging interpretation and treatment sequencing — rather than general text comprehension.
- **Clinical reasoning**: rows are tagged `question_type`; the sampled items are `Reasoning`, meaning the correct option is not stated in the prompt but must be derived by reasoning over the vignette (e.g. which surgical step contributes most to rotatory stability).
- **Discriminative choice under near-synonym distractors**: the five options are often clinically similar (different surgical steps, different imaging approaches). Picking the "most optimal" option requires discriminating fine distinctions between plausible alternatives, so keyword-matching a surface term in the vignette is not enough.
- **Instruction following under constrained output**: the prompt demands exactly one option letter in the form `ANSWER: X`. Because scoring is an exact letter match, output-format compliance is as important as clinical correctness.

**What this does NOT measure (boundary)**: no free-form diagnosis generation, no multi-hop conversation or follow-up questions, no numeric dosage computation beyond the vignette, no cross-lingual content, and no interpretation of the attached radiology images — the `images` field is present in the rows but the text adapter does not feed images to the model. The reasoning measured here is single-answer multiple-choice only.

**Scoring strictness / failure modes**: there is no partial credit. A sample fails if (a) the model picks the wrong letter, (b) the response contains no single `[A-Z]` letter, (c) the letter is outside the option set, or (d) the normalized text does not equal the expected letter. Format errors are as fatal as medical errors, so the score conflates "knows the medicine" with "will format the answer correctly".

## 3. Reference metrics

The MedXpertQA paper (arXiv:2501.18362) reports that state-of-the-art GPT-4-class models score well below expert clinicians on this suite, leaving real headroom, but the registry carries no exact numeric reference score for the `test` split mirrored here.

> No public reference score yet; to be filled after first evaluation.

No human baseline is fabricated for this entry.

## 4. How the model is evaluated

### Run command

From the repository root, with model credentials exported (see `framework/README.md`):

```bash
framework/.venv/bin/python framework/run_eval.py medxpertqa \
  --limit 500 --trial-name my-trial
```

- `--limit N`: rows per benchmark (the dataset has 2,000 rows; pass `--limit 2000` to run in full — the thin runner defaults to 3).
- `--trial-name`: output goes to `data/output/medxpertqa/<trial>/`.
- Dry-run first to verify resolution without calling the model: `... medxpertqa --dry-run --validate-data`.

### Input

The `hf_multiple_choice` adapter builds the user message from the `question` field (which already embeds an `Answer Choices: ...` list in the source text) plus the lettered options rendered from the `options` dict, then a fixed answer directive:

```
{question}

A. {option0}
B. {option1}
C. {option2}
D. {option3}
E. {option4}

Return the single correct option letter as 'ANSWER: X'.
```

### Output and scoring

The model should return exactly one option letter in the form `ANSWER: X` (X in A–E). `_targets()` derives the gold letter directly from the `label` field (here `"A"`; the label is already a letter, not an option text, so it is passed through as the target). `scoring.py`'s `multiple_choice_match()` then:

1. takes the text after the last `final answer:` / `answer:` marker (`final_text`);
2. extracts a single `[A-Z]` letter via `\b[A-Z]\b`;
3. judges correct if that letter (normalized) hits the target set, otherwise falls back to exact normalized full-text equality.

`normalize()` applies NFKC folding, lower-casing, whitespace collapse and trailing-period stripping, so `ANSWER: a` and `answer: A` both match.

### Runtime conditions

From `config/config.yaml`: `temperature=0.0` (greedy decoding), `max_tokens=4096`, `timeout=180s`, `max_workers=8`, dataset revision pinned to `7e7c465a68eb2b866926bfa59c8c9d17a8daba65`.

### Edge cases

No `[A-Z]` letter extracted, or a letter outside the target set, counts as wrong; the normalized-text fallback still failing counts as wrong. Transport/request failures are recorded as `incomplete` (`request_error`) and are excluded from the denominator — they are not counted as model failures (see `apply_failure_scoring.py`).

## 5. Config alignment

- **split**: `default` — the only declared evaluation set in `config.json`, which maps to the mirrored `test` split `data/datasets/medxpertqa/test.jsonl` (**2,000 rows**, `spec.dataset.local_rows`); the full set is the default evaluation target.
- **eval-method**: `multiple_choice`, i.e. `scoring.py::multiple_choice_match` — exact option-letter matching, no LLM judge.
- **judge_quality_score**: the registry carries no `quality_score`, so none is stated.
- The scorer is not `unscored`, so no `required-metric` is needed.
