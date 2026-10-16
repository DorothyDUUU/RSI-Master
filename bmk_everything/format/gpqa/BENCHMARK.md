# GPQA

> **Benchmark ID**: `gpqa` · **Domain**: `domain_knowledge` · **Scorer**: `exact letter match`

## 1. What this benchmark is

GPQA (**G**raduate-**L**evel **G**oogle-**P**roof **Q**&**A**) is a benchmark of graduate-level science multiple-choice questions introduced in the paper [*GPQA: A Graduate-Level Google-Proof Q&A Benchmark*](https://arxiv.org/abs/2311.12022). The questions cover biology, physics, chemistry, and related sciences, and are deliberately written to be hard enough that domain experts do not reliably agree and that looking up the answer online is not a shortcut — this pressures models that rely on retrieval or memorization. The source data is [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa) on Hugging Face; this repository keeps a local copy at `data/datasets/gpqa/gpqa_main.jsonl` (**448 rows**), which the benchmark runner reads directly.

Each item is a four-way multiple-choice question (A/B/C/D). Real sample (first row of the local data, raw fields):

> Question: A large gene has dozens of exons, of which the central ones code for folded triple helical repeats that connect the cytoskeleton with sarcolemma and extracellular space. Each exon usually codes for one folded triple alpha helix. The most common mutations of the gene are central exon deletions that create out-of-frame peptides and progressive degenerative organ waste. A solution is to deliver a Morpholino that recognizes the 5' end of the out-of-frame exon in pre-mRNA. The molecule prevents binding of the spliceosome and creates exon skipping and in-frame joining. Several missing exons are well tolerated by an organism. Which structure below is not involved in the proposed therapy?
> Correct Answer: R-loops
> Incorrect Answer 1: lariat
> Incorrect Answer 2: polyA tail
> Incorrect Answer 3: antisense

The correct answer is `R-loops`; the three other options are the distractors. In one sentence: the model must read a hard graduate-level science question and select the single correct option letter. The questions are in English and aimed at a graduate/expert level in science.

## 2. Capabilities measured

- **Deep, cross-disciplinary scientific knowledge**: the items test advanced knowledge across subfields — here molecular biology of splicing (exon skipping therapy), elsewhere astronomy, physics, and chemistry. The questions are written to be genuinely hard: reasoning from first principles matters, and the gold answer is often subtle enough that it is not recoverable by keyword matching or web search.
- **Multiple-choice discrimination among near-options**: the four options are semantically close (e.g. `R-loops`, `lariat`, `polyA tail`, `antisense` — all nucleic-acid/splicing structures). The model must identify the one that is *not* involved, i.e. reason about the mechanism and reject the plausible decoys rather than match a single salient term.
- **Instruction following and step-by-step reasoning under constrained output**: the prompt template (see §4) tells the model to "Think step by step, then give your final answer" and to end its response with `ANSWER: $LETTER`. The scorer extracts that letter, so the answer must be present in the mandated form at the end of the response — format compliance is as important as content correctness.
- **Position-bias robustness**: option order is shuffled deterministically by `shuffle_seed` (default 42) so the correct answer is not fixed in position A; the model must choose by content, not by position.
- **Answer extraction over free-form reasoning**: because the model is encouraged to reason openly, the scorer must recover the final letter from a long reasoning trace — this measures whether the model can both reason *and* reliably commit to a final answer letter.

**What this does NOT measure (boundary)**: no common-sense reasoning, no code execution, no arithmetic-only computation, no open-ended generation, and no multi-turn dialogue. It is a closed-set, four-choice discriminative task at graduate science level.

**Scoring strictness / failure modes**: no partial credit — a sample fails if the model (a) picks the wrong letter, (b) produces no extractable `ANSWER: X` and no standalone A–D letter (so extraction returns nothing), or (c) formats the answer so the final letter is ambiguous. A reasoning trace that arrives at the right answer but forgets to state `ANSWER: X` may still be recovered by the standalone-letter fallback, but only if the last non-empty line contains a single A–D letter; otherwise it is wrong.

## 3. Reference metrics

The GPQA paper reports public reference values (source: [arXiv:2311.12022](https://arxiv.org/abs/2311.12022)), reproduced here for context:

| Model / baseline | Split | Score | Condition | Source |
|---|---|---|---|---|
| GPT-4 | default | ~39 | zero-shot, chain-of-thought | Paper |
| Domain experts | default | ~74 | limited time | Paper |

These are the paper's published numbers, not results from this repository. The actual registry carries no local numeric reference scores, so local results will be recorded after first evaluation. No human baseline is fabricated here beyond the paper-reported expert figure above.

## 4. How the model is evaluated

### Run command

This is a hand-written benchmark whose data lives in `data/datasets/gpqa/gpqa_main.jsonl`, and it is run through the same unified entrypoint:

From the repository root, with model credentials exported (see `framework/README.md`):

```bash
framework/.venv/bin/python framework/run_eval.py gpqa \
  --limit 500 --trial-name my-trial
```

- `--limit N`: rows per split (`default` evaluates all 448 rows; `--limit 500` runs the full set). Pass `--split sample_100 --limit 100` for a quick smoke run.
- `--trial-name`: output goes to `data/output/gpqa/<trial>/`.
- Dry-run first to verify resolution without calling the model: `... gpqa --dry-run --validate-data`.

### Input

The runner (`format/gpqa/run_benchmark.py`) builds the prompt from the `Question` field and the shuffled choices using `GPQA_PROMPT_TEMPLATE`:

```
Answer the following multiple choice question. The last line of your response should be of the form "ANSWER: $LETTER" (without quotes) where $LETTER is one of A, B, C, D.

{question}

A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Think step by step, then give your final answer.

Reasoning:
```

The choices are shuffled with a deterministic `random.Random(shuffle_seed)` (seed 42 by default), and the gold letter is computed as the position of the correct answer after shuffling.

### Output and scoring

The model should end its response with `ANSWER: X` where X ∈ {A, B, C, D}. `extract_answer()` follows a priority order:

1. the last match of `(?:ANSWER|answer|Answer)\s*:\s*([A-Da-d])` anywhere in the response;
2. if none, the last standalone A–D letter (`\b([A-Da-d])\b`) on the last non-empty line of the response.

The predicted letter is then compared to the gold letter with a **case-insensitive exact equality** (`score_one`). There is no LLM judge and no tolerance.

### Runtime conditions

From `config/config.yaml` `custom`: `temperature=0.0` (greedy decoding), `max_tokens=16000`, `timeout=600s`, `max_workers=6`, `shuffle_seed=42`. Because the prompt asks for step-by-step reasoning before the final letter, `max_tokens` is set high (16000) to accommodate long CoT traces; raising it further is possible for very long-reasoning models.

### Edge cases

If no `ANSWER: X` and no standalone A–D letter can be extracted, the sample is scored wrong. A transport/request failure is recorded as incomplete (`complete=False`, `correct=None`) and excluded from the accuracy denominator; the reward JSON reports `reward: accuracy` when all samples complete, otherwise `null` with the incomplete count in metadata.

## 5. Config alignment

- **split**:
  - `default` — all rows of the local `data/datasets/gpqa/gpqa_main.jsonl` (**448 rows**; the runner's `default` limit is −1 = all). The original GPQA paper's most-validated subset is the Diamond set of 198 questions; this local file is the Main set, so the evaluated pool is 448.
  - `sample_100` — the first 100 rows, used as a fast smoke-test split (runner `SPLITS["sample_100"] = 100`).
- **eval-method**: exact letter match — the custom `extract_answer` + case-insensitive letter equality described above; deterministic, no LLM judge.
- **judge_quality_score**: `85` (from `config.json`) — the maintainers' data-quality self-assessment out of 100, not a model performance score.
- The scorer is not `unscored`, so no `required-metric` is needed.
