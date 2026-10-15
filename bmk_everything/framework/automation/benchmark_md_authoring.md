# Benchmark MD Authoring Guide (English rewrite)

Your task is to rewrite `BENCHMARK.md` files for a list of benchmarks. **Every file must be
fully English, 5 sections, substantive — never a stub.** The reference for the exact style is
`format/bigbench-arithmetic/BENCHMARK.md` — read it first and match its structure, tone and density.

## Where the facts come from

For each benchmark `<id>` under `format/<id>/`, read ALL of these before writing:

1. **`format/<id>/BENCHMARK.md`** (current, in Chinese) — the richest source. Translate its
   substance into English and keep everything that is concrete and specific. Do not discard detail.
2. **`format/<id>/spec.json`** — the single source of truth at runtime: `adapter`, `scorer`,
   `fields` (prompt/target/choices field names), `target-transform` / `prompt-transform` (if any),
   `dataset` (path, split, `local_rows`, `hf.path` / `hf.name` / `hf.revision`).
3. **`format/<id>/config.json`** — `benchmark-name`, `domain`, `eval-method`, `split` list,
   `required-metric` (present only when `eval-method` is `unscored`).
4. **`format/<id>/config/config.yaml`** — `custom` section gives the real runtime conditions:
   `max_tokens`, `max_workers`, `timeout`, `temperature`, `dataset_revision`.
5. **Local data** — locate the JSONL first: `find data/datasets/<id> -name '*.jsonl'` (most benches
   have `data/datasets/<id>/<split>.jsonl` directly; the 3 hand-written ones live elsewhere, e.g.
   `gsm8k` → `main/`, `gpqa` → `gpqa_main.jsonl`, `aime2025` → `test.jsonl`). Read the first 2–3 rows
   to quote a **real sample** verbatim (question/prompt + choices + target). Do not invent a sample.
6. **Scoring logic** `framework/core/scoring.py` — only if you need to describe the scorer
   precisely (e.g. exact_match / multiple_choice / numeric / boolean / squad / label). Skim the
   relevant function; do not guess.

The registry carries **no numeric reference scores** for most tasks. Do not fabricate any.

## The 5 sections (headings must match exactly)

Write exactly these five `##` headings (the `1.`–`5.` must match the verifier):

```
## 1. What this benchmark is
## 2. Capabilities measured
## 3. Reference metrics
## 4. How the model is evaluated
## 5. Config alignment
```

Also open the file with a `# <Benchmark-name>` H1 and a one-line intro blockquote:
`> **Benchmark ID**: \`<id>\` · **Domain**: \`<domain>\` · **Scorer**: \`<eval-method or scorer>\``

### §1 What this benchmark is
- Origin/background: which suite / paper / source (e.g. "a subtask of BIG-bench", the HF dataset
  `path`/`name`), and a link to the source.
- Data: how many rows (`spec.dataset.local_rows`), local path, HF revision.
- A **real sample** quoted from the local JSONL (prompt + choices + target), then one sentence on
  what the model must do.
- Language and intended audience/level, if clear.

### §2 Capabilities measured  ← the section that must be deepest
Do NOT write "reasoning ability" and stop. For every benchmark enumerate the **specific** skills,
each with its own bullet and concrete grounding:
- What the task actually requires (e.g. multi-step arithmetic + letter-format output;
  extractive QA + span localization; boolean entailment; JSON emission; code traceback, …).
- **Distractor / surface-cue structure** — what makes it a genuine probe (e.g. near-miss options,
  reordered tokens, must-ignore context). Quote real distractors from the sample when possible.
- **Instruction-following / output-format coupling** — the scorer is strict (e.g. exact letter
  match); explain that format compliance is as important as content correctness.
- A **"What this does NOT measure (boundary)"** paragraph: what the task deliberately does not
  test (no algebra, no world knowledge, no multi-hop, no code execution, no follow-up, …).
- **Scoring strictness / failure modes**: enumerate the concrete ways a sample fails (wrong value,
  wrong letter, no parsable answer, malformed JSON, extra text breaking exact match, …).

### §3 Reference metrics
- If `registry`/official metrics exist and you can cite a number **with a source** (paper/table),
  include it with model/split/condition/source columns.
- Otherwise write the honest line (do not fabricate):
  > No public reference score yet; to be filled after first evaluation.
- State explicitly that no human baseline is fabricated.

### §4 How the model is evaluated
Must include, in this order:

**Run command** — a fenced bash block run from the repository root:
````
From the repository root, with model credentials exported (see `framework/README.md`):

```bash
framework/.venv/bin/python framework/run_eval.py <id> \
  --limit 500 --trial-name my-trial
```
````
Then 2–4 bullet lines explaining `--limit`, `--trial-name`, and that `--dry-run --validate-data`
works without calling the model.

**Input** — the prompt shape: which fields feed the user message, any `prompt-transform` /
few-shot, the answer directive. Quote the exact template block.

**Output and scoring** — the expected output format (`ANSWER: X` letter / number / span / yes-no /
JSON), then the scoring chain: `final_text()` (text after the last `final answer:`/`answer:` marker)
+ `normalize()` (NFKC, lower, whitespace-collapse) + the scorer's exact rule (letter extraction
`\b[A-Z]\b` + target set; exact normalized equality; numeric with tolerance 1e-6; SQuAD
normalization; boolean alias folding; label stem prefix). Say who is NOT used when no LLM judge.

**Runtime conditions** — from `config/config.yaml` `custom`: `temperature`, `max_tokens`,
`max_workers`, `timeout`, and the split actually run. If there is a config override documented for
hard reasoning models (e.g. `data/config-overrides/`), mention that `max_tokens` may need raising.

**Edge cases** — unparseable answer counts as wrong; transport/`request_error` stays incomplete and
is excluded from the denominator (not counted as model failure).

### §5 Config alignment
- **split**: list every declared split from `config.json` `split` and its size (from
  `spec.dataset.local_rows` or the local JSONL line count).
- **eval-method**: state the scorer name and that it is deterministic, no LLM judge.
- If `eval-method` is `unscored`, state the `required-metric` (e.g. `bleu`) and that the bench
  produces answers but no reward until that metric is implemented.
- **judge_quality_score**: only if present in `config.json`.

## Hard constraints (the verifier `verify_benchmark_md.py` enforces these)

- ≥ 1500 bytes (aim for a full document, typically 3–6 KB).
- Exactly the five `## N.` headings above (regex `^##\s*N[\.、:：]`).
- No `<[A-Z_]+>` placeholders — do not leave `<MODEL>`, `<SPLIT_NAME>` etc. anywhere.
- The `benchmark-name` (or the id) must appear in the text.
- The `eval-method` (e.g. `multiple_choice`, `exact_match`, `unscored`) must appear in the text.
- Every declared `split` must appear in the text (usually `default`).
- `unscored` benchmarks must mention their `required-metric`.

## Do / Don't

- DO overwrite `format/<id>/BENCHMARK.md` entirely with the English version.
- DO keep every concrete fact from the Chinese original; translate, never drop.
- DO quote a real sample from the local JSONL (check the actual row field names in `fields`).
- DO NOT write generic filler like "measures reasoning" — ground every claim in the task mechanics.
- DO NOT invent reference scores, runtime params, or samples. When unsure, write what the
  spec/config actually say and mark unknowns honestly ("no public reference score yet").
- DO NOT leave any Chinese in the file.
- DO NOT touch any file other than the `BENCHMARK.md` files assigned to you.

## Self-check before reporting done

1. Re-read your file; confirm it is fully English and reads like the reference.
2. Run: `framework/.venv/bin/python framework/automation/verify_benchmark_md.py --ids <id>`
   → expect `{"checked": 1, "passed": 1, "failed": 0}`.
3. Fix any failure; repeat until all your ids pass.
