from __future__ import annotations

"""Generation-time spec fixes for the curated benchmark catalogue.

The overrides below were reviewed and baked into the on-disk `spec.json` of each
generated benchmark (see apply_spec_fixes.py --check for drift).  They live here,
out of `core/`, so the runtime generic runner treats `spec.json` as the sole
source of truth and `core/` never carries benchmark-specific knowledge.
"""


BIGBENCH_REQUIRED_METRICS = {
    "bigbench-auto-categorization": "bleu",
    "bigbench-codenames": "bleu",
    "bigbench-few-shot-nlg": "bleurt",
    "bigbench-hindi-question-answering": "bleu",
    "bigbench-international-phonetic-alphabet-transliterate": "bleu",
    "bigbench-polish-sequence-labeling": "sequence_f1",
    "bigbench-qa-wikidata": "rougeLsum",
    "bigbench-rephrase": "log_likelihood",
    "bigbench-scientific-press-release": "bleu",
    "bigbench-semantic-parsing-in-context-sparc": "bleu",
    "bigbench-semantic-parsing-spider": "bleu",
    "bigbench-social-support": "macro_f1",
    "bigbench-tellmewhy": "bleu",
    "bigbench-topical-chat": "bleu",
}


SPEC_OVERRIDES = {
    "boolq": {
        "fields": {"context": "passage", "prompt": "question", "target": "answer"},
        "scorer": "boolean",
        "target-transform": "boolean_text",
    },
    "global-mmlu": {
        "fields": {"choice_fields": ["option_a", "option_b", "option_c", "option_d"],
                   "prompt": "question", "target": "answer"},
        "scorer": "multiple_choice",
    },
    "global-mmlu-lite": {
        "fields": {"choice_fields": ["option_a", "option_b", "option_c", "option_d"],
                   "prompt": "question", "target": "answer"},
        "scorer": "multiple_choice",
    },
    "xcopa": {
        "fields": {"choice_fields": ["choice1", "choice2"], "prompt": "premise",
                   "question_type": "question", "target": "label"},
        "prompt-transform": "xcopa",
        "scorer": "multiple_choice",
        "target-transform": "choice_index",
    },
    "earthse-earth-silver": {
        "adapter": "hf_multiple_choice",
        "scorer": "multiple_choice",
    },
    "cti-mcq": {
        "adapter": "hf_multiple_choice",
        "exclude-indices": [2235],
        "fields": {"choice_fields": ["Option A", "Option B", "Option C", "Option D"],
                   "prompt": "Question", "target": "GT"},
        "scorer": "multiple_choice",
    },
    "econlogicqa": {
        "fields": {"choice_fields": ["A", "B", "C", "D"],
                   "prompt": "Question", "target": "Answer"},
        "prompt-transform": "event_sequence",
        "scorer": "exact_match",
    },
    "wmdp-bio": {
        "preserve-none-choice": True,
        "target-transform": "choice_index",
    },
    "squad": {
        "fields": {"context": "context", "prompt": "question", "target": "answers"},
        "scorer": "squad",
        "target-transform": "answers_text",
    },
    "squad-2-0": {
        "fields": {"context": "context", "prompt": "question", "target": "answers"},
        "scorer": "squad",
        "target-transform": "answers_text",
    },
    "drop": {
        "fields": {"context": "passage", "prompt": "question", "target": "answer"},
        "scorer": "squad",
        "target-transform": "drop_answer",
    },
    "hf-code-debugging-qa-176ac0": {
        "scorer": "unscored",
        "required-metric": "semantic_or_execution_judge",
    },
    "legalbench": {"scorer": "label"},
    "thai-onet": {
        "fields": {"choices_text": "input", "prompt": "instruction", "target": "result"},
        "scorer": "multiple_choice",
        "target-transform": "leading_choice_number_to_letter",
    },
    "openai-gsm8k": {"target-transform": "gsm8k_final"},
    "rucaibox-gaokao-bench": {"scorer": "multiple_choice"},
    "svc-huggingface-minerva-math": {"target-transform": "boxed_final"},
}


# BigBench tasks whose rows carry no `multiple_choice_targets`/`multiple_choice_scores`
# (empty for every row): their answer lives in `targets` as free-form strings, so the
# scorer must be `exact_match` (BigBench official free-form protocol), not `multiple_choice`.
# Verified by validate_local_datasets.py: mc_choices_lt2 flag on all of these.
BIGBENCH_FREEFORM_IDS = frozenset({
    "bigbench-ascii-word-recognition",
    "bigbench-bridging-anaphora-resolution-barqa",
    "bigbench-chess-state-tracking",
    "bigbench-chinese-remainder-theorem",
    "bigbench-conlang-translation",
    "bigbench-cryptonite",
    "bigbench-disfl-qa",
    "bigbench-gem",
    "bigbench-gender-inclusive-sentences-german",
    "bigbench-language-games",
    "bigbench-linguistic-mappings",
    "bigbench-linguistics-puzzles",
    "bigbench-list-functions",
    "bigbench-matrixshapes",
    "bigbench-minute-mysteries-qa",
    "bigbench-modified-arithmetic",
    "bigbench-mult-data-wrangling",
    "bigbench-object-counting",
    "bigbench-operators",
    "bigbench-paragraph-segmentation",
    "bigbench-parsinlu-reading-comprehension",
    "bigbench-physics-questions",
    "bigbench-repeat-copy-logic",
    "bigbench-simp-turing-concept",
    "bigbench-simple-arithmetic-json",
    "bigbench-simple-arithmetic-json-subtasks",
    "bigbench-simple-text-editing",
    "bigbench-sufficient-information",
    "bigbench-tense",
    "bigbench-word-sorting",
    "bigbench-word-unscrambling",
})


def apply_spec_fixes(spec: dict) -> dict:
    fixed = dict(spec)
    override = SPEC_OVERRIDES.get(spec.get("benchmark-id"), {})
    fixed.update(override)
    if spec.get("benchmark-id") in BIGBENCH_FREEFORM_IDS:
        fixed["scorer"] = "exact_match"
    required = BIGBENCH_REQUIRED_METRICS.get(spec.get("benchmark-id"))
    if required:
        fixed["scorer"] = "unscored"
        fixed["required-metric"] = required
    return fixed
