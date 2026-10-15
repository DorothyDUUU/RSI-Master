#!/usr/bin/env python3
"""Build a reproducible 200-item lightweight benchmark registry.

Only primary Hugging Face metadata/dataset-server responses are accepted.  The
script never calls a model and never downloads full datasets.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path


PERMISSIVE = {
    "apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "cc0-1.0",
    "cc-by-3.0", "cc-by-4.0", "cc-by-sa-3.0", "cc-by-sa-4.0",
}
HEAVY = re.compile(
    r"(android|browser|docker|gui|multimodal|osworld|repoexec|screen|swe[-_ ]?bench|"
    r"terminal|webarena|webshop|webvoyager|windows|workarena|mathvision|math-v|gsm8k-v)", re.I
)
BLOCKED_DATASETS = {"gabeorlanski/bc-humaneval", "heegyu/bbq"}  # legacy dataset scripts; unsupported by datasets>=4
PROMPTS = ("question", "prompt", "input", "inputs", "problem", "instruction", "query", "text")
TARGETS = ("answer", "answers", "target", "targets", "label", "output", "solution",
           "reference_answer", "correct_answer")
CHOICES = ("choices", "options", "multiple_choice_targets", "endings", "candidates")
SPLIT_ORDER = ("test", "validation", "dev", "train")

BIGBENCH_DOMAINS = {
    "agentic_tool_calling": [
        "goal_step_wikihow", "navigate", "intent_recognition", "known_unknowns",
        "temporal_sequences", "strategyqa", "logical_deduction",
        "reasoning_about_colored_objects", "movie_dialog_same_or_different", "play_dialog_same_or_different", "codenames",
        "evaluating_information_essentiality", "fact_checker", "fantasy_reasoning",
        "minute_mysteries_qa", "disambiguation_qa", "implicit_relations", "epistemic_reasoning",
        "causal_judgment", "cause_and_effect", "auto_categorization", "topical_chat",
        "social_support", "sufficient_information", "question_selection", "tellmewhy",
        "penguins_in_a_table", "which_wiki_edit", "what_is_the_tao",
        "unnatural_in_context_learning", "real_or_fake_text",
    ],
    "code_swe": [
        "auto_debugging", "code_line_description", "cs_algorithms", "list_functions",
        "dyck_languages", "key_value_maps", "ascii_word_recognition", "mult_data_wrangling",
        "simple_text_editing", "operators", "python_programming", "program_synthesis",
        "repeat_copy_logic", "regular_expression", "unit_interpretation", "word_sorting",
        "symbol_interpretation", "linguistic_mappings", "formal_fallacies_syllogisms_negation",
        "logical_args", "logic_grid_puzzle", "object_counting", "tracking_shuffled_objects",
        "semantic_parsing_in_context_sparc", "semantic_parsing_spider", "undo_permutation",
        "word_unscrambling", "paragraph_segmentation", "simple_arithmetic_json",
    ],
    "math_reasoning": [
        "arithmetic", "modified_arithmetic", "elementary_math_qa", "chinese_remainder_theorem",
        "mathematical_induction", "intersect_geometry", "geometric_shapes", "logical_sequence",
        "abstract_narrative_understanding", "analogical_similarity", "analytic_entailment",
        "checkmate_in_one", "chess_state_tracking", "date_understanding", "matrixshapes",
        "riddle_sense", "physical_intuition", "physics", "physics_questions",
        "identify_math_theorems", "simple_arithmetic_json_subtasks", "unit_conversion",
        "periodic_elements", "mnist_ascii", "odd_one_out", "similarities_abstraction",
    ],
}

CURATED = [
    ("arc-challenge", "ARC Challenge", "domain_knowledge", "allenai/ai2_arc", "ARC-Challenge", "test",
     "210d026faf9955653af8916fad021475a3f00453", "cc-by-sa-4.0",
     {"prompt": "question", "target": "answerKey", "choices": "choices"}, None, "multiple_choice"),
    ("commonsenseqa", "CommonsenseQA", "domain_knowledge", "tau/commonsense_qa", "default", "validation",
     "94630fe30dad47192a8546eb75f094926d47e155", "mit",
     {"prompt": "question", "target": "answerKey", "choices": "choices"}, None, "multiple_choice"),
    ("drop", "DROP", "domain_knowledge", "EleutherAI/drop", "default", "validation",
     "619386fc7ffe6dace3cc6d3483c69b164ebf6bf3", "cc-by-4.0",
     {"prompt": "question", "context": "passage", "target": "answer.spans"}, None, "exact_match"),
    ("lab-bench-dbqa", "LAB-Bench DBQA", "domain_knowledge", "futurehouse/lab-bench", "DbQA", "train",
     "5c77cec648430f30611808808861eb86f81d5eaa", "cc-by-sa-4.0",
     {"prompt": "question", "target": "ideal", "choices_merge": ["ideal", "distractors"]},
     "choice_text_to_letter", "multiple_choice"),
    ("wmdp-bio", "WMDP", "domain_knowledge", "cais/wmdp", "wmdp-bio", "test",
     "7125571f22f032c56415e7980f48d877dd830ff8", "mit",
     {"prompt": "question", "target": "answer", "choices": "choices"}, "choice_index", "multiple_choice"),
    ("bbeh", "BIG-Bench Extra Hard", "math_reasoning", "BBEH/bbeh", "default", "train",
     "08e07a803851822c04399782ece3c4a07ce419f9", "apache-2.0",
     {"prompt": "input", "target": "target"}, None, "exact_match"),
    ("musr", "MuSR", "math_reasoning", "TAUR-Lab/MuSR", "default", "murder_mysteries",
     "7c365b439a222150f317764d4f16ae6c96d7d94a", "cc-by-4.0",
     {"prompt": "question", "context": "narrative", "target": "answer_index", "choices": "choices"},
     "choice_index", "multiple_choice"),
    ("pre-flight", "Pre-Flight", "domain_knowledge", "AirsideLabs/pre-flight-06", "default", "test",
     "fdb1e431033db1b9fc7a2e87bdb62478eb0738e1", "mit",
     {"prompt": "input", "target": "target", "choices": "choices"}, None, "multiple_choice"),
    ("thai-onet", "Thai O-NET", "domain_knowledge", "matichon/thai-onet-m6-exam", "default", "test",
     "93ffb5e3f3ec630b73e501937805984dd24f2365", "apache-2.0",
     {"prompt": "input", "context": "instruction", "target": "result"}, None, "exact_match"),
]


def curated_specs() -> list[dict]:
    result = []
    for benchmark_id, name, domain, path, config, split, revision, license_name, fields, transform, scorer in CURATED:
        item = {
            "benchmark-id": benchmark_id, "benchmark-name": name, "domain": domain,
            "adapter": f"hf_{scorer}", "scorer": scorer,
            "dataset": {"path": path, "name": config, "split": split, "revision": revision},
            "fields": fields,
            "license": {"spdx": license_name, "evidence": f"https://huggingface.co/api/datasets/{path}"},
            "resource-class": "light", "source": f"https://huggingface.co/datasets/{path}",
            "selection": "curated-primary-metadata-and-schema-verified",
        }
        if transform:
            item["target-transform"] = transform
        if benchmark_id == "lab-bench-dbqa":
            item["shuffle-choices"] = True
        result.append(item)
    return result


def get_json(url: str) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": "BmkEverything-registry-builder/1"})
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.load(response)


def slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:58]
    return value or hashlib.sha1(text.encode()).hexdigest()[:12]


def license_from_tags(tags: list[str]) -> str | None:
    licenses = [tag.removeprefix("license:").lower() for tag in tags if tag.startswith("license:")]
    return licenses[0] if len(licenses) == 1 and licenses[0] in PERMISSIVE else None


def infer_fields(row: dict) -> dict | None:
    prompt = next((key for key in PROMPTS if key in row and isinstance(row[key], str)), None)
    target = next((key for key in TARGETS if key in row), None)
    choices = next((key for key in CHOICES if key in row and row[key]), None)
    if not prompt or not target:
        return None
    result = {"prompt": prompt, "target": target}
    if choices and choices != target:
        result["choices"] = choices
    return result


def inspect_hf(candidate: dict) -> dict | None:
    hf_id = candidate.get("hf_id")
    if not hf_id and "huggingface.co/datasets/" in candidate.get("url", ""):
        hf_id = candidate["url"].split("/datasets/", 1)[1].strip("/").split("/")[0:2]
        hf_id = "/".join(hf_id)
    if not hf_id or hf_id in BLOCKED_DATASETS or HEAVY.search(candidate.get("name", "")):
        return None
    encoded = urllib.parse.quote(hf_id, safe="")
    metadata = get_json(f"https://huggingface.co/api/datasets/{hf_id}")
    license_name = license_from_tags(metadata.get("tags", []))
    if not license_name:
        return None
    splits_payload = get_json(f"https://datasets-server.huggingface.co/splits?dataset={encoded}")
    splits = splits_payload.get("splits", [])
    if not splits:
        return None
    chosen = min(splits, key=lambda item: (SPLIT_ORDER.index(item["split"])
                 if item["split"] in SPLIT_ORDER else 99, item["config"]))
    params = urllib.parse.urlencode({"dataset": hf_id, "config": chosen["config"], "split": chosen["split"]})
    first = get_json(f"https://datasets-server.huggingface.co/first-rows?{params}")
    rows = first.get("rows", [])
    if not rows:
        return None
    fields = infer_fields(rows[0]["row"])
    if not fields:
        return None
    feature_text = json.dumps(first.get("features", []), ensure_ascii=False).lower()
    if '"image"' in feature_text or '"audio"' in feature_text:
        return None
    scorer = "multiple_choice" if "choices" in fields else (
        "numeric" if candidate.get("domain") == "math_reasoning" else "exact_match"
    )
    return {
        "benchmark-id": slug(candidate.get("id") or candidate["name"]),
        "benchmark-name": candidate["name"],
        "domain": candidate["domain"],
        "adapter": f"hf_{scorer}",
        "scorer": scorer,
        "dataset": {"path": hf_id, "name": chosen["config"], "split": chosen["split"],
                    "revision": metadata["sha"]},
        "fields": fields,
        "license": {"spdx": license_name, "evidence": f"https://huggingface.co/api/datasets/{hf_id}"},
        "resource-class": "light",
        "source": f"https://huggingface.co/datasets/{hf_id}",
        "selection": "survey-revalidated-and-schema-inspected",
    }


def bigbench_specs(existing: list[dict], needed: int) -> list[dict]:
    split_data = get_json("https://datasets-server.huggingface.co/splits?dataset=tasksource%2Fbigbench")
    configs = sorted({item["config"] for item in split_data["splits"]})
    used = {item["dataset"].get("name") for item in existing if item["dataset"]["path"] == "tasksource/bigbench"}
    domain_by_config = {name: domain for domain, names in BIGBENCH_DOMAINS.items() for name in names}
    ordered = [name for names in BIGBENCH_DOMAINS.values() for name in names]
    ordered += [name for name in configs if name not in ordered]
    result = []
    counts = Counter(item["domain"] for item in existing)
    # First close domain deficits, then quality-first fill from all remaining tasks.
    deficit_order = []
    for domain in ("agentic_tool_calling", "code_swe", "math_reasoning", "domain_knowledge"):
        deficit_order.extend(name for name in ordered if domain_by_config.get(name, "domain_knowledge") == domain)
    for name in deficit_order + ordered:
        if name in used or name not in configs:
            continue
        domain = domain_by_config.get(name, "domain_knowledge")
        # Ensure every top-level domain reaches 25 before quality-first fill.
        if any(counts[key] < 25 for key in ("agentic_tool_calling", "code_swe", "math_reasoning", "domain_knowledge")):
            if counts[domain] >= 25:
                continue
        split = next(item["split"] for item in split_data["splits"] if item["config"] == name
                     and item["split"] in SPLIT_ORDER)
        scorer = "multiple_choice" if name not in {"auto_debugging", "code_line_description"} else "exact_match"
        result.append({
            "benchmark-id": f"bigbench-{slug(name)}",
            "benchmark-name": f"BIG-bench: {name}",
            "domain": domain,
            "adapter": "bigbench",
            "scorer": scorer,
            "dataset": {"path": "tasksource/bigbench", "name": name, "split": split,
                        "revision": "210c156767d2f4f05d2f4fd0bb275017a67040fd"},
            "fields": {"prompt": "inputs", "target": "targets", "choices": "multiple_choice_targets"},
            "license": {"spdx": "apache-2.0", "evidence": "https://github.com/google/BIG-bench/blob/main/LICENSE"},
            "resource-class": "light",
            "source": f"https://huggingface.co/datasets/tasksource/bigbench/viewer/{name}",
            "selection": "bigbench-distinct-task-lightweight-deterministic",
        })
        used.add(name)
        counts[domain] += 1
        if len(result) >= needed:
            break
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=int, default=200)
    args = parser.parse_args()
    candidates = []
    seen = set()
    for path in args.inputs:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = re.sub(r"[^a-z0-9]", "", row.get("name", "").lower())
            if key and key not in seen and row.get("data_source") == "huggingface":
                seen.add(key)
                candidates.append(row)
    specs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(inspect_hf, row): row for row in candidates}
        for future in concurrent.futures.as_completed(futures):
            try:
                spec = future.result()
            except Exception as exc:
                print(f"skip {futures[future].get('name')}: {exc}")
                continue
            if spec:
                specs.append(spec)
    specs.extend(curated_specs())
    specs.sort(key=lambda item: (-int(item["domain"] == "agentic_tool_calling"), item["domain"], item["benchmark-id"]))
    ids = set()
    specs = [item for item in specs if not (item["benchmark-id"] in ids or ids.add(item["benchmark-id"]))]
    specs += bigbench_specs(specs, args.target - len(specs))
    specs = specs[:args.target]
    counts = Counter(item["domain"] for item in specs)
    unique_ids = {item["benchmark-id"] for item in specs}
    if len(specs) != args.target or len(unique_ids) != args.target or any(
        counts[domain] < 25 for domain in BIGBENCH_DOMAINS | {"domain_knowledge": []}
    ):
        raise SystemExit(f"selection gates failed: count={len(specs)} domains={dict(counts)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in specs))
    print(json.dumps({"count": len(specs), "domains": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
