from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .provider import chat_detailed
from .legacy_bridge import parse_sample_indices
from .scoring import (boolean_match, exact_match, json_match, label_match,
                      multiple_choice_match, numeric_match, squad_match, unscored)


# BmkEverything/data/datasets/<id>/<split>.jsonl — root of the local data tree.
REPO_ROOT = Path(os.environ.get("BENCHMARK_ROOT") or Path(__file__).resolve().parents[2]).expanduser().resolve()


SCORERS = {
    "exact_match": exact_match,
    "boolean": boolean_match,
    "label": label_match,
    "multiple_choice": multiple_choice_match,
    "numeric": numeric_match,
    "json": json_match,
    "squad": squad_match,
    "unscored": unscored,
}


def _read(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def _nested(row: dict, path: str):
    value = row
    for part in path.split("."):
        if isinstance(value, dict):
            value = value[part]
        else:
            raise KeyError(path)
    return value


def _targets(row: dict, spec: dict) -> list[object]:
    if spec["adapter"] == "bigbench":
        choices = row.get("multiple_choice_targets") or []
        scores = row.get("multiple_choice_scores") or []
        if choices and scores:
            return [chr(65 + i) for i, score in enumerate(scores) if score == max(scores)]
        return list(row.get("targets") or [])
    value = _nested(row, spec["fields"]["target"])
    transform = spec.get("target-transform")
    if transform == "answers_text":
        answers = list((value or {}).get("text") or [])
        return answers or ["unanswerable", "no answer"]
    if transform == "boolean_text":
        return ["yes" if bool(value) else "no"]
    if transform == "drop_answer":
        answer = value or {}
        targets = list(answer.get("spans") or [])
        if answer.get("number") not in (None, ""):
            targets.append(str(answer["number"]))
        date = answer.get("date") or {}
        parts = [str(date.get(key)) for key in ("day", "month", "year") if date.get(key)]
        if parts:
            targets.append(" ".join(parts))
        return targets
    if transform == "gsm8k_final":
        matches = re.findall(r"####\s*([^\n]+)", str(value))
        return [matches[-1].strip()] if matches else [value]
    if transform == "boxed_final":
        text = str(value)
        start = text.rfind(r"\boxed{")
        if start != -1:
            # Extract the box content with balanced-brace matching, so nested
            # LaTeX like \frac{a}{b} inside \boxed{...} is kept whole.
            depth = 0
            i = start + len(r"\boxed{") - 1
            while i < len(text):
                ch = text[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        content = text[start + len(r"\boxed{"):i]
                        return [content.strip()]
                i += 1
        numbers = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)
        return [numbers[-1]] if numbers else [value]
    if transform == "json_list_target":
        parsed = json.loads(value) if isinstance(value, str) else value
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        return [str(parsed)] if parsed is not None else [""]
    if transform == "leading_choice_number_to_letter":
        candidate = value[0] if isinstance(value, list) and value else value
        match = re.match(r"\s*(\d+)", str(candidate))
        if not match:
            raise ValueError(f"target has no leading choice number: {candidate!r}")
        return [chr(64 + int(match.group(1)))]
    if transform == "choice_index":
        return [chr(65 + int(value))]
    if transform == "choice_text_to_letter":
        choices = _choices(row, spec)
        return [chr(65 + choices.index(value))]
    if spec.get("scorer") == "multiple_choice":
        choices = _choices(row, spec)
        if isinstance(value, int) and 0 <= value < len(choices):
            return [chr(65 + value)]
        if value in choices:
            return [chr(65 + choices.index(value))]
    result = value if isinstance(value, list) else [value]
    # Content-aware MCQ: some answer keys are option letters while the model
    # restates the option *text* (e.g. CS-Bench, ChemLLMBench).  Accept both the
    # letter and the exact option content so a correct-but-not-letter answer
    # still scores.
    if spec.get("fields", {}).get("choice_fields"):
        choices = _choices(row, spec)
        if choices:
            expanded = []
            for target in result:
                text = str(target).strip()
                if len(text) == 1 and text.upper() in "ABCD":
                    index = ord(text.upper()) - 65
                    if 0 <= index < len(choices):
                        expanded.append(text.upper())
                        expanded.append(str(choices[index]))
                        continue
                expanded.append(target)
            result = expanded
    return result


def _choices(row: dict, spec: dict) -> list[object]:
    fields = spec.get("fields", {})
    if fields.get("choice_fields"):
        result = [_nested(row, item) for item in fields["choice_fields"]]
    elif fields.get("choices_merge"):
        result = []
        for item in fields["choices_merge"]:
            value = _nested(row, item)
            result.extend(value if isinstance(value, list) else [value])
    else:
        choice_path = fields.get("choices")
        choices = _nested(row, choice_path) if choice_path else []
        # Some HF mirrors serialize list/dict literals as strings ("['a','b']",
        # "{'0':'x','1':'y'}"). Recover the real structure before interpreting.
        if isinstance(choices, str) and choices.lstrip().startswith(("[", "{")):
            try:
                choices = ast.literal_eval(choices)
            except (ValueError, SyntaxError):
                pass
        if isinstance(choices, dict):
            choices = choices.get("text") or choices.get("choices") or list(choices.values())
        result = list(choices or [])
    # Drop null/blank options. Some datasets use the string "None" as a
    # placeholder, while others (for example WMDP) use it as a legitimate
    # answer choice. The spec opts into preserving the latter case so the
    # generic runner does not need benchmark-id-specific logic.
    blank_strings = {""} if spec.get("preserve-none-choice") else {"", "None"}
    result = [
        item for item in result
        if item is not None and str(item).strip() not in blank_strings
    ]
    if fields.get("choices_text"):
        result = []
        # Handles both numeric ("1. opt", "1) opt") and lettered ("a) opt")
        # option lines, which several domain datasets serialize as text blocks.
        for line in str(_nested(row, fields["choices_text"])).splitlines():
            match = re.match(r"\s*(?:\d+|[A-Za-z])[.)]\s*(.+)", line)
            if match:
                result.append(match.group(1).strip())
    if spec.get("shuffle-choices"):
        identity = row.get("id") or row.get("idx") or json.dumps(row, sort_keys=True, default=str)
        seed = int(hashlib.sha256(f"{spec['benchmark-id']}:{identity}".encode()).hexdigest()[:16], 16)
        random.Random(seed).shuffle(result)
    return result


def _prompt(row: dict, spec: dict) -> str:
    if spec["adapter"] == "bigbench":
        question = row["inputs"]
        choices = row.get("multiple_choice_targets") or []
    else:
        question = _nested(row, spec["fields"]["prompt"])
        if spec.get("prompt-transform") == "xcopa":
            relation = str(_nested(row, spec["fields"]["question_type"])).lower()
            question = f"Premise: {question}\nWhich option is the most plausible {relation}?"
        context_path = spec.get("fields", {}).get("context")
        if context_path:
            question = f"{_nested(row, context_path)}\n\n{question}"
        if spec.get("prompt-transform") == "evidence_context":
            # Inject a document passage (list-of-dicts with evidence_text, as in
            # FinanceBench) ahead of the question so the model can ground its answer.
            evidence = _nested(row, spec.get("fields", {}).get("evidence") or "evidence")
            texts = []
            for item in evidence if isinstance(evidence, list) else [evidence]:
                if isinstance(item, dict):
                    texts.append(str(item.get("evidence_text") or item.get("evidence_text_full_page") or ""))
                else:
                    texts.append(str(item))
            if texts:
                question = "\n\n".join(texts) + "\n\n" + question
        choices = _choices(row, spec)
        if spec.get("prompt-transform") == "chemllmbench_mcq":
            # ChemLLMBench questions are phrased as generation ("Create a
            # molecule…") but the answer key is a letter into the A–D options,
            # which are wrapped in first-person filler ("In my opinion, the
            # answer would be …").  Rephrase as explicit selection and strip the
            # wrapper so the model treats the options as candidates, not
            # free-form generations.
            question = "Which of the following candidate molecules satisfies the description?\n\n" + question
            wrapper = re.compile(
                r"^(?:[Ii] would say that|[Ii]n my opinion, the answer would be|"
                r"[Aa]s for my answer, it is|[Ii]n response, I offer|[Ii] believe the answer is|"
                r"[Ii] would conclude that|[Ii] would respond by saying|[Ff]rom my perspective, the answer is|"
                r"[Mm]y response is|[Mm]y resp|[Mm]y answer is)\s+"
            )
            choices = [wrapper.sub("", str(item).strip()) for item in choices]
    if choices:
        options = "\n".join(f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices))
        if spec.get("prompt-transform") == "event_sequence":
            return f"{question}\n\n{options}\n\nReturn the ordered sequence of option letters (comma-separated) as 'ANSWER: A, B, C, D'."
        return f"{question}\n\n{options}\n\nReturn the single correct option letter as 'ANSWER: X'."
    if spec.get("scorer") == "multiple_choice":
        return f"{question}\n\nReturn the single correct option letter as 'ANSWER: X'."
    return f"{question}\n\nReturn only the final answer, prefixed with 'ANSWER:'."


def _load_rows(spec: dict, limit: int):
    source = spec["dataset"]
    excluded_indices = {int(index) for index in spec.get("exclude-indices", [])}
    local_path = source.get("path", "")
    if local_path.startswith("data/datasets"):
        # Local-first: rows were downloaded to data/datasets/<id>/<split>.jsonl.
        jsonl_path = REPO_ROOT / local_path / f"{source['split']}.jsonl"
        rows = []
        with jsonl_path.open(encoding="utf-8") as stream:
            for source_index, line in enumerate(stream):
                line = line.strip()
                if not line:
                    continue
                if source_index in excluded_indices:
                    continue
                rows.append(json.loads(line))
                if len(rows) >= limit:
                    break
        return rows
    if spec["adapter"] == "bigbench":
        params = {
            "dataset": source["path"],
            "config": source["name"],
            "split": source["split"],
            "offset": 0,
            "length": limit,
        }
        if source.get("revision"):
            params["revision"] = source["revision"]
        url = "https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(params)
        request = urllib.request.Request(url, headers={"User-Agent": "BmkEverything/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.load(response)
        return [dict(item["row"]) for item in payload.get("rows", [])]

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("missing dependency: datasets") from exc
    kwargs = {"path": source["path"], "split": source["split"], "streaming": True}
    if source.get("name"):
        kwargs["name"] = source["name"]
    if source.get("revision"):
        kwargs["revision"] = source["revision"]
    dataset = load_dataset(**kwargs)
    rows = []
    for row in dataset:
        rows.append(dict(row))
        if len(rows) >= limit:
            break
    return rows


def dry_run(spec: dict) -> None:
    required = {"benchmark-id", "adapter", "dataset", "scorer", "license", "resource-class"}
    missing = sorted(required - spec.keys())
    if missing:
        raise ValueError(f"missing spec keys: {missing}")
    if spec["scorer"] not in SCORERS:
        raise ValueError(f"unsupported scorer: {spec['scorer']}")
    if spec["resource-class"] != "light":
        raise ValueError("initial cohort only permits resource-class=light")
    if spec["adapter"] != "bigbench" and not {"prompt", "target"}.issubset(spec.get("fields", {})):
        raise ValueError("generic HF adapter requires explicit prompt and target fields")


def run(bench_dir: Path, config_path: Path, limit: int, *, validate_data: bool = False) -> None:
    spec = _read(bench_dir / "spec.json")
    dry_run(spec)
    if validate_data:
        rows = _load_rows(spec, min(limit, 1))
        if not rows:
            raise RuntimeError("dataset returned no rows")
        _prompt(rows[0], spec)
        _targets(rows[0], spec)
        print(f"[dry-run] {spec['benchmark-id']}: dataset schema OK")
        return

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("missing dependency: pyyaml") from exc
    config = yaml.safe_load(config_path.read_text()) or {}
    general = config.get("general") or {}
    output_dir = Path(general["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model = general.get("model") or "kimi-k2.7-code"
    custom = config.get("custom") or {}
    sample_indices = parse_sample_indices()
    load_limit = max(sample_indices) + 1 if sample_indices else limit
    rows = _load_rows(spec, load_limit)
    if sample_indices:
        missing = [index for index in sample_indices if index >= len(rows)]
        if missing:
            raise IndexError(f"dataset did not return requested indices: {missing}")
        indexed_rows = [(index, rows[index]) for index in sample_indices]
    else:
        indexed_rows = list(enumerate(rows))
    scorer = SCORERS[spec["scorer"]]

    def evaluate(index_row):
        index, row = index_row
        completion = chat_detailed(
            [{"role": "user", "content": _prompt(row, spec)}],
            model=model,
            max_tokens=int(custom.get("max_tokens", 2048)),
            temperature=float(custom.get("temperature", 1.0)),
            timeout=int(custom.get("timeout", 180)),
        )
        targets = _targets(row, spec)
        # A successful provider response is a completed benchmark attempt even
        # when the model emits no final ``content`` (for example, it exhausts
        # the token budget inside ``reasoning_content``).  That is a model
        # failure and must score zero, not invalidate the whole evaluation as
        # an infrastructure failure.
        answer_present = bool(completion.content.strip())
        if answer_present:
            model_failure_reason = None
            correct = scorer(completion.content, targets)
        else:
            if completion.finish_reason == "length":
                model_failure_reason = "length_without_answer"
            elif completion.reasoning_content.strip():
                model_failure_reason = "reasoning_only"
            else:
                model_failure_reason = "empty_content"
            correct = False
        return {
            "index": index,
            "response": completion.content,
            "reasoning_content": completion.reasoning_content,
            "finish_reason": completion.finish_reason,
            "usage": completion.usage,
            "request_complete": True,
            "complete": True,
            "answer_present": answer_present,
            "model_failure_reason": model_failure_reason,
            "incomplete_reason": None,
            "targets": targets,
            "correct": correct,
            "provider": completion.provider,
        }

    started = time.time()
    results = []
    workers = min(int(custom.get("max_workers", 8)), max(len(indexed_rows), 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(evaluate, pair): pair for pair in indexed_rows}
        for future in as_completed(futures):
            index, row = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append({
                    "index": index,
                    "response": "",
                    "reasoning_content": "",
                    "finish_reason": None,
                    "usage": {},
                    "request_complete": False,
                    "complete": False,
                    "answer_present": False,
                    "model_failure_reason": None,
                    "incomplete_reason": "request_error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "targets": _targets(row, spec),
                    "correct": None,
                    "provider": None,
                })
    results.sort(key=lambda item: item["index"])
    completed = sum(item["complete"] for item in results)
    incomplete = len(results) - completed
    correct = sum(item["correct"] is True for item in results)
    scoring_supported = spec["scorer"] != "unscored"
    with (output_dir / "samples.jsonl").open("w", encoding="utf-8") as sink:
        for result in results:
            sink.write(json.dumps(result, ensure_ascii=False) + "\n")
    observed_reward = correct / len(results) if results and scoring_supported else None
    reward = observed_reward if incomplete == 0 and scoring_supported else None
    (output_dir / "reward.json").write_text(json.dumps({
        "reward": reward,
        "meta_data": {"benchmark": spec["benchmark-id"], "model": model,
                      "metric": spec["scorer"], "required_metric": spec.get("required-metric"),
                      "scoring_supported": scoring_supported, "attempted": len(results),
                      "completed": completed, "incomplete": incomplete, "correct": correct,
                      "observed_reward": observed_reward,
                      "elapsed_s": round(time.time() - started, 3)},
    }, ensure_ascii=False, indent=2) + "\n")
    if incomplete:
        raise RuntimeError(
            f"evaluation incomplete: {incomplete}/{len(results)} samples; "
            "artifacts were written but the run is not successful"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--validate-data", action="store_true")
    args = parser.parse_args()
    run(args.benchmark_dir, args.config, args.limit, validate_data=args.validate_data)


if __name__ == "__main__":
    main()
