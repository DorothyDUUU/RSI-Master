"""CPU/OpenAI-compatible adapter for the pinned DataPrep-Bench downstream suite."""
from __future__ import annotations

import argparse
import json
import os
import re
import string
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .legacy_bridge import parse_sample_indices
from .provider import Endpoint, chat_detailed


REPO_ROOT = Path(os.environ.get("BENCHMARK_ROOT") or Path(__file__).resolve().parents[2]).expanduser().resolve()


def normalize(text: object) -> str:
    return str(text).translate(str.maketrans("", "", string.punctuation)).strip().lower()


def normalize_label(text: object) -> str:
    """Canonicalize a classification label without using fuzzy matching."""
    return " ".join(str(text).strip().casefold().split())


def clean_legal(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"^(answer|the answer is|result|prediction)[:\s]+", "", cleaned.strip().lower())
    if cleaned.startswith("yes"):
        return "Yes"
    if cleaned.startswith("no"):
        return "No"
    for label in ("generic", "descriptive", "suggestive", "arbitrary", "fanciful"):
        if label in cleaned:
            return label
    if "ucc" in cleaned:
        return "UCC"
    if "common law" in cleaned:
        return "Common Law"
    return cleaned.capitalize()


def extract_choice(text: str) -> str:
    try:
        value = json.loads(text.strip())
        if isinstance(value, dict):
            answer = str(value.get("answer", "")).strip().upper()
            if answer in "ABCD" and len(answer) == 1:
                return answer
    except (ValueError, TypeError):
        pass
    match = re.search(r"answer\s*[:：]\s*['\"]?([A-D])\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    letters = re.findall(r"\b([A-D])\b", text, re.IGNORECASE)
    return letters[-1].upper() if letters else ""


def extract_tag(text: str, tag: str) -> str:
    match = re.search(fr"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()


def parse_lexglue(text: str, gold: object) -> object:
    if isinstance(gold, list):
        match = re.search(r"\[[\s\S]*?\]", text)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    return sorted({str(item).strip().lower() for item in parsed if str(item).strip()})
            except ValueError:
                pass
        parts = re.split(r"[\n,;]", text)
        return sorted({part.strip().strip("'\"").lower() for part in parts if part.strip()})
    expected = str(gold).strip()
    if expected and expected.lower() in text.lower():
        return expected
    first = text.strip().splitlines()[0] if text.strip() else ""
    return first.strip().strip("'\"")


def balanced_accuracy(rows: list[dict]) -> float:
    by_label: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        by_label[normalize(row["gold"])].append(normalize(row["prediction"]) == normalize(row["gold"]))
    return sum(sum(values) / len(values) for values in by_label.values()) / len(by_label) if by_label else 0.0


def aggregate(benchmark_id: str, rows: list[dict]) -> tuple[float, dict]:
    if benchmark_id == "legalbench":
        tasks: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            tasks[row["task"]].append(row)
        detail = {task: {"balanced_accuracy": balanced_accuracy(values), "total": len(values)}
                  for task, values in sorted(tasks.items())}
        total = sum(item["total"] for item in detail.values())
        reward = sum(item["balanced_accuracy"] * item["total"] for item in detail.values()) / total
        return reward, {"metric": "weighted_balanced_accuracy", "subsets": detail}
    if benchmark_id == "lexglue":
        tasks: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            tasks[row["task"]].append(row)
        detail = {}
        for task, values in sorted(tasks.items()):
            if task == "ledgar":
                score = sum(row["correct"] for row in values) / len(values) if values else 0.0
                detail[task] = {"score": score, "metric": "accuracy", "total": len(values),
                                "skipped": 0}
            else:
                matched = total_gold = 0
                for row in values:
                    gold = set(row["gold"])
                    pred = set(row["prediction"])
                    total_gold += len(gold)
                    matched += len(gold & pred)
                score = matched / total_gold if total_gold else 0.0
                detail[task] = {"score": score, "metric": "multilabel_accuracy",
                                "matched_gold_labels": matched, "total_gold_labels": total_gold}
        reward = sum(item["score"] for item in detail.values()) / len(detail) if detail else 0.0
        return reward, {"metric": "average_of_subsets", "subsets": detail}
    correct = sum(row["correct"] for row in rows)
    reward = correct / len(rows) if rows else 0.0
    meta = {"metric": "diagnostic_accuracy" if benchmark_id == "medcasereasoning" else "accuracy",
            "correct": correct}
    if benchmark_id == "medcasereasoning":
        meta["reasoning_recall"] = sum(row.get("reasoning_recall", 0.0) for row in rows) / len(rows) if rows else 0.0
    return reward, meta


def _judge(messages: list[dict], model: str, custom: dict, max_tokens: int = 8) -> str:
    required = ("JUDGE_OPENAI_API_BASE", "JUDGE_OPENAI_API_TOKEN")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"missing judge environment: {', '.join(missing)}")
    endpoint = Endpoint("openai", os.environ["JUDGE_OPENAI_API_BASE"].rstrip("/"),
                        "JUDGE_OPENAI_API_TOKEN")
    judge_model = os.environ.get("JUDGE_MODEL", model)
    judge_min_tokens = int(custom.get("judge_min_tokens")
                           or os.environ.get("JUDGE_MIN_TOKENS", "0"))
    result = chat_detailed(messages, model=judge_model, endpoint=endpoint,
                           max_tokens=max(max_tokens, judge_min_tokens),
                           temperature=0.0, timeout=int(custom.get("judge_timeout", 180)))
    if not result.complete or result.finish_reason in {"length", "max_tokens"}:
        raise RuntimeError(f"judge returned no complete decision (finish_reason={result.finish_reason})")
    return result.content


def _score(benchmark_id: str, record: dict, response: str, model: str, custom: dict) -> dict:
    metadata = record["metadata"]
    if benchmark_id == "ckqa":
        gold = str(metadata["standard_answer"]).upper()
        pred = extract_choice(response)
        return {"gold": gold, "prediction": pred, "correct": pred == gold}
    if benchmark_id == "lexglue":
        gold_raw = metadata["gold"]
        gold = (sorted({str(item).strip().lower() for item in gold_raw}) if isinstance(gold_raw, list)
                else normalize_label(gold_raw))
        pred = parse_lexglue(response, gold_raw)
        if not isinstance(gold_raw, list):
            pred = normalize_label(pred)
        return {"task": metadata["task_name"], "gold": gold, "prediction": pred, "correct": pred == gold}
    if benchmark_id == "legalbench":
        gold = str(metadata["gold"])
        pred = clean_legal(response)
        calibrated = False
        if pred != gold:
            prompt = ("Return only the correct answer for this legal task.\n"
                      f"Task: {metadata['task_name']}\nQuestion:\n{metadata['prompt_text']}\n"
                      f"Model answer:\n{response}")
            pred = clean_legal(_judge([{"role": "user", "content": prompt}], model, custom, 32))
            calibrated = True
        return {"task": metadata["task_name"], "gold": gold, "prediction": pred,
                "correct": pred == gold, "was_calibrated": calibrated}
    diagnosis = extract_tag(response, "answer")
    reasoning = extract_tag(response, "think")
    true_diagnosis = str(metadata["final_diagnosis"])
    answer = _judge([{"role": "user", "content":
                     f"Is the predicted diagnosis correct (y/n)?\nPredicted diagnosis: {diagnosis}, "
                     f"True diagnosis: {true_diagnosis}\nAnswer [y/n]."}], model, custom)
    correct = "y" in answer.lower()
    points = [part.strip() for part in re.split(r"\d+\.", metadata["diagnostic_reasoning"]) if part.strip()]
    hits = 0
    for point in points:
        answer = _judge([{"role": "user", "content":
                         "Does the model reasoning mention or imply this clinician point? Answer [y/n].\n"
                         f"Clinician point: {point}\nModel reasoning: {reasoning}"}], model, custom)
        hits += "y" in answer.lower()
    return {"gold": true_diagnosis, "prediction": diagnosis, "correct": correct,
            "reasoning_recall": hits / len(points) if points else 0.0, "reasoning_points": len(points)}


def _score_truncated(benchmark_id: str, record: dict) -> dict:
    """Represent a candidate budget exhaustion as an incorrect scored attempt."""
    metadata = record["metadata"]
    if benchmark_id == "legalbench":
        return {"task": metadata["task_name"], "gold": str(metadata["gold"]),
                "prediction": "", "correct": False, "was_calibrated": False}
    if benchmark_id == "lexglue":
        gold_raw = metadata["gold"]
        gold = (sorted({normalize_label(item) for item in gold_raw}) if isinstance(gold_raw, list)
                else normalize_label(gold_raw))
        return {"task": metadata["task_name"], "gold": gold,
                "prediction": [] if isinstance(gold_raw, list) else "", "correct": False}
    if benchmark_id == "ckqa":
        return {"gold": str(metadata["standard_answer"]).upper(), "prediction": "", "correct": False}
    return {"gold": str(metadata["final_diagnosis"]), "prediction": "", "correct": False,
            "reasoning_recall": 0.0, "reasoning_points": 0}


def _load(bench_dir: Path, split: str, limit: int) -> list[tuple[int, dict]]:
    path = REPO_ROOT / "data" / "datasets" / bench_dir.name / "test.jsonl"
    if not path.is_file():
        raise RuntimeError(f"dataset missing: {path}; run python {bench_dir / 'bootstrap.py'}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if split not in {"default", "all"}:
        rows = [row for row in rows if row["metadata"].get("task_name") == split]
    indices = parse_sample_indices()
    if indices:
        missing = [index for index in indices if index >= len(rows)]
        if missing:
            raise IndexError(f"dataset did not return requested indices: {missing}")
        return [(index, rows[index]) for index in indices]
    return list(enumerate(rows[:limit]))


def _score_generated(benchmark_id: str, generated: list[dict], model: str,
                     custom: dict, workers: int) -> tuple[list[dict], int, int]:
    """Score cases concurrently; each case issues one judge request at a time."""
    def score_one(item: dict) -> None:
        if not item["complete"]:
            if item.get("finish_reason") in {"length", "max_tokens"}:
                item["score"] = _score_truncated(benchmark_id, item["record"])
                item["scored_as_incorrect"] = True
            else:
                item["score"] = None
            return
        try:
            item["score"] = _score(benchmark_id, item["record"], item["response"], model, custom)
        except Exception as exc:
            item["score"] = None
            item["judge_error"] = f"{type(exc).__name__}: {exc}"

    if workers == 1:
        for item in generated:
            score_one(item)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # Consume results to propagate unexpected worker errors. Aggregate
            # in input order, regardless of request completion order.
            for _ in pool.map(score_one, generated):
                pass
    scored = [item["score"] for item in generated if item["score"] is not None]
    incomplete = sum(item["score"] is None for item in generated)
    truncated = sum(bool(item.get("scored_as_incorrect")) for item in generated)
    return scored, incomplete, truncated


def run(bench_dir: Path, config_path: Path, limit: int, validate_data: bool = False) -> None:
    import yaml
    spec = json.loads((bench_dir / "spec.json").read_text(encoding="utf-8"))
    for key in ("benchmark-id", "dataset", "official-evaluation-revision", "expected-rows"):
        if key not in spec:
            raise ValueError(f"missing spec key: {key}")
    path = REPO_ROOT / "data" / "datasets" / bench_dir.name / "test.jsonl"
    if validate_data:
        if not path.is_file():
            raise RuntimeError(f"dataset missing: run python {bench_dir / 'bootstrap.py'}")
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(lines) != spec["expected-rows"]:
            raise ValueError(f"expected {spec['expected-rows']} rows, got {len(lines)}")
        row = json.loads(lines[0])
        if not isinstance(row.get("messages"), list) or row.get("metadata", {}).get("source") != spec["source-key"]:
            raise ValueError("dataset schema/source mismatch")
        print(f"[dry-run] {bench_dir.name}: {len(lines)} pinned rows and schema OK")
        return
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    general, custom = config.get("general", {}), config.get("custom", {})
    judge_workers = int(custom.get("judge_workers", 16 if bench_dir.name == "medcasereasoning" else 1))
    if judge_workers < 1:
        raise ValueError("custom.judge_workers must be positive")
    model = general.get("model") or os.environ.get("MODEL_NAME") or "kimi-k2.7-code"
    split = os.environ.get("SPLIT_NAME") or general.get("split") or "all"
    selected = _load(bench_dir, split, limit)
    judge_workers = min(judge_workers, max(len(selected), 1))
    if bench_dir.name in {"legalbench", "medcasereasoning"}:
        missing = [name for name in ("JUDGE_OPENAI_API_BASE", "JUDGE_OPENAI_API_TOKEN") if not os.environ.get(name)]
        if missing:
            raise RuntimeError(f"official protocol requires judge environment: {', '.join(missing)}")

    def generate(pair: tuple[int, dict]) -> dict:
        index, record = pair
        args = record.get("completion_args") or {}
        configured_max_tokens = custom.get("max_tokens")
        result = chat_detailed(record["messages"], model=model,
                               max_tokens=int(configured_max_tokens if configured_max_tokens is not None
                                              else args.get("max_tokens") or 4096),
                               temperature=float(args.get("temperature") or 0.0),
                               timeout=int(custom.get("timeout", 180)))
        truncated = result.finish_reason in {"length", "max_tokens"}
        return {"index": index, "record": record, "response": result.content,
                "reasoning_content": result.reasoning_content, "provider": result.provider,
                "finish_reason": result.finish_reason, "usage": result.usage,
                "complete": result.complete and not truncated,
                **({"error": f"incomplete generation: finish_reason={result.finish_reason}"}
                   if truncated else {})}

    started = time.time()
    generated = []
    workers = min(int(custom.get("max_workers", 8)), max(len(selected), 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(generate, pair): pair for pair in selected}
        for future in as_completed(futures):
            index, record = futures[future]
            try:
                generated.append(future.result())
            except Exception as exc:
                generated.append({"index": index, "record": record, "response": "", "complete": False,
                                  "error": f"{type(exc).__name__}: {exc}", "provider": None,
                                  "reasoning_content": "", "finish_reason": None, "usage": {}})
    generated.sort(key=lambda item: item["index"])
    generation_elapsed = time.time() - started
    print(f"[dataprep] generation finished in {generation_elapsed:.1f}s; "
          f"scoring {len(generated)} cases with judge_workers={judge_workers}", flush=True)
    scoring_started = time.time()
    scored, incomplete, truncated_as_incorrect = _score_generated(
        bench_dir.name, generated, model, custom, judge_workers
    )
    scoring_elapsed = time.time() - scoring_started
    print(f"[dataprep] scoring finished in {scoring_elapsed:.1f}s; incomplete={incomplete}", flush=True)
    output = Path(general["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / "samples.jsonl").open("w", encoding="utf-8") as sink:
        for item in generated:
            item.pop("record", None)
            sink.write(json.dumps(item, ensure_ascii=False) + "\n")
    reward, detail = aggregate(bench_dir.name, scored) if scored else (0.0, {"metric": spec["metric"]})
    payload = {"reward": reward if not incomplete else None,
               "meta_data": {"benchmark": bench_dir.name, "model": model, "split": split,
                             "scoring_supported": True, "attempted": len(generated),
                             "completed": len(scored) - truncated_as_incorrect,
                             "scored": len(scored), "incomplete": incomplete,
                             "truncated_as_incorrect": truncated_as_incorrect,
                             "observed_reward": reward if scored else None, **detail,
                             "generation_elapsed_s": round(generation_elapsed, 3),
                             "scoring_elapsed_s": round(scoring_elapsed, 3),
                             "judge_workers": judge_workers,
                             "elapsed_s": round(time.time() - started, 3)}}
    (output / "reward.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(bench_file: str) -> None:
    bench_dir = Path(bench_file).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-data", action="store_true")
    args = parser.parse_args()
    spec = json.loads((bench_dir / "spec.json").read_text(encoding="utf-8"))
    if args.dry_run:
        print(json.dumps({"benchmark": bench_dir.name, "ready": (REPO_ROOT / "data/datasets" / bench_dir.name / "test.jsonl").is_file(),
                          "expected_rows": spec["expected-rows"], "judge_required": bench_dir.name in {"legalbench", "medcasereasoning"}},
                         ensure_ascii=False))
        if args.validate_data:
            run(bench_dir, args.config, 1, validate_data=True)
        return
    run(bench_dir, args.config, args.limit)
