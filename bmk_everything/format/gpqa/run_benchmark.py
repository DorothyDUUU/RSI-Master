#!/usr/bin/env python3
"""
Unified GPQA evaluation runner.

Evaluates one split of GPQA (Graduate-Level Google-Proof Q&A) against a model
served through an OpenAI-compatible API (e.g. vLLM) and writes the results
into output_dir:

    <output_dir>/reward.json    # contract output: {"reward": accuracy, "meta_data": {...}}
    <output_dir>/samples.jsonl  # per-sample detail

Usage:
    .venv/bin/python run_benchmark.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "framework"))
from core.legacy_bridge import call_completion, completion_fields, parse_sample_indices

BENCHMARK_ID = "gpqa"

# split name -> max samples (-1 = all)
SPLITS = {
    "default": -1,
    "sample_100": 100,
}

GPQA_PROMPT_TEMPLATE = """Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of A,B,C,D. Think step by step before answering.

{question}

A) {choice_a}
B) {choice_b}
C) {choice_c}
D) {choice_d}""".strip()

ANSWER_LETTER_RE = re.compile(
    r"(?:ANSWER|answer|Answer)\s*:\s*([A-Da-d])",
    re.IGNORECASE,
)
STANDALONE_LETTER_RE = re.compile(r"\b([A-Da-d])\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_general(config: dict, key: str, env: str | None = None, default=None):
    """Read a `general` key. Falls back to env var / default when the value is a
    runtime placeholder (`<...>`) that the framework substitutes."""
    val = (config.get("general") or {}).get(key)
    if val is None or (isinstance(val, str) and val.startswith("<") and val.endswith(">")):
        if env and env in os.environ:
            return os.environ[env]
        return default
    return val


def resolve_custom(config: dict, key: str, default=None):
    return (config.get("custom") or {}).get(key, default)


def resolve_output_dir(config: dict) -> Path:
    template = resolve_general(config, "output_dir") or "output"
    out = template.replace("<BENCHMARK_ID>", BENCHMARK_ID)
    if "<TRIAL_NAME>" in out:
        trial = os.environ.get("TRIAL_NAME") or time.strftime("%Y%m%d_%H%M%S")
        out = out.replace("<TRIAL_NAME>", trial)
    path = Path(out)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_dataset(dataset_path: str, seed: int, limit: int) -> list[dict]:
    data_root = Path(dataset_path)
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root
    data_file = data_root / "gpqa_main.jsonl"
    rows = []
    with open(data_file) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    rng = random.Random(seed)
    samples = []
    for row in rows:
        choices = [
            str(row["Correct Answer"]),
            str(row["Incorrect Answer 1"]),
            str(row["Incorrect Answer 2"]),
            str(row["Incorrect Answer 3"]),
        ]
        # Shuffle options so the correct answer is not always "A" (mitigate position bias)
        shuffled = choices.copy()
        rng.shuffle(shuffled)
        correct_letter = chr(ord("A") + shuffled.index(str(row["Correct Answer"])))
        samples.append({
            "question": row["Question"],
            "correct_letter": correct_letter,
            "choices": shuffled,
        })
        if limit >= 0 and len(samples) >= limit:
            break
    return samples


def build_prompt(sample: dict) -> str:
    return GPQA_PROMPT_TEMPLATE.format(
        question=sample["question"],
        choice_a=sample["choices"][0],
        choice_b=sample["choices"][1],
        choice_c=sample["choices"][2],
        choice_d=sample["choices"][3],
    )


# ---------------------------------------------------------------------------
# Extraction & scoring
# ---------------------------------------------------------------------------

def extract_answer(text: str) -> str | None:
    text = (text or "").strip()
    matches = ANSWER_LETTER_RE.findall(text)
    if matches:
        return matches[-1].upper()
    for line in reversed(text.split("\n")):
        line = line.strip()
        if not line:
            continue
        letter_matches = STANDALONE_LETTER_RE.findall(line)
        if letter_matches:
            return letter_matches[-1].upper()
    return None


def score_one(predicted: str | None, target: str) -> bool:
    if predicted is None:
        return False
    return predicted.upper() == target.upper()


# ---------------------------------------------------------------------------
# Model calling (OpenAI-compatible API)
# ---------------------------------------------------------------------------

def call_model(prompt: str, base_url: str, model_name: str, api_key: str,
               max_tokens: int, temperature: float, timeout: int):
    return call_completion(
        prompt, base_url=base_url, model=model_name, api_key=api_key,
        max_tokens=max_tokens, temperature=temperature, timeout=timeout,
    )


def _eval_one(task: dict, base_url: str, model: str, api_key: str,
              max_tokens: int, temperature: float, timeout: int) -> dict[str, Any]:
    completion = call_model(task["prompt"], base_url, model, api_key, max_tokens, temperature, timeout)
    predicted = extract_answer(completion.content)
    return {
        "index": task["global_idx"],
        "question": task["question"],
        "choices": task["choices"],
        "correct_letter": task["correct_letter"],
        "predicted": predicted,
        "correct": score_one(predicted, task["correct_letter"]) if completion.complete else None,
        **completion_fields(completion),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Unified GPQA evaluation")
    parser.add_argument("--config", required=True, help="Path to config/config.yaml")
    parser.add_argument("--split", default=None, help="Override split (default: config general.split)")
    parser.add_argument("--limit", type=int, default=None, help="Override max samples (-1 = all)")
    args = parser.parse_args()

    config = load_config(args.config)

    # --- model & endpoint (framework-injected) ---
    model = resolve_general(config, "model", env="MODEL_NAME")
    if model is None:
        sys.exit("error: model is not set (config general.model or env MODEL_NAME)")
    base_url = resolve_general(config, "openai_api_base", env="OPENAI_API_BASE",
                               default="http://0.0.0.0:8101/v1")
    api_key = resolve_general(config, "openai_api_token", env="OPENAI_API_TOKEN", default="sk-123")

    # --- benchmark knobs ---
    dataset_path = resolve_custom(
        config, "dataset_path",
        default="data/datasets/gpqa")
    shuffle_seed = int(resolve_custom(config, "shuffle_seed", 42))
    max_tokens = int(resolve_custom(config, "max_tokens", 16000))
    temperature = float(resolve_custom(config, "temperature", 0.0))
    max_workers = int(resolve_custom(config, "max_workers", 6))
    timeout = int(resolve_custom(config, "timeout", 600))

    # --- split ---
    split = args.split or resolve_general(config, "split", default="default")
    if split not in SPLITS:
        sys.exit(f"error: unknown split {split!r}; available: {sorted(SPLITS)}")
    split_limit = SPLITS[split]
    limit = args.limit if args.limit is not None else split_limit

    output_dir = resolve_output_dir(config)

    # --- load data ---
    print(f"Loading {BENCHMARK_ID} from {dataset_path} ...")
    sample_indices = parse_sample_indices()
    load_limit = max(sample_indices) + 1 if sample_indices else limit
    test_samples = load_dataset(dataset_path, shuffle_seed, load_limit)
    if sample_indices:
        missing = [index for index in sample_indices if index >= len(test_samples)]
        if missing:
            raise IndexError(f"dataset did not return requested indices: {missing}")
        indexed_samples = [(index, test_samples[index]) for index in sample_indices]
    else:
        indexed_samples = list(enumerate(test_samples))
    total = len(indexed_samples)
    print(f"  Evaluating: {total} samples (split={split})")

    tasks = [
        {
            "idx": i,
            "global_idx": i,
            "question": s["question"],
            "choices": s["choices"],
            "correct_letter": s["correct_letter"],
            "prompt": build_prompt(s),
        }
        for i, s in indexed_samples
    ]

    # --- evaluate ---
    correct = 0
    results_map: dict[int, dict[str, Any]] = {}
    start_time = time.time()
    interrupted = False
    jsonl_path = output_dir / "samples.jsonl"
    with open(jsonl_path, "w") as jsonl_file:
        executor = ThreadPoolExecutor(max_workers=max_workers)
        try:
            future_map = {
                executor.submit(_eval_one, t, base_url, model, api_key,
                                max_tokens, temperature, timeout): t
                for t in tasks
            }
            for future in as_completed(future_map):
                task = future_map[future]
                try:
                    result = future.result()
                except Exception as e:
                    print(f"\n[ERROR] sample {task['global_idx']}: {e}", file=sys.stderr)
                    result = {
                        "index": task["global_idx"],
                        "question": task["question"],
                        "choices": task["choices"],
                        "correct_letter": task["correct_letter"],
                        "predicted": None,
                        "correct": False,
                        "response": "",
                    }
                results_map[task["idx"]] = result
                if result["correct"]:
                    correct += 1
                jsonl_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                jsonl_file.flush()
        except KeyboardInterrupt:
            interrupted = True
            print("\nCtrl+C — saving partial results...", file=sys.stderr)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    attempted = len(results_map)
    done = sum(bool(result.get("complete")) for result in results_map.values())
    incomplete = attempted - done
    elapsed = time.time() - start_time
    accuracy = correct / done if done else None

    # --- contract output: reward.json ---
    reward = {
        "reward": accuracy if not incomplete else None,
        "meta_data": {
            "benchmark_id": BENCHMARK_ID,
            "model": model,
            "split": split,
            "metric": "accuracy",
            "total": total,
            "attempted": attempted,
            "completed": done,
            "incomplete": incomplete,
            "correct": correct,
            "accuracy": accuracy,
            "shuffle_seed": shuffle_seed,
            "elapsed_s": round(elapsed, 2),
            "interrupted": interrupted,
        },
    }
    reward_path = output_dir / "reward.json"
    with open(reward_path, "w") as f:
        json.dump(reward, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"{BENCHMARK_ID} results")
    print(f"  Model:    {model}")
    print(f"  Split:    {split}")
    print(f"  Accuracy: {accuracy if accuracy is not None else 'n/a'}  ({correct}/{done})")
    print(f"  Elapsed:  {elapsed:.1f}s")
    print(f"  reward  -> {reward_path}")
    print(f"  samples -> {jsonl_path}")
    print("=" * 60)

    if interrupted or incomplete:
        sys.exit(1)


if __name__ == "__main__":
    main()
