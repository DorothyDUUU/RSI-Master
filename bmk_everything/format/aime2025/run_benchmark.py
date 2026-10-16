#!/usr/bin/env python3
"""
Unified AIME 2025 evaluation runner.

Evaluates one split of AIME 2025 (American Invitational Mathematics Examination)
against a model served through an OpenAI-compatible API (e.g. vLLM) and writes
the results into output_dir:

    <output_dir>/reward.json    # contract output: {"reward": accuracy, "meta_data": {...}}
    <output_dir>/samples.jsonl  # per-sample detail

Usage:
    .venv/bin/python run_benchmark.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
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

BENCHMARK_ID = "aime2025"

# split name -> max samples (-1 = all)
SPLITS = {
    "default": -1,
    "sample_10": 10,
    # "sample_100": 100,  # 本地 test.jsonl 共 90 题，sample_100 即全量
}

AIME_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.
""".strip()

ANSWER_RE = re.compile(
    r"(?:ANSWER|answer|Answer)\s*:\s*"
    r"([-]?\d+(?:,\d{3})*(?:\.\d+)?)"
)
BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


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

def load_dataset(dataset_path: str, limit: int) -> list[dict]:
    # data/datasets/aime2025/aime_2025.jsonl
    data_root = Path(dataset_path)
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root
    data_file = data_root / "aime_2025.jsonl"
    samples = []
    with open(data_file) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            samples.append({
                "problem": row["problem"],
                "answer": str(row["answer"]),
                "id": str(row.get("id", len(samples))),
            })
            if limit >= 0 and len(samples) >= limit:
                break
    return samples


def build_prompt(problem: str) -> str:
    return AIME_PROMPT_TEMPLATE.format(prompt=problem)


# ---------------------------------------------------------------------------
# Extraction & scoring
# ---------------------------------------------------------------------------

def _clean_number(s: str) -> str:
    s = s.strip().replace(",", "").replace("$", "").replace("%", "")
    return s.rstrip(".。")


def extract_answer(text: str) -> str | None:
    text = (text or "").strip()
    answer_matches = ANSWER_RE.findall(text)
    if answer_matches:
        return _clean_number(answer_matches[-1])
    boxed = BOXED_RE.findall(text)
    if boxed:
        return _clean_number(boxed[-1])
    for line in reversed(text.split("\n")):
        line = line.strip()
        if not line:
            continue
        m = re.search(r"([-]?\d+(?:\.\d+)?)", line.replace(",", ""))
        if m:
            return _clean_number(m.group(1))
    return None


def score_one(predicted: str | None, target: str) -> bool:
    if predicted is None:
        return False
    try:
        pred_num = float(predicted)
        tgt_num = float(target)
        return pred_num == tgt_num
    except (ValueError, TypeError):
        return predicted.strip() == target.strip()


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
        "problem": task["problem"],
        "target": task["answer"],
        "predicted": predicted,
        # 空答(complete=False)也判错: score_one(None, target) -> False,计入分母.
        "correct": score_one(predicted, task["answer"]),
        **completion_fields(completion),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Unified AIME 2025 evaluation")
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
        default="data/datasets/aime2025")
    max_tokens = int(resolve_custom(config, "max_tokens", 16000))
    temperature = float(resolve_custom(config, "temperature", 0.0))
    max_workers = int(resolve_custom(config, "max_workers", 10))
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
    test_samples = load_dataset(dataset_path, load_limit)
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
            "problem": s["problem"],
            "answer": s["answer"],
            "prompt": build_prompt(s["problem"]),
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
                        "problem": task["problem"],
                        "target": task["answer"],
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
    accuracy = correct / attempted if attempted else None  # 空答计入分母(有效分)

    # --- contract output: reward.json ---
    reward = {
        "reward": accuracy,
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
    print(f"  Accuracy: {accuracy if accuracy is not None else 'n/a'}  ({correct}/{attempted})")
    print(f"  Elapsed:  {elapsed:.1f}s")
    print(f"  reward  -> {reward_path}")
    print(f"  samples -> {jsonl_path}")
    print("=" * 60)

    if interrupted or incomplete:
        sys.exit(1)


if __name__ == "__main__":
    main()
