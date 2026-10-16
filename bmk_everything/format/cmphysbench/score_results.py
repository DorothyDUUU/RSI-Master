#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def extract_boxed_latex(text: str) -> str:
    if not isinstance(text, str):
        return ""
    start = text.find(r"\boxed{")
    if start < 0:
        return ""
    start += len(r"\boxed{")
    depth = 1
    end = start
    while end < len(text):
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
            if depth == 0:
                return text[start:end].strip()
        end += 1
    return ""


def score_one(args):
    official_root, row = args
    sys.path.insert(0, official_root)
    from SEED.SEED import SEED

    gt_values = row.get("gt_answer") or []
    gt = gt_values[0] if isinstance(gt_values, list) and gt_values else ""
    pred = extract_boxed_latex((row.get("llm_answer") or [""])[0])
    try:
        score, _, _, _ = SEED(gt, pred, row.get("answer_type", "Expression")) if gt and pred else (0, None, None, None)
        score = float(score)
    except Exception as exc:
        return {"index": row.get("index"), "boxed_answer": pred, "score": 0.0, "score_error": str(exc)}
    return {"index": row.get("index"), "boxed_answer": pred, "score": score}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--responses", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.responses).read_text(encoding="utf-8").splitlines() if line.strip()]
    scored = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(score_one, (args.official_root, row)) for row in rows]
        for done, future in enumerate(as_completed(futures), 1):
            scored.append(future.result())
            print(f"scoring {done}/{len(rows)}", flush=True)
    score_map = {row["index"]: row for row in scored}
    for row in rows:
        row.update(score_map.get(row["index"], {"boxed_answer": "", "score": 0.0}))
    rows.sort(key=lambda row: row["index"])
    output_dir = Path(args.output_dir)
    (output_dir / "scored_results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    scores = [float(row.get("score", 0.0)) for row in rows]
    seed_score = sum(scores) / len(scores) if scores else 0.0
    exact = sum(abs(score - 100.0) < 1e-4 for score in scores)
    reward = {
        "reward": seed_score / 100.0,
        "meta_data": {
            "accuracy": seed_score / 100.0,
            "seed_score": seed_score,
            "exact_accuracy": exact / len(scores) if scores else 0.0,
            "correct": exact,
            "scoring_supported": True,
            "primary_metric": "SEED/100",
        },
    }
    (output_dir / "reward.json").write_text(
        json.dumps(reward, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
