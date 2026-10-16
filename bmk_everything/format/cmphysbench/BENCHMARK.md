# CMPhysBench

CMPhysBench contains 520 graduate-level condensed-matter calculation problems. This integration pins the official repository and Hugging Face dataset, uses the upstream system prompt and pass@1 sampling settings (`temperature=0.6`, `top_p=0.95`, `max_tokens=16384`), extracts the first balanced `\\boxed{...}` answer exactly as the official evaluator does, and calls the official SEED scorer without modifying it.

The primary BmkEverything reward is mean SEED divided by 100, so it lies in `[0, 1]`. `reward.json` also reports the paper's 0–100 `seed_score` and the strict `exact_accuracy` (fraction whose SEED is 100). No LLM judge is used.

```bash
framework/.venv/bin/python framework/run_eval.py cmphysbench \
  --model /path/to/model --split default \
  --openai-api-base http://127.0.0.1:8000/v1 --openai-api-token sk-local
```

Official source commit: `b2cd8571279450f0861759f47d98e9fc577aa993`. Dataset revision: `43d185851f731e23aa5737c3667b3a9e87bf8cd1`. The runner verifies both the repository commit and the local dataset SHA256 before generation.
