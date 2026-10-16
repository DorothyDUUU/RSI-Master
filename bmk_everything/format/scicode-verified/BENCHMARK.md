# SciCode-Verified v2

SciCode-Verified is the human-verified correction of SciCode. The released v2 test set has 64 main problems and 287 scored subproblems. This integration invokes the official `eval_clean/run_deepseek_eval.py` harness unchanged, including cumulative step prompting, code extraction, per-step execution, skipped-step gold injection, and strict all-subproblems-pass whole-problem scoring.

The primary reward is whole-problem accuracy. `reward.json` also reports subproblem (`step_accuracy`) accuracy. `with_background` and `without_background` are separate splits and must not be averaged together. The default split is `with_background`.

```bash
framework/.venv/bin/python framework/run_eval.py scicode-verified \
  --model /path/to/model --split with_background \
  --openai-api-base http://127.0.0.1:8000/v1 --openai-api-token sk-local
```

Official source commit: `ddab4a92f8d80a7113ab946628e994b52354d838`. The wrapper verifies the official v2 JSONL and 1.1 GB HDF5 target checksums before every run. Raw prompts, responses, extracted code, per-environment step verdicts, and official `results.json` are retained under `official_run/`.
