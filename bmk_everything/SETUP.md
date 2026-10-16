# Benchmark setup

BmkEverything runs 24 benchmark integrations through a shared entrypoint.
The [supported benchmark table](../README.md#supported-benchmarks) lists each
integration's ID, evaluation method and additional requirements.
[`benchmarks.json`](benchmarks.json) contains the IDs, and
[`datasets.json`](datasets.json) records dataset sources and preparation commands.

## 1. Install the framework

From the RSI-Master repository root:

```bash
export RSI_REPO="$(pwd)"
export BENCHMARK_ROOT="$RSI_REPO/bmk_everything"
uv venv --python 3.12.3 "$BENCHMARK_ROOT/framework/.venv"
uv pip install --python "$BENCHMARK_ROOT/framework/.venv/bin/python" \
  -r "$BENCHMARK_ROOT/framework/requirements-core.txt"
```

This installs the shared runtime. Benchmarks that use an external evaluator,
judge service or code execution also need the dependencies described in their
`format/<id>/BENCHMARK.md`. In particular, install the execution environments for
HumanEval, LiveCodeBench, SciCode-Verified and CMPhysBench before scoring code.

## 2. Configure paths

`BENCHMARK_ROOT` points to the directory containing `framework/`, `format/` and
`data/`, rather than to `framework/` itself:

```text
bmk_everything/
├── benchmarks.json
├── datasets.json
├── framework/
│   ├── run_eval.py
│   └── .venv/bin/python
├── format/<benchmark-id>/
│   ├── config.json
│   ├── config/config.yaml
│   ├── run_benchmark.py
│   └── BENCHMARK.md
├── data/
│   ├── datasets/<benchmark-id>/...
│   └── output/<benchmark-id>/<trial>/...
└── external/                       # evaluator checkouts, where required
```

The unified entrypoint and generic data loader resolve relative dataset paths
against `BENCHMARK_ROOT`. Specialized evaluators can also have fields such as
`official_root`, `scorer_python` or `grading_envs`; configure these according to
the integration guide. HorizonMath accepts `HORIZONMATH_ROOT` for its external
checkout.

For ExpOS, set `bench_id` to a supported ID and `eval_all.bmk_root` to this same
root. The `eval_all` wrapper uses `--bmk-root` before `BENCHMARK_ROOT` and forwards
the selected root to its subprocess. All paths must be visible inside the
machine or container actually running evaluation, and `data/output/` must be
writable.

## 3. Prepare datasets

Datasets are downloaded separately. Find the source, revision, split and local
layout in [`datasets.json`](datasets.json) and the benchmark's
`format/<id>/BENCHMARK.md`. Keep the specified revision and preprocessing steps;
a dataset with a similar name can have different fields, subsets or scoring.
Some upstream datasets, including GPQA, require access approval and your own
Hugging Face login.

### LegalBench, MedCaseReasoning and CKQA

These integrations share a revision-pinned DataPrep-Bench download utility. It
downloads the source files, verifies SHA256 checksums and extracts the records
for each integration:

```bash
cd "$BENCHMARK_ROOT"
framework/.venv/bin/python framework/automation/bootstrap_dataprep_eval.py --benchmark all
```

To prepare just one integration, replace `all` with `legalbench`,
`medcasereasoning` or `ckqa`. The resulting files are:

| ID | Local file | Rows |
|---|---|---:|
| `legalbench` | `data/datasets/legalbench/test.jsonl` | 413 |
| `medcasereasoning` | `data/datasets/medcasereasoning/test.jsonl` | 897 |
| `ckqa` | `data/datasets/ckqa/test.jsonl` | 311 |

### Other integrations

Follow the linked integration guide in the supported benchmark table. Some
integrations consume JSONL; others require a Hugging Face dataset layout,
filtered subset, evaluator checkout or task resources. A raw dataset download
alone does not supply those additional assets.

MedXpertQA uses the **MM-data text-only variant**: it evaluates the text question
and five options from the MM configuration, without sending images. This is
different from the full multimodal evaluation.

## 4. Configure model and judge services

Start an OpenAI-compatible model server, such as vLLM, and set its endpoint and
served model name. For a local server without authentication:

```bash
export OPENAI_API_BASE=http://127.0.0.1:18101/v1
export OPENAI_API_TOKEN=local-placeholder
export MODEL_NAME=smoke-model
```

For authenticated services, supply your own token through the environment.
LegalBench and MedCaseReasoning also need a separate judge configuration:

```bash
export JUDGE_OPENAI_API_BASE=https://your-judge.example.com/v1
export JUDGE_MODEL=your-judge-model
# Set JUDGE_OPENAI_API_TOKEN in your shell or secret manager.
```

HLE and HealthBench graders accept `JUDGE_OPENAI_API_BASE` and `JUDGE_MODEL`;
the credential variable is selected by `custom.grader_api_token_env` in their
configuration, commonly `OPENAI_API_KEY`. Arena-Hard and HorizonMath have
integration-specific judge/validator settings documented in their guides.
Replace example endpoints before evaluating and keep tokens out of committed
configuration files. Pin judge models and parameters when comparing results.

## 5. Check setup and run

List the integrations and check the required files for your selected benchmark:

```bash
cd "$BENCHMARK_ROOT"
framework/.venv/bin/python framework/run_eval.py --list --status runnable
framework/.venv/bin/python framework/check_setup.py --root "$BENCHMARK_ROOT" --bench-id gsm8k
framework/.venv/bin/python framework/run_eval.py gsm8k --dry-run --validate-data
```

The file check reports missing assets. Dry runs resolve configuration; data
validation depends on the runner. Neither calls the model to verify inference.
After preparing the data and model service, run a small evaluation:

```bash
framework/.venv/bin/python framework/run_eval.py gsm8k --limit 3 --trial-name setup-smoke
```

Replace `gsm8k` with any supported ID and use the split declared in its
`config.json`. Results are written under
`data/output/<benchmark-id>/<trial>/`; inspect `reward.json`, sample outputs and
logs to confirm generation and scoring completed.

## 6. Run tests

```bash
uv pip install --python "$BENCHMARK_ROOT/framework/.venv/bin/python" 'pytest>=8'
cd "$BENCHMARK_ROOT"
PYTHONPATH="$BENCHMARK_ROOT/framework${PYTHONPATH:+:$PYTHONPATH}" \
  framework/.venv/bin/python -m pytest -q framework/tests
```

Tests that require optional external assets can skip when those assets are
absent. Integration availability and validation coverage are distinct: see
[verification notes](../docs/OPEN_SOURCE_VALIDATION.md) for completed checks and
remaining end-to-end validation.
