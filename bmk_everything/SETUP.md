# Benchmark setup

The candidate package contains the 24 paper-result benchmark definitions listed in
[`paper_manifest.json`](paper_manifest.json). Data, external checkouts, graders
and execution environments are not bundled. Integration-specific sources and
preparation commands are recorded in [`paper_downloads.json`](paper_downloads.json)
and each `format/<id>/BENCHMARK.md`. This package is not yet marked as fully
reproducing the paper; see the verification section below.

For an in-repo setup, set `BENCHMARK_ROOT` to the absolute `bmk_everything/` directory.

## Directory and path configuration

```text
<BENCHMARK_ROOT>/
  framework/run_eval.py
  framework/.venv/bin/python
  format/<id>/config.json          # benchmark manifest
  format/<id>/run_benchmark.py
  format/<id>/BENCHMARK.md
  format/<id>/spec.json            # generated integrations; optional for legacy
  format/<id>/config/config.yaml   # runner configuration, when supplied
  data/datasets/<id>/...
  data/output/<id>/<trial>/...
```

Set `BENCHMARK_ROOT` to this root, not its `framework/` directory. Both the unified
entrypoint and generic runner honor it; their default is the in-repository
`bmk_everything/`. The `eval_all` wrapper resolves `--bmk-root` before the environment
variable and passes that selected root to its child process. Set ExpOS
`eval_all.bmk_root` consistently. Paths must exist inside the actual execution
container/GPU host, and `data/output/` must be writable there.

Legacy runners can have their own path rules: check their `dataset_path` configuration.
For the generic runner, relative `data/datasets/...` paths resolve against
`BENCHMARK_ROOT`, independently of the current directory.

## Python environment

From the candidate checkout:

```bash
export BENCHMARK_ROOT=/absolute/path/to/RSI-Master/bmk_everything
uv venv --python 3.12.3 "$BENCHMARK_ROOT/framework/.venv"
uv pip install --python "$BENCHMARK_ROOT/framework/.venv/bin/python" \
  -r /absolute/path/to/RSI-Master/bmk_everything/framework/requirements-core.txt
```

These are the core dependencies, not every benchmark's environment. Install the
selected integration's additional requirements separately. PostTrainBench/Inspect,
code execution, and judge tasks require their own dependencies or services. Automation
scripts often call `framework/.venv/bin/python` directly, so create this environment
even though the eval wrapper can fall back to its current interpreter.

## Datasets

Every released integration must document its upstream URL, immutable revision,
split, download/conversion command, license/access requirements, and expected local
files. Use the selected definition and its source manifest; specialized integrations may require conversions or external assets beyond a raw Hugging Face download.

For generated integrations with `.automated-integration` and `spec.json`, the provided
`framework/automation/download_datasets.py` accepts `--format-dir`, `--data-dir`,
`--hf-cache`, `--state`, and `--ids`. Run it only after obtaining the corresponding
integrations: it downloads data and rewrites their specs. Legacy GSM8K/GPQA/AIME
integrations have a separate data path; `POSTTRAINBENCH_DATASETS` can reference a
pre-existing legacy dataset tree. An empty download job list does not verify setup.

## Validate before using GPUs

```bash
python3 /absolute/path/to/RSI-Master/bmk_everything/framework/check_setup.py \
  --root "$BENCHMARK_ROOT" --bench-id gsm8k
```

This read-only check reports missing entrypoints, environment and benchmark files,
plus data-root availability. It does not validate dataset schema or model requests.
Then run the selected tree's entrypoint:

```bash
cd "$BENCHMARK_ROOT"
framework/.venv/bin/python framework/run_eval.py --list --status runnable
framework/.venv/bin/python framework/run_eval.py gsm8k --dry-run --validate-data
export OPENAI_API_BASE=http://127.0.0.1:18101/v1
export OPENAI_API_TOKEN=local-placeholder
export MODEL_NAME=smoke-model
framework/.venv/bin/python framework/run_eval.py gsm8k --limit 3 --trial-name setup-smoke
```

Replace `gsm8k` with a supplied integration. Explicit endpoint/model settings avoid
the entrypoint's external defaults. Judge tasks additionally need their documented
judge configuration. Check sample outputs and `reward.json`; a dry run or nonempty
benchmark list is not proof of a completed evaluation. For legacy integrations,
`--dry-run --validate-data` may only resolve configuration, so inspect the runner's
own data validation instructions.

The existing framework tests are not collected without their local `format/` tree.
`no tests ran` is not a passing benchmark result. Run the benchmark tests from the
complete checkout and record the selected versions and results.

## Running the framework test suite

Tests need the complete `format/` tree and the core dependencies. Install pytest
in a separate test environment (or the framework environment), then run from the
complete tree's root with the framework import path explicitly set:

```bash
uv pip install --python "$BENCHMARK_ROOT/framework/.venv/bin/python" 'pytest>=8'
cd "$BENCHMARK_ROOT"
PYTHONPATH="$BENCHMARK_ROOT/framework${PYTHONPATH:+:$PYTHONPATH}" \
  framework/.venv/bin/python -m pytest -q framework/tests
```

Passing unit tests does not verify every dataset download or a real model run.
HorizonMath additionally needs its external checkout and data. GPQA, HLE and
BioMysteryBench may require Hugging Face access approval and your own HF login;
a public metadata page does not guarantee anonymous access to the data files.

## Paper-specific DataPrep datasets

From `$BENCHMARK_ROOT`, download the revision-pinned files, verify their SHA256
checksums, and materialize only the selected source records:

```bash
framework/.venv/bin/python framework/automation/bootstrap_dataprep_eval.py --benchmark all
```

This prepares LegalBench (413 rows across five tasks), MedCaseReasoning (897 rows),
and CKQA (311 rows). All three full downloads and checksums were verified on
2026-10-16. LegalBench and MedCaseReasoning require a judge; CKQA is deterministic.
Set `JUDGE_OPENAI_API_BASE`, `JUDGE_OPENAI_API_TOKEN`, and `JUDGE_MODEL` for DataPrep.
For HLE/HealthBench graders set `JUDGE_OPENAI_API_BASE`, `JUDGE_MODEL` and the key
variable named by `custom.grader_api_token_env` (commonly `OPENAI_API_KEY`).
Example-domain endpoints in configurations must be replaced or overridden before
actual grading. Pin the judge model used in the experiment; changing it changes
the protocol. Never put real API keys in committed YAML files.

## Candidate verification (2026-10-16)

The selected 24-task package passed 88 framework tests, with the excluded ARC-AGI
module skipped. All 24 unified dry runs passed using existing local data and
external checkouts. Full local row mapping invariants were checked for the seven
generic domain tasks; LegalBench/CKQA gold-answer round trips scored 1.0. These
checks do not exercise real model inference or judge/code execution for all tasks.

MedXpertQA preserves the original integration's explicitly documented variant:
2,000 MM-config samples evaluated using the text question and five options only.
Images are not sent to the text-only Qwen3 model. Label this as an MM-data text-only
variant; its results are not interchangeable with the full official multimodal
protocol. This choice follows the original BENCHMARK.md; confirm it against the
final experiment records before claiming exact reproduction of the paper numbers.
