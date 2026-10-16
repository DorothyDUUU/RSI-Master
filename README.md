# RSI-Master

Official code for [RSI-Master: Structuring Experiments to Guide Autonomous Model Improvement](https://arxiv.org/abs/2609.35561).

RSI-Master includes Experiment OS, the `cc_rollout` playground, and a unified evaluation framework with 24 benchmark integrations.

An agent framework for self-evolving research workflows. A **playground** seeds a private
workspace for an agent, launches it, and records what it did; the agent's job is to run a
full research loop inside that workspace — read the benchmark spec, build training data,
submit training, evaluate, and report.

The framework layer (`evomaster/`) is playground-agnostic. This repository ships one
playground, **`cc_rollout`**, which launches a single `claude -p` subprocess and lets it
roll out end to end without branching.

## Layout

| Path | What it is |
|---|---|
| `evomaster/` | Framework core: playground registry, agent loop, sessions, tools, skills |
| `run.py` | Entry point — `python run.py --agent <name> --config <yaml> --task <text>` |
| `playground/cc_rollout/` | The `cc_rollout` playground (seeding, runner, orchestrator, prompts) |
| `configs/cc_rollout/` | Config templates (`config.yaml` = naive, `config_cc_with_expos.yaml`) |
| `backend/` | GPU-side plumbing: `gpu_backend` HTTP client, `server`, `submit_pipeline`, eval wrappers |
| `bmk_everything/` | The benchmark suite — `framework/` (unified eval entry point). 24 supported benchmark integrations; datasets are prepared separately |
| `tools/ExpOS/` | `expos` — the experiment-state / submit-pipeline CLI used by `cc_with_expos` |
| `tools/cc_rollout_expos/` | Thin wrapper that binds `expos` to the submit pipeline |
| `tools/bash_security/`, `tools/bash_backend/` | Sandbox filtering + the bash execution backend |
| `public/` | **Not in the repo** — built locally; see below |

## Install

Requires Python 3.12.3 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                 # creates .venv and installs dependencies
uv sync --extra search  # optional: the standalone search CLI tools
```

Verify the framework imports and the playground registers:

```bash
uv run python -c "import playground.cc_rollout.core; \
  from evomaster.core import list_registered_playgrounds; \
  print(list_registered_playgrounds())"
# -> ['cc_rollout']
```

## Building `public/`

`public/` is deliberately **not** checked in — it holds multi-hundred-gigabyte model
weights and pre-built environments. The code resolves everything under it by path, so you
only need to populate the pieces you actually use:

```
public/
├── models/              # HF-format base models, e.g. models/Qwen/Qwen3-4B-Base
├── datasets/            # eval datasets referenced by the eval wrappers
├── vllm_env_0819/       # serving env (vLLM + transformers 5.x), for stage-2 serve
├── llama_factory_env/   # training env, for the llama_factory SFT path
└── distill_env/         # optional: distillation teacher env
```

Each environment is a normal Python venv/conda env built from the corresponding source
tree — `backend/backend/llama_factory/` contains the vendored training framework,
`backend/backend/vllm/` the serving docs. Training stages activate them by absolute path
(`<repo>/public/.../bin/activate`), so build them in place rather than symlinking.

### Benchmark suite (`bmk_everything/`)

BmkEverything provides a unified evaluation entrypoint for the benchmarks below.
Install the framework, prepare the dataset and any external evaluator, then select
an integration with its benchmark ID. See the [setup guide](bmk_everything/SETUP.md)
for environment, path, endpoint and dataset configuration.

[`benchmarks.json`](bmk_everything/benchmarks.json) lists the supported IDs;
[`datasets.json`](bmk_everything/datasets.json) records data sources, revisions and
available preparation commands. Data, model weights and evaluator environments
are prepared separately. Validation results are recorded in
[the verification notes](docs/OPEN_SOURCE_VALIDATION.md).

#### Supported benchmarks

| Benchmark | ID | Domain | Evaluation | Additional setup |
|---|---|---|---|---|
| [AIME 2025](bmk_everything/format/aime2025/BENCHMARK.md) | `aime2025` | Mathematics | Numeric exact match | Local JSONL data |
| [AIME 2026](bmk_everything/format/aime2026/BENCHMARK.md) | `aime2026` | Mathematics | Numeric exact match | Local JSONL data |
| [Arena-Hard Writing](bmk_everything/format/arenahardwriting/BENCHMARK.md) | `arenahardwriting` | Writing | LLM-judged preference score | External evaluator and judge API |
| [BFCL](bmk_everything/format/bfcl/BENCHMARK.md) | `bfcl` | Tool use | Function-calling evaluation | BFCL evaluator and datasets |
| [CKQA](bmk_everything/format/ckqa/BENCHMARK.md) | `ckqa` | Finance | Multiple-choice accuracy | DataPrep dataset bootstrap |
| [CMPhysBench](bmk_everything/format/cmphysbench/BENCHMARK.md) | `cmphysbench` | Physics | Official SEED score | External evaluator and scorer environment |
| [CTI-MCQ](bmk_everything/format/cti-mcq/BENCHMARK.md) | `cti-mcq` | Cybersecurity | Multiple-choice accuracy | Local JSONL data |
| [Earth Silver](bmk_everything/format/earthse-earth-silver/BENCHMARK.md) | `earthse-earth-silver` | Earth science | Multiple-choice accuracy | Local JSONL data |
| [EconLogicQA](bmk_everything/format/econlogicqa/BENCHMARK.md) | `econlogicqa` | Economics | Event-order sequence exact match | Local JSONL data |
| [GPQA](bmk_everything/format/gpqa/BENCHMARK.md) | `gpqa` | Science | Multiple-choice accuracy | Dataset access approval and local data |
| [GSM8K](bmk_everything/format/gsm8k/BENCHMARK.md) | `gsm8k` | Mathematics | Numeric-answer accuracy | Local Hugging Face dataset |
| [HealthBench Easy](bmk_everything/format/healthbench_easy/BENCHMARK.md) | `healthbench_easy` | Medicine | Rubric-based score | Prepared Easy subset and judge API |
| [HealthBench Professional](bmk_everything/format/healthbench_professional/BENCHMARK.md) | `healthbench_professional` | Medicine | Rubric-based score | Dataset and judge API |
| [HLE](bmk_everything/format/hle/BENCHMARK.md) | `hle` | General knowledge | Answer grading | Dataset and judge API |
| [HorizonMath](bmk_everything/format/horizonmath/BENCHMARK.md) | `horizonmath` | Research mathematics | Official solution validation | External checkout and validation dependencies |
| [HumanEval](bmk_everything/format/humaneval/BENCHMARK.md) | `humaneval` | Coding | Execution-based pass@1 | Dataset and isolated code-execution environment |
| [LegalBench](bmk_everything/format/legalbench/BENCHMARK.md) | `legalbench` | Law | Weighted balanced accuracy across five tasks | DataPrep dataset bootstrap and judge API |
| [LEXam](bmk_everything/format/lexam/BENCHMARK.md) | `lexam` | Law | 16-option multiple-choice accuracy | Local JSONL data |
| [LiveCodeBench](bmk_everything/format/livecodebench/BENCHMARK.md) | `livecodebench` | Coding | Execution-based pass@1 | Code Generation Lite v6 and execution environment |
| [MedCaseReasoning](bmk_everything/format/medcasereasoning/BENCHMARK.md) | `medcasereasoning` | Medicine | Diagnostic accuracy; reasoning recall | DataPrep dataset bootstrap and judge API |
| [MedXpertQA](bmk_everything/format/medxpertqa/BENCHMARK.md) | `medxpertqa` | Medicine | Five-option accuracy; MM-data text-only variant | Local MM-config JSONL data; images are not sent |
| [Pre-Flight](bmk_everything/format/pre-flight/BENCHMARK.md) | `pre-flight` | Aviation | Multiple-choice accuracy | Local JSONL data |
| [SciCode-Verified](bmk_everything/format/scicode-verified/BENCHMARK.md) | `scicode-verified` | Scientific coding | Official execution-based scoring | External SciCode-Verified checkout and grading environments |
| [WMDP-Bio](bmk_everything/format/wmdp-bio/BENCHMARK.md) | `wmdp-bio` | Biology | Multiple-choice accuracy | Local JSONL data |

### The slime training environment

The `slime_sft` path in `backend/backend/submit_pipeline` runs
[slime](https://github.com/THUDM/slime) (Megatron-LM training + SGLang rollout). None of
that stack is vendored — it is a ~25 GB conda environment plus three source checkouts, so
it is built locally:

```bash
export WORKSPACE_DIR="$HOME/slime-workspace"   # holds the checkouts; needs disk
export CONDA_DIR="$HOME/miniconda3"
git clone https://github.com/THUDM/slime.git "$WORKSPACE_DIR/slime"

bash backend/backend/slime/docs/env_construct/build_slime_env.sh        # step 1
bash backend/backend/slime/docs/env_construct/build_slime_env_step2.sh  # step 2
bash backend/backend/slime/docs/env_construct/verify_gpu_env.sh         # on the GPU box
```

Exact pins (torch 2.11.0+cu129, SGLang `0b3bb0cbe`, Megatron-LM `1dcf0dafa`, TE 2.16.1,
apex `10417ace`, flash-attn 2.8.3) are in
[that directory's README](backend/backend/slime/docs/env_construct/README.md). Export
`SLIME_ENV` / `SLIME_MEGATRON_LM` / `SLIME_CONVERT_TO_HF` afterwards — a rendered
`pipeline.sh` aborts with a pointer to those docs if they are unset, rather than silently
using the wrong interpreter.

The `llama_factory_sft` path uses the vendored `backend/backend/llama_factory/` instead
and needs only the `public/llama_factory_env` built above.

## Running the playground

Two modes, selected by `mode:` in the config:

| mode | Workspace gets | How the agent trains |
|---|---|---|
| `cc_naive` | backend snapshots + `README.md` / `BENCHMARK.md` / `GLOBAL_MEMORY.md` | writes its own scripts, drives them with `public/gpu-backend` |
| `cc_with_expos` | the above **plus** `experiment_os/`, `SUBMIT_GUIDE.md` and an `./expos` launcher | `./expos worker submit` → the serverless submit pipeline |

```bash
# cc_naive (default)
uv run python run.py --agent cc_rollout \
  --config configs/cc_rollout/config.yaml \
  --task "Prepare a post-training attempt for aime2025 with clean final answers." \
  --set run.single_agent_running_time_seconds=3600

# cc_with_expos
uv run python run.py --agent cc_rollout \
  --config configs/cc_rollout/config_cc_with_expos.yaml \
  --task "..." --set run.single_agent_running_time_seconds=3600
```

The rollout CLI can also be driven directly:

```bash
uv run python -m playground.cc_rollout.core.rollout run    --config configs/cc_rollout/config.yaml
uv run python -m playground.cc_rollout.core.rollout health  # check the GPU backend
```

### Run directory layout

```
runs/<run_id>_<timestamp>/
├── workspace/            # the only thing the agent sees (its cwd)
│   ├── README.md         #   protocol — passed as --append-system-prompt
│   ├── BENCHMARK.md      #   eval spec
│   ├── GLOBAL_MEMORY.md  #   cross-context scratchpad
│   ├── public/           #   backend module snapshots + gpu-backend launcher
│   ├── FINISH            #   the agent writes here to end the run
│   └── experiment_os/ SUBMIT_GUIDE.md expos    # (cc_with_expos only)
├── configs/              # invisible to the agent (CLAUDE_CONFIG_DIR, settings.json, run.json)
├── logs/agent.log
└── summary.md
```

## Prerequisites to actually run

Running an agent end to end needs more than the Python environment:

- **`claude` on `PATH`** — the runner launches `claude -p`. Override with `claude.bin`.
- **A model endpoint** — either the claude CLI's own default, or an
  Anthropic-Messages-compatible router via `claude.base_url` + `claude.auth_token`.
- **A GPU backend** — `backend/server/` (FastAPI + `tmux`) on the GPU host, with
  `backend/gpu_backend/` as the client. Defaults to `localhost:8989`.
- **A benchmark suite checkout with `format/`** — see `bmk_everything/` above.
- For `cc_with_expos`, the submit pipeline additionally needs the training/serving envs
  under `public/` and the eval wrappers under `backend/backend/eval_*`.

Without these, seeding still succeeds — every missing source degrades to a warning and a
stub, so you can exercise the seeding path on a bare checkout.

## Configuration

Paths live in `configs/cc_rollout/*.yaml`; environment variables override the rest.
Notable ones:

| Variable | Purpose |
|---|---|
| `RSI_MASTER_ROOT` | Repo root, when it is not where the file layout implies |
| `RSI_MASTER_RUN_ROOT` | Run output directory (default `<repo>/runs`) |
| `BENCHMARK_ROOT` | Benchmark suite checkout (default `<repo>/bmk_everything`) |
| `CLAUDE_CONFIG_DIR` | Set by the runner — keeps the transcript out of the workspace |
| `CC_ROLLOUT_AUTH_TOKEN` | Overrides `claude.auth_token` so no key is committed |
| `SLIME_ENV`, `SLIME_MEGATRON_LM`, `SLIME_CONVERT_TO_HF`, `SLIME_MODEL_SPEC_PREFIX`, `SLIME_WORKSPACE` | slime stack (built separately — see above); unset ⇒ the slime stage aborts with a build-docs pointer |
| `BASH_BACKEND_URL`, `BASH_BACKEND_WHITELIST` | Bash backend server + extra allowlisted dirs |

`claude.auth_token` ships as the placeholder `sk-123`. It **must be non-empty** — the
claude CLI treats an empty token as "Not logged in" and exits immediately.

## Security model

The runner invokes `claude` with `--dangerously-skip-permissions` (for headless runs).
Under that flag only **deny** rules still bind, so `playground/cc_rollout/core/rollout/permissions.py`
generates a per-run `settings.json` that denies upward traversal, the agent's own
invisible `configs/` tree, and absolute reads/writes of system roots. That is defense in
depth; the real boundary is the process one — the agent's cwd is its private workspace and
anything outside must be explicitly allowlisted via `run.public_dirs`. A determined agent
can still escape string-based rules via subprocesses, so run untrusted work in a
container or as a non-root user.

## Vendored third-party code

`backend/backend/llama_factory/` contains a vendored copy of
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) `0.9.6.dev0` (Apache-2.0 — see
`backend/backend/llama_factory/llama_factory/LICENSE`), with local additions under
`llama_factory/scripts/`. `backend/backend/vllm/` contains upstream vLLM documentation
only. Neither is modified relative to upstream except for the local scripts noted above.

## Development

```bash
uv run python -m pytest -q      # framework + tools test suites
```

## Citation

```bibtex
@misc{du2026rsimaster,
  title={RSI-Master: Structuring Experiments to Guide Autonomous Model Improvement},
  author={Yaxin Du and Xiyuan Yang and Zhifan Zhou and Yujie Ge and Cheng Wang and Jiajun Wang and Sijie Chen and Zehui Liu and Yuxin Zhang and Weicheng Gu and Julian Zhang and Zixing Lei and Siheng Chen},
  year={2026},
  eprint={2609.35561},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2609.35561}
}
```
