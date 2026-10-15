# RSI-Master

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
| `bmk_everything/` | The benchmark suite — `framework/` (unified eval entry point). The `format/` benchmark definitions and dataset tree are **not** in the repo; see below |
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

### The benchmark suite (`bmk_everything/`)

`bmk_everything/framework/` is the unified evaluation entry point (`run_eval.py` plus
`core/` and `automation/`). It resolves benchmark definitions relative to itself:

```
bmk_everything/
├── framework/           # in the repo: run_eval.py, core/, automation/, tests/
└── format/              # NOT in the repo — per-benchmark specs and runners
    └── <bench_id>/      #   BENCHMARK.md, config.json, run_benchmark.py
```

`format/` and the dataset tree are not vendored. Point `$BENCHMARK_ROOT` at a full
checkout that has them; the default is the in-repo `bmk_everything/`. Consumers read
`$BENCHMARK_ROOT/format/<bench_id>/BENCHMARK.md` and the datasets under
`$BENCHMARK_ROOT/data/datasets/`. When a benchmark has no `format/<id>/BENCHMARK.md`, the
seed writes a stub instead of failing.

The suite's own tests live in `bmk_everything/framework/tests/` and skip themselves when
`format/` is absent.

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
