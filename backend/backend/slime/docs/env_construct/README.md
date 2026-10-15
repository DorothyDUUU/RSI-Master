# slime training environment (Megatron-LM + slime)

Build scripts for the training environment used by the `slime_sft` path in
`backend/backend/submit_pipeline`. The stack is
[slime](https://github.com/THUDM/slime) (Megatron training + SGLang rollout),
with the same version pins and patch order as upstream's
`slime/build_conda.sh` and `slime/docker/Dockerfile` — no modifications.

The build is split into steps because the official single-script
`build_conda.sh` fails partway on a conda-CUDA headers layout issue (see
[Why three steps](#why-three-steps)). Each step is resumable.

## Prerequisites

- A Linux box with **NVIDIA driver ≥ 560** (CUDA 12.9) and enough disk
  (~25 GB for the env, plus the three checkouts).
- An existing **conda** (miniconda/anaconda) installation. The scripts create a
  `slime` env inside it rather than installing micromamba into `$HOME`.
- **slime cloned** — the scripts install it in place and need its
  `docker/patch/` tree:

  ```bash
  export WORKSPACE_DIR="${WORKSPACE_DIR:-$HOME/slime-workspace}"
  mkdir -p "$WORKSPACE_DIR"
  git clone https://github.com/THUDM/slime.git "$WORKSPACE_DIR/slime"
  ```

`sglang` and `Megatron-LM` are cloned by `build_slime_env.sh` at the pinned
commits if they are not already present.

### Configurable paths

| Variable | Default | Meaning |
|---|---|---|
| `WORKSPACE_DIR` | `$HOME/slime-workspace` | Holds the `slime` / `Megatron-LM` / `sglang` checkouts and all build artifacts |
| `CONDA_DIR` | `$HOME/miniconda3` | conda root that hosts the `slime` env |
| `SLIME_DIR` | `$WORKSPACE_DIR/slime` | slime checkout (must exist before step 1) |
| `TORCH_CUDA_ARCH_LIST` | `9.0` | Target arch (Hopper). Set to your GPU's arch, e.g. `8.0` for A100 |

Export them before each step, or run the scripts from a shell that has them set.

## What gets installed

| Component | Version | Note |
|---|---|---|
| conda env | `slime` | Python 3.12.13 |
| PyTorch | 2.11.0+cu129 | with torchvision 0.26.0+cu129, torchaudio 2.11.0+cu129 |
| CUDA toolkit | 12.9.1 | conda-installed (nccl 2.30.7, cudnn 9.16.0.29) |
| SGLang | 0.5.15.post1 | pinned commit `0b3bb0cbe`, 4 sglang patches applied |
| Megatron-LM | 0.16.0rc0 (megatron-core) | pinned commit `1dcf0dafa`, `megatron.patch` applied |
| slime | 0.3.1 | editable install |
| Transformer Engine | 2.16.1 | built from source (no cu12 wheel) |
| apex | 0.1 | pinned commit `10417ace`, `--cpp_ext --cuda_ext` |
| flash-attn | 2.8.3 | community cu12 / torch2.11 / cxx11abi wheel (SHA256-pinned) |
| torch_memory_saver | 0.0.9.post1+slime | pinned commit `8d30c59` |
| sglang_router | 0.3.2+slime | slime-branch wheel |
| fake_int4_quant_cuda | built | int4 QAT kernels |

## Building

```bash
export WORKSPACE_DIR="$HOME/slime-workspace"   # adjust as needed
export CONDA_DIR="$HOME/miniconda3"
export TORCH_CUDA_ARCH_LIST=9.0                # your GPU arch

bash build_slime_env.sh          # step 1: conda env, CUDA, PyTorch, SGLang stack
bash build_slime_env_step2.sh    # step 2: TE, apex, TMS, Megatron-LM, slime, patches, smoke
# bash build_slime_env_step3.sh  # step 3: only if step 2 died at sglang.patch (see below)
```

All three are idempotent — re-running redoes only what is missing.

## Verifying

On the GPU machine:

```bash
bash verify_gpu_env.sh
```

It checks the driver, the conda env, `torch+cu129`, the
`sglang`/TE/flash-attn/apex/TMS imports, the `megatron+slime` import chain, patch
state, and Megatron argument parsing. All green means the env is ready.

Single-line smoke test:

```bash
conda activate slime
python -c "import torch, megatron, slime, transformer_engine; print('ok', torch.__version__)"
```

## Activating

```bash
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate slime
export CUDA_HOME="$CONDA_PREFIX"
export TORCH_CUDA_ARCH_LIST=9.0    # match the arch you built for
export NVTE_CUDA_ARCHS=9.0
```

Point the RSI-Master side at it — see `backend/backend/slime/README.md` for the
full env contract (`SLIME_ENV`, `SLIME_MEGATRON_LM`, `SLIME_CONVERT_TO_HF`,
`SLIME_MODEL_SPEC_PREFIX`, `WORKSPACE`).

> If `~/.bashrc` sets `PIP_CONSTRAINT=/etc/pip/constraint.txt`, `unset
> PIP_CONSTRAINT` before installing anything — the system tooling pins conflict
> with this env's fixed versions. The build scripts already do this.

## Layout after a build

```
$WORKSPACE_DIR/               # WORKSPACE_DIR
├── docs/env_construct/       #   this directory
├── slime/                    #   THUDM/slime v0.3.1 (RL post-training framework)
├── Megatron-LM/              #   NVIDIA Megatron-LM (train_rl.py / pretrain_gpt.py)
└── sglang/                   #   sgl-project/sglang (rollout engine)

$CONDA_DIR/envs/slime/        # the built conda env
```

## Running SFT (upstream Qwen3-4B example)

```bash
conda activate slime
cd "$WORKSPACE_DIR"
bash slime/scripts/run-qwen3-4B-base-sft.sh   # includes --debug-train-only smoke
```

That upstream script carries hardcoded paths and must be edited before use:

| In the script | Change to |
|---|---|
| `PYTHONPATH: /root/Megatron-LM/` | `$WORKSPACE_DIR/Megatron-LM/` |
| `--hf-checkpoint /root/Qwen3-4B-Base/` | your model path |
| `--ref-load /root/Qwen3-4B-Base_torch_dist` | your reference weights (RL) |
| `--load /root/Qwen3-4B-Base_slime/` | your checkpoint directory |
| `--prompt-data /root/openhermes2_5.parquet` | your training data (parquet, `messages` column) |
| `ray start --num-gpus 8` | your GPU count |
| `--actor-num-gpus-per-node 8` | your GPUs per node |

For this repo's own launcher, use `backend/backend/slime/scripts/launch_sft.sh`
instead — it takes all of the above as env vars and needs no source edits.

### Notes

- **Pure SFT still requires SGLang.** `slime/backends/sglang_utils/arguments.py`
  does `import sglang` at module load; that is slime's design and cannot be
  bypassed, so SGLang must be installed even for offline SFT.
- Megatron's `megatron.training` / `megatron.rl` are importable only from the
  `Megatron-LM/` repo root (slime points at it via `PYTHONPATH`). That is the
  standard usage, not an installation problem.
- RL (GRPO) uses `train_rl.py`. Getting SFT to run end to end is enough to
  validate the stack; RL additionally needs ref/rollout weights.

## Why three steps

Upstream's `build_conda.sh` is a single script, but it reliably dies at
`transformer_engine==2.16.1`: conda's CUDA 12.x toolkit puts headers under
`<env>/targets/x86_64-linux/include` while TE (and torch's `cpp_extension`)
only search `<env>/include`. Split into steps, that failure only needs a re-run
of step 2 instead of rebuilding the three slow layers above it. Steps 2 and 3
share some setup (activation, env vars, header symlinks) so each can be run
standalone.

**`build_slime_env_step3.sh`** exists for one specific failure: if step 2 dies
at `sglang.patch` because the `sglang` working tree was already in the patched
state (a migrated/copied checkout rather than a fresh clone), step 3 resumes
from the patch phase with tolerant patch application. A fresh clone does not
need it.

## Known gotchas

- **No GPU on the build box**: builds and local checks skip the CUDA smoke
  test; `torch.cuda.is_available()` being `False` there is expected. Real GPU
  validation is `verify_gpu_env.sh`.
- **Driver requirement**: NVIDIA driver ≥ 560 on the GPU machine (CUDA 12.9).
- **conda CUDA header layout**: handled automatically by the build scripts and
  `verify_gpu_env.sh` (they symlink `<env>/targets/x86_64-linux/include` into
  `<env>/include`).
- **`fake_int4_quant_cuda`**: `import torch` must come first (the real training
  path does this naturally). Importing it standalone raises an undefined
  symbol error — expected.
- **`kernels` pin**: `kernels<0.15.0` is required, otherwise
  `transformers.integrations.hub_kernels` raises a `ValueError` and
  `import sglang` fails at runtime.
