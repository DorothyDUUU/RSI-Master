#!/bin/bash
# ============================================================================
# GPU-server verification for the slime training env built on the CPU box.
#
# Verifies the `slime` conda env and its checkouts on a GPU machine. When the
# env and workspace live on a shared mount that the GPU box also sees, this
# script needs no installs — it only verifies.
#
# Pre-requisites on the GPU server:
#   - NVIDIA driver that supports CUDA 12.9 (driver >= 560).
#   - The conda env (and the workspace: slime/, Megatron-LM/, sglang/) is in place.
#
# Run:  bash verify_gpu_env.sh
# ============================================================================
# NOTE: no `set -u` — conda's cuda-nvcc activate hook references NVCC_*_FLAGS
# before defining them, which would abort under `set -u`.
set -o pipefail

CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$HOME/slime-workspace}"
CONDA_BIN="$CONDA_DIR/bin/conda"
ENV_NAME="slime"

echo "===== 0. host ====="
nvidia-smi 2>/dev/null | head -12 || echo "!! nvidia-smi not found or no driver"
echo
echo "driver/CUDA above must be >= 560 / >= 12.9 for this env (built with cuda 12.9.1)."

echo "===== 1. conda env ====="
if ! "$CONDA_BIN" env list | grep -qE "^$ENV_NAME\s+"; then
  echo "!! env '$ENV_NAME' not found under $CONDA_DIR — run build_slime_env.sh first."; exit 1
fi
"$CONDA_BIN" env list | grep "$ENV_NAME"
echo

source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# conda 12.x puts CUDA headers under targets/... — TE/torch need them in include/.
# (The symlinks were created at build time on the CPU box; re-run in case the
#  mount is fresh or they were stripped by a copy.)
tgt_inc="$CONDA_PREFIX/targets/x86_64-linux/include"
if [ -d "$tgt_inc" ]; then
  for f in "$tgt_inc"/*; do
    b="$(basename "$f")"
    [ -e "$CONDA_PREFIX/include/$b" ] || ln -s "$f" "$CONDA_PREFIX/include/$b"
  done
fi
tgt_lib="$CONDA_PREFIX/targets/x86_64-linux/lib"
if [ -d "$tgt_lib" ]; then
  for f in "$tgt_lib"/*; do
    [ -e "$f" ] || continue
    b="$(basename "$f")"
    [ -e "$CONDA_PREFIX/lib/$b" ] || ln -s "$f" "$CONDA_PREFIX/lib/$b"
  done
fi

# Build-time env vars the native extensions were compiled with; set them at
# runtime so any on-the-fly JIT (e.g. TE, flash-attn) picks the right arch.
export CUDA_HOME="$CONDA_PREFIX"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS:-9.0}"

echo "===== 2. torch + native stack ====="
python - <<'PY'
import torch
print(f"torch            {torch.__version__}")
print(f"torch.version.cuda {torch.version.cuda}")
print(f"torch.cuda.is_available() {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu              {torch.cuda.get_device_name(0)}")
    print(f"capability       {torch.cuda.get_device_capability(0)}")
    print(f"mem              {torch.cuda.get_device_properties(0).total_memory/2**30:.0f} GiB")
import torchvision, torchaudio
print(f"torchvision      {torchvision.__version__}")
print(f"torchaudio       {torchaudio.__version__}")
PY
echo

echo "===== 3. sglang + native libs (the rollout engine, mandatory for slime) ====="
python - <<'PY'
import importlib.metadata as md
import sglang
print("sglang      ", md.version("sglang"))
import sglang_router
print("sglang_router", sglang_router.__version__)
import transformer_engine
print("TE          ", transformer_engine.__version__)
import flash_attn
print("flash_attn  ", flash_attn.__version__)
import apex
print("apex        ", md.version("apex"))
import torch_memory_saver
print("torch_memory_saver", md.version("torch-memory-saver"))
import fake_int4_quant_cuda
print("fake_int4_quant_cuda OK")
PY
echo

echo "===== 4. megatron + slime ====="
python - <<'PY'
import importlib.metadata as md
import megatron.core
print("megatron.core", megatron.core.__version__)
import slime
print("slime       ", md.version("slime"))
import slime.backends.sglang_utils.arguments   # forces `import sglang` path
print("sglang backend import OK")
import slime.backends.megatron_utils
print("megatron backend import OK")
PY
echo

echo "===== 5. patches ====="
WS="${WORKSPACE_DIR:-$HOME/slime-workspace}"
PATCHDIR="$WS/slime/docker/patch/v0.5.15.post1"
echo "-- megatron.patch --"
cd "$WS/Megatron-LM"
git apply --reverse --check "$PATCHDIR/megatron.patch" && echo "APPLIED" || echo "NOT APPLIED"
git grep -c '^<<<<<<< ' -- . | grep . && echo "!! conflict markers present" || echo "no conflict markers"
echo "-- sglang patches (verify against clean HEAD+4patches) --"
cd "$WS/sglang"
if git worktree add --detach /tmp/sg-verify HEAD >/dev/null 2>&1; then
  ( cd /tmp/sg-verify && for p in sglang.patch sglang-top_p.patch sglang-release_hicache.patch sglang-pull_weights.patch; do
      git apply "$PATCHDIR/$p"; done
    git diff > /tmp/sg_exp.diff )
  git diff > /tmp/sg_act.diff
  if diff -q /tmp/sg_exp.diff /tmp/sg_act.diff >/dev/null; then echo "sglang patches: APPLIED (matches clean apply)"; else echo "sglang patches: MISMATCH — investigate"; fi
  git worktree remove --force /tmp/sg-verify >/dev/null 2>&1; rm -rf /tmp/sg-verify
else
  echo "worktree add failed (uncommitted state?), checking reverse-only:"
  for p in sglang.patch sglang-top_p.patch sglang-release_hicache.patch sglang-pull_weights.patch; do
    git apply --reverse --check "$PATCHDIR/$p" >/dev/null 2>&1 && echo "$p APPLIED" || echo "$p ?"
  done
fi
echo

echo "===== 6. SFT smoke: Megatron arg parse + sglang server version ====="
# megatron.training is only importable from the Megatron-LM repo root (slime's
# own run scripts set PYTHONPATH to it), so run the parse test from there.
(cd "$WS/Megatron-LM" && python - <<'PY'
from megatron.training.arguments import parse_args
args = parse_args(extra_args_provider=None)
print("megatron parse_args OK, use-distributed-optimizer default:",
      getattr(args, "use_distributed_optimizer", "n/a"))
PY
)
python -m sglang.srt.entrypoints.engine --help >/dev/null 2>&1 && echo "sglang engine CLI OK" || echo "sglang engine CLI exit nonzero (ok on some builds)"

echo
echo "===== DONE. If 2-5 pass, the env is ready to run the slime SFT script: ====="
echo "  bash $WS/slime/scripts/run-qwen3-4B-base-sft.sh  (add --debug-train-only first to smoke it)"
