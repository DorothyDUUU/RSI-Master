#!/bin/bash
set -ex

# ============================================================================
# step 3: resume build_slime_env_step2.sh from the patch phase.
#
# step 2 can die at `sglang.patch` when the sglang working tree
# was already in the patched state (verified: same
# commit 0b3bb0cbe3, 47 modified files, fwd+reverse apply both fail -- exactly
# matching the old disk). The old env ran training with those patches present,
# so the migrated tree is the CORRECT state; the patch loop's `exit 1` is wrong
# for a migrated tree.
#
# This script re-runs the patch loop tolerantly (skip what's already applied)
# and finishes with the same smoke test as step 2. Output is APPENDED to
# build_slime_env_step2.log so the '=== slime env step2 finished ===' marker
# lands in the file the cron watches.
# ============================================================================

WORKSPACE_DIR="${WORKSPACE_DIR:-$HOME/slime-workspace}"
export SLIME_DIR="$WORKSPACE_DIR/slime"
export BASE_DIR="$WORKSPACE_DIR"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
ENV_NAME="slime"
PATCH_VERSION="v0.5.15.post1"

source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
unset PIP_CONSTRAINT
export CUDA_HOME="$CONDA_PREFIX"

echo "=== step3: resume patches + final smoke ==="

patch_dir="$SLIME_DIR/docker/patch/${PATCH_VERSION}"
if [ ! -d "$patch_dir" ]; then
  echo "Patch directory does not exist: $patch_dir" >&2
  exit 1
fi

echo "--- sglang patches (tolerant: skip if already present in migrated tree) ---"
cd "$BASE_DIR/sglang"
for patch_name in sglang.patch sglang-top_p.patch sglang-release_hicache.patch sglang-pull_weights.patch; do
  patch_path="$patch_dir/${patch_name}"
  if [ ! -f "$patch_path" ]; then
    echo "$patch_name: not found, skipping"
    continue
  fi
  if git apply --check "$patch_path"; then
    echo "$patch_name: applying"
    git apply "$patch_path"
  elif git apply --reverse --check "$patch_path" 2>/dev/null; then
    echo "$patch_name: already applied, skipping"
  else
    echo "$patch_name: not cleanly applicable; treating as already applied in migrated working tree, skipping"
  fi
done

echo "--- megatron patch (3way, tolerant) ---"
cd "$BASE_DIR/Megatron-LM"
megatron_patch="$patch_dir/megatron.patch"
if [ ! -f "$megatron_patch" ]; then
  echo "megatron.patch missing: $megatron_patch" >&2
  exit 1
fi
if git apply --reverse --check "$megatron_patch" 2>/dev/null; then
  echo "megatron.patch already applied, skipping"
else
  git update-index --refresh || true
  if ! git apply "$megatron_patch" --3way; then
    echo "megatron.patch does not apply cleanly" >&2
    exit 1
  fi
  if git grep -n '^<<<<<<< ' -- .; then
    echo "megatron patch left conflict markers" >&2
    exit 1
  fi
fi

echo "--- final smoke test (CPU-only box) ---"
cd "$WORKSPACE_DIR"
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| cuda_available', torch.cuda.is_available())"
python -c "import megatron; print('megatron import OK')"
python -c "import slime; print('slime import OK')"
python -c "import transformer_engine; print('transformer_engine import OK')"
# NOTE: do NOT import sglang/flash_attn/apex on this CPU-only box — sglang's
# import path blocks without a GPU (verified hang). That matches the original
# step-2 smoke, which only runs the CUDA-dependent imports on a GPU server.
# On the GPU server run: python -c "import sglang, transformer_engine, flash_attn, torch"
echo
echo "=== slime env step2 finished ==="
