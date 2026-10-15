#!/bin/bash
set -ex

# ============================================================================
# Continuation of build_slime_env.sh — resumes after the SGLang stack.
#
# The full run died at `transformer_engine==2.16.1` because the conda CUDA
# toolkit puts its headers under <env>/targets/x86_64-linux/include while TE
# (and torch's cpp_extension) only search <env>/include. The <env>/lib dir
# already contains the CUDA shared libs, so only the headers need to be made
# visible. This is the "CPU box has to build TE from source because the
# v2.16.1 GitHub release ships cu13-only wheels" path.
#
# Steps below mirror build_slime_env.sh from the transformer_engine step on.
# ============================================================================

# The workspace root defaults to $HOME/slime-workspace; override WORKSPACE_DIR to
# put the checkouts elsewhere (they need a lot of disk).
WORKSPACE_DIR="${WORKSPACE_DIR:-$HOME/slime-workspace}"
export SLIME_DIR="${SLIME_DIR:-$WORKSPACE_DIR/slime}"
export BASE_DIR="${BASE_DIR:-$WORKSPACE_DIR}"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
CONDA_BIN="$CONDA_DIR/bin/conda"
ENV_NAME="slime"
MEGATRON_COMMIT="1dcf0dafa884ad52ffb243625717a3471643e087"
TMS_COMMIT="8d30c59ca12a68d9deccbc9c6599076a1218cbc5"
PATCH_VERSION="v0.5.15.post1"

source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
unset PIP_CONSTRAINT

export CUDA_HOME="$CONDA_PREFIX"
export NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS:-9.0}"     # H100/H200 (Hopper)
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

# ---------------------------------------------------------------------------
# 0) Make the conda CUDA toolkit look like a standard layout: headers from
#    <env>/targets/x86_64-linux/include are symlinked into <env>/include.
#    (libs are already present in <env>/lib.)
# ---------------------------------------------------------------------------
echo "=== linking CUDA headers into \$(CONDA_PREFIX)/include ==="
tgt_inc="$CONDA_PREFIX/targets/x86_64-linux/include"
for f in "$tgt_inc"/*; do
  b="$(basename "$f")"
  if [ ! -e "$CONDA_PREFIX/include/$b" ]; then
    ln -s "$f" "$CONDA_PREFIX/include/$b"
  fi
done
# safety: link any libs that live only under targets/ into <env>/lib
tgt_lib="$CONDA_PREFIX/targets/x86_64-linux/lib"
for f in "$tgt_lib"/*; do
  [ -e "$f" ] || continue
  b="$(basename "$f")"
  if [ ! -e "$CONDA_PREFIX/lib/$b" ]; then
    ln -s "$f" "$CONDA_PREFIX/lib/$b"
  fi
done
python -c "import os; p=os.path.join(os.environ['CONDA_PREFIX'],'include','cuda_runtime.h'); assert os.path.exists(p), p; print('CUDA headers now visible at', p)"

# ---------------------------------------------------------------------------
# 1) transformer_engine 2.16.1 from source (no cu12 wheel exists for torch 2.11)
# ---------------------------------------------------------------------------
pip install --no-build-isolation "transformer_engine[pytorch]==2.16.1"

# ---------------------------------------------------------------------------
# 2) apex from the pinned commit (--cpp_ext --cuda_ext)
# ---------------------------------------------------------------------------
NVCC_APPEND_FLAGS="--threads 4" \
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

# ---------------------------------------------------------------------------
# 3) torch_memory_saver (zhuzilin fork)
# ---------------------------------------------------------------------------
TMS_CUDA_MAJOR="${TMS_CUDA_MAJOR:-$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')}"
export TMS_CUDA_MAJOR
pip install -v git+https://github.com/zhuzilin/torch_memory_saver.git@${TMS_COMMIT} \
  --no-cache-dir --force-reinstall --no-build-isolation

# ---------------------------------------------------------------------------
# 4) nvidia-modelopt + sgl-router
# ---------------------------------------------------------------------------
pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation
pip install https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-9daabcd/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl --force-reinstall
python -c "import sglang_router; assert 'slime' in sglang_router.__version__"

# ---------------------------------------------------------------------------
# 5) megatron (use the existing clone; pinned commit + editable install)
# ---------------------------------------------------------------------------
pip install "setuptools<80.0.0" pybind11 "packaging>=24.2"
cd "$BASE_DIR/Megatron-LM" && git checkout ${MEGATRON_COMMIT} && pip install -e . --no-build-isolation

# ---------------------------------------------------------------------------
# 6) slime
# ---------------------------------------------------------------------------
cd "$SLIME_DIR"
pip install -r requirements.txt
pip install -e . --no-deps
cd "$SLIME_DIR/slime/backends/megatron_utils/kernels/int4_qat"
pip install . --no-build-isolation

# ---------------------------------------------------------------------------
# 7) final pins
# ---------------------------------------------------------------------------
pip install nvidia-cudnn-cu12==9.16.0.29
pip install "numpy==1.26.4" "scipy==1.17.1"
pip install "kernels<0.15.0"

# ---------------------------------------------------------------------------
# 8) apply patches (same order as the Dockerfile / build_conda.sh)
# ---------------------------------------------------------------------------
patch_dir="$SLIME_DIR/docker/patch/${PATCH_VERSION}"
[ -d "$patch_dir" ] || { echo "Patch dir missing: $patch_dir" >&2; exit 1; }

cd "$BASE_DIR/sglang"
for patch_name in sglang.patch sglang-top_p.patch sglang-release_hicache.patch sglang-pull_weights.patch; do
  patch_path="$patch_dir/${patch_name}"
  [ -f "$patch_path" ] || continue
  if git apply --check "$patch_path"; then
    git apply "$patch_path"
  elif git apply --reverse --check "$patch_path"; then
    echo "$patch_name already applied, skipping"
  else
    echo "$patch_name does not apply cleanly" >&2
    exit 1
  fi
done

cd "$BASE_DIR/Megatron-LM"
megatron_patch="$patch_dir/megatron.patch"
if git apply --reverse --check "$megatron_patch"; then
  echo "megatron.patch already applied, skipping"
else
  git update-index --refresh || true
  git apply "$megatron_patch" --3way || { echo "megatron.patch does not apply cleanly" >&2; exit 1; }
  if git grep -n '^<<<<<<< ' -- .; then
    echo "megatron patch failed to apply cleanly. Please resolve conflicts." >&2
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# 9) smoke test (GPU-dependent parts skipped on a CPU-only box)
# ---------------------------------------------------------------------------
cd "$WORKSPACE_DIR"
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| cuda_available', torch.cuda.is_available())"
python -c "import megatron; print('megatron import OK')"
python -c "import slime; print('slime import OK')"
python -c "import transformer_engine; print('transformer_engine import OK')"
if python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  python - <<'PY'
import sglang
import torch, torchaudio, torchvision
import transformer_engine, flash_attn, apex
print("GPU smoke test PASSED")
PY
else
  echo
  echo "No GPU detected -- CUDA-dependent imports skipped."
  echo "On the GPU server run:"
  echo '  python -c "import sglang, transformer_engine, flash_attn, torch; print(\"native stack ok\")"'
  echo '  python -c "import megatron, slime; print(\"megatron+slime ok\")"'
fi

echo
echo "=== slime env step2 finished ==="
