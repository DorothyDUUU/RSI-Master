#!/bin/bash
set -ex

# ============================================================================
# Adapted from slime/build_conda.sh to build the env into an existing conda
# installation instead of installing micromamba into $HOME.
#
# Changes vs. the official script:
#   1. Uses CONDA_DIR (default $HOME/miniconda3) instead of installing
#      micromamba into $HOME. The "slime" env is created inside it.
#   2. WORKSPACE_DIR (default $HOME/slime-workspace) holds the slime /
#      Megatron-LM / sglang checkouts and all built artifacts.
#   3. TORCH_CUDA_ARCH_LIST=9.0 (H100/H200) so CUDA-kernel builds work on a
#      box with no GPU to auto-detect archs from.
#   4. Unsets the host-wide PIP_CONSTRAINT (/etc/pip/constraint.txt) so the
#      training env uses the exact pins in this script instead of the system
#      tooling pins (protobuf==4.24.4, mkl==2021.1.1, ...).
#   5. The final GPU-only smoke test is skipped when no GPU is present.
#
# All package versions and patch steps are identical to slime/build_conda.sh.
# ============================================================================

# The workspace root defaults to $HOME/slime-workspace; override WORKSPACE_DIR to
# put the checkouts elsewhere (they need a lot of disk).
WORKSPACE_DIR="${WORKSPACE_DIR:-$HOME/slime-workspace}"
export SLIME_DIR="${SLIME_DIR:-$WORKSPACE_DIR/slime}"
export BASE_DIR="${BASE_DIR:-$WORKSPACE_DIR}"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
CONDA_BIN="$CONDA_DIR/bin/conda"
ENV_NAME="slime"

echo "WORKSPACE_DIR=$WORKSPACE_DIR"
echo "BASE_DIR=$BASE_DIR"
echo "SLIME_DIR=$SLIME_DIR"
echo "CONDA_BIN=$CONDA_BIN"
echo "ENV_NAME=$ENV_NAME"

[ -x "$CONDA_BIN" ] || { echo "ERROR: conda not found at $CONDA_BIN — set CONDA_DIR" >&2; exit 1; }
# slime must already be checked out: this script installs it in place and needs
# its docker/patch/ tree. sglang and Megatron-LM are cloned below if absent.
[ -d "$SLIME_DIR" ] || {
  echo "ERROR: slime not found at SLIME_DIR=$SLIME_DIR" >&2
  echo "  git clone https://github.com/THUDM/slime.git \"$SLIME_DIR\"" >&2
  exit 1
}
mkdir -p "$BASE_DIR"

# drop the host-wide pip constraint so the env uses the pins in this script
unset PIP_CONSTRAINT

# --- create the env (idempotent) ---
if ! "$CONDA_BIN" env list | grep -qE "^$ENV_NAME\s+"; then
  "$CONDA_BIN" create -n "$ENV_NAME" python=3.12 pip -y -c conda-forge
fi

# shell activation (non-interactive)
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
export CUDA_HOME="$CONDA_PREFIX"
# H100/H200 -> arch 9.0; needed on a box without a GPU to auto-detect archs
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS:-9.0}"

# conda's CUDA 12.x toolkit puts headers under <env>/targets/x86_64-linux/include,
# while TE and torch's cpp_extension only search <env>/include. Symlink the
# headers (and any targets-only libs) into the standard layout so source builds
# (TE, apex, torch_memory_saver) can find them.
tgt_inc="$CONDA_PREFIX/targets/x86_64-linux/include"
if [ -d "$tgt_inc" ]; then
  for f in "$tgt_inc"/*; do
    b="$(basename "$f")"
    if [ ! -e "$CONDA_PREFIX/include/$b" ]; then
      ln -s "$f" "$CONDA_PREFIX/include/$b"
    fi
  done
fi
tgt_lib="$CONDA_PREFIX/targets/x86_64-linux/lib"
if [ -d "$tgt_lib" ]; then
  for f in "$tgt_lib"/*; do
    [ -e "$f" ] || continue
    b="$(basename "$f")"
    if [ ! -e "$CONDA_PREFIX/lib/$b" ]; then
      ln -s "$f" "$CONDA_PREFIX/lib/$b"
    fi
  done
fi

# ----------------------------------------------------------------------------
# Pins, kept in sync with slime/docker/Dockerfile:
#   - SGLANG_IMAGE_TAG (ARG)   -> SGLANG_VERSION
#   - MEGATRON_COMMIT (ARG)    -> MEGATRON_COMMIT
#   - PATCH_VERSION (ARG)      -> PATCH_VERSION
#   - TMS_COMMIT (ARG)         -> TMS_COMMIT
# ----------------------------------------------------------------------------
export SGLANG_VERSION="v0.5.15.post1"
export SGLANG_COMMIT="0b3bb0cbe31873994c9f989fddfe2f87ca839fdd"
export MEGATRON_COMMIT="1dcf0dafa884ad52ffb243625717a3471643e087"
export PATCH_VERSION="v0.5.15.post1"
export TMS_COMMIT="8d30c59ca12a68d9deccbc9c6599076a1218cbc5"

# --- conda: CUDA 12.9 toolkit, nccl, cudnn, rust ---
"$CONDA_BIN" install -n "$ENV_NAME" \
  cuda=12.9.1 \
  cuda-nvtx=12.9.79 \
  cuda-nvtx-dev=12.9.79 \
  nccl \
  -c nvidia/label/cuda-12.9.1 \
  -c nvidia \
  -c conda-forge \
  -y
"$CONDA_BIN" install -n "$ENV_NAME" -c conda-forge cudnn -y
# sglang's editable install builds a Rust extension (sglang-grpc), so the env
# needs a working rustc + cargo.
"$CONDA_BIN" install -n "$ENV_NAME" -c conda-forge rust -y

pip install cuda-python==12.9

# --- sglang (rollout engine) ---
if [ ! -d "$BASE_DIR/sglang" ]; then
  git clone https://github.com/sgl-project/sglang.git "$BASE_DIR/sglang"
fi
cd "$BASE_DIR/sglang"
git checkout ${SGLANG_COMMIT}
pip install -e "python[all]" --extra-index-url https://download.pytorch.org/whl/cu129
pip install --force-reinstall --no-deps \
  torch==2.11.0+cu129 torchvision==0.26.0+cu129 torchaudio==2.11.0+cu129 \
  --index-url https://download.pytorch.org/whl/cu129
pip install --force-reinstall --no-deps \
  sglang-kernel==0.4.4 sgl-deep-gemm==0.1.4 \
  --index-url https://docs.sglang.ai/whl/cu129/
# clean up the cu13 spill that pypi defaults pulled in:
#   1. uninstall the cu13 nvidia-* runtime libs, then
#   2. reinstall the cu12 equivalents to repair the site-packages/nvidia/* dirs.
pip uninstall -y \
  nvidia-cublas \
  nvidia-cuda-cupti \
  nvidia-cuda-nvrtc \
  nvidia-cuda-runtime \
  nvidia-cudnn-cu13 \
  nvidia-cufft \
  nvidia-cufile \
  nvidia-curand \
  nvidia-cusolver \
  nvidia-cusparse \
  nvidia-cusparselt-cu13 \
  nvidia-nccl-cu13 \
  nvidia-nvjitlink \
  nvidia-nvshmem-cu13 \
  nvidia-nvtx \
  nvidia-cutlass-dsl-libs-cu13 \
  || true
pip install --force-reinstall --no-deps \
  nvidia-cublas-cu12 \
  nvidia-cuda-cupti-cu12 \
  nvidia-cuda-nvrtc-cu12 \
  nvidia-cuda-runtime-cu12 \
  nvidia-cudnn-cu12==9.16.0.29 \
  nvidia-cufft-cu12 \
  nvidia-cufile-cu12 \
  nvidia-curand-cu12 \
  nvidia-cusolver-cu12 \
  nvidia-cusparse-cu12 \
  nvidia-cusparselt-cu12 \
  nvidia-nccl-cu12 \
  nvidia-nvjitlink-cu12 \
  nvidia-nvshmem-cu12 \
  nvidia-nvtx-cu12 \
  --index-url https://download.pytorch.org/whl/cu129 \
  --extra-index-url https://pypi.org/simple

pip install cmake ninja

# The SGLang dependency set includes FA4, while the validated TE 2.16 CP stack
# uses FA2 2.7.4 through 2.8.3. This wheel targets Python 3.12, PyTorch 2.11,
# CUDA 12, and the CXX11 ABI used by the conda environment. Upstream does not
# publish that combination, so pin the community build linked from upstream
# issue #2425 by its SHA256 digest.
pip uninstall -y flash-attn-4 flash_attn_4 || true
pip install --no-deps \
  "https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.11/flash_attn-2.8.3%2Bcu12torch2.11cxx11abiTRUE-cp312-cp312-linux_x86_64.whl#sha256=3d0c8e60f820321eedd7166e79c33cb816263d8be6e35c3f5ba8fe2df6fea697"

pip install flash-linear-attention==0.4.2
# FlashQLA: optional GDN backend for Qwen3.5/Qwen3-Next (--qwen-gdn-backend flashqla; requires SM90+)
pip install git+https://github.com/QwenLM/FlashQLA.git --no-build-isolation
# tilelang (matches Dockerfile)
pip install tilelang -f https://tile-ai.github.io/whl/nightly/cu128/

pip install --no-build-isolation "transformer_engine[pytorch]==2.16.1"

NVCC_APPEND_FLAGS="--threads 4" \
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

TMS_CUDA_MAJOR="${TMS_CUDA_MAJOR:-$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')}"
export TMS_CUDA_MAJOR
pip install -v git+https://github.com/zhuzilin/torch_memory_saver.git@${TMS_COMMIT} \
  --no-cache-dir --force-reinstall --no-build-isolation
pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation
pip install https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-9daabcd/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl --force-reinstall
python -c "import sglang_router; assert 'slime' in sglang_router.__version__"

# --- megatron ---
# pre-install Megatron's build deps explicitly since we use --no-build-isolation
pip install "setuptools<80.0.0" pybind11 "packaging>=24.2"
if [ ! -d "$BASE_DIR/Megatron-LM" ]; then
  git clone https://github.com/NVIDIA/Megatron-LM.git --recursive "$BASE_DIR/Megatron-LM"
fi
cd "$BASE_DIR/Megatron-LM" && git checkout ${MEGATRON_COMMIT} && pip install -e . --no-build-isolation

# --- slime ---
cd "$SLIME_DIR"
# Install slime's pure-python runtime deps first (wandb, ray, accelerate,
# transformers, etc.) from its requirements.txt, then install slime itself
# with --no-deps so pip doesn't re-resolve and stomp our pinned native libs
# (torch+cu129, sglang-kernel+cu129, ...).
pip install -r requirements.txt
pip install -e . --no-deps

# int4_qat kernel (matches Dockerfile)
cd "$SLIME_DIR/slime/backends/megatron_utils/kernels/int4_qat"
pip install . --no-build-isolation

# https://github.com/pytorch/pytorch/issues/168167
pip install nvidia-cudnn-cu12==9.16.0.29
pip install "numpy==1.26.4" "scipy==1.17.1"
# kernels 0.15.x trips a ValueError on transformers.integrations.hub_kernels;
# pin to <0.15 so `import sglang` works at runtime.
pip install "kernels<0.15.0"

# --- apply patches in the same order as the Dockerfile ---
patch_dir="$SLIME_DIR/docker/patch/${PATCH_VERSION}"
if [ ! -d "$patch_dir" ]; then
  echo "Patch directory does not exist: $patch_dir" >&2
  exit 1
fi

cd "$BASE_DIR/sglang"
for patch_name in sglang.patch sglang-top_p.patch sglang-release_hicache.patch sglang-pull_weights.patch; do
  patch_path="$patch_dir/${patch_name}"
  if [ ! -f "$patch_path" ]; then
    continue
  fi
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
if [ ! -f "$megatron_patch" ]; then
  echo "Megatron patch does not exist: $megatron_patch" >&2
  exit 1
fi
if git apply --reverse --check "$megatron_patch"; then
  echo "megatron.patch already applied, skipping"
else
  git update-index --refresh || true
  if ! git apply "$megatron_patch" --3way; then
    echo "megatron.patch does not apply cleanly" >&2
    exit 1
  fi
  if git grep -n '^<<<<<<< ' -- .; then
    echo "megatron patch failed to apply cleanly. Please resolve conflicts." >&2
    exit 1
  fi
fi

# --- final smoke test (needs a GPU for sglang import) ---
cd "$WORKSPACE_DIR"
if python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  python - <<'PY'
import sglang
import torch
import torchaudio
import torchvision
assert torch.__version__ == "2.11.0+cu129"
assert torchaudio.__version__ == "2.11.0+cu129"
assert torchvision.__version__ == "0.26.0+cu129"
assert hasattr(torch.ops.torchvision, "nms")
print("GPU smoke test PASSED")
PY
else
  echo
  echo "No GPU detected on this machine -- skipped the CUDA smoke test."
  echo "On the GPU server, verify with:"
  echo '  python -c "import torch; print(torch.__version__, torch.version.cuda)"'
  echo '  python -c "import sglang, transformer_engine, flash_attn; print(\"native stack ok\")"'
  echo '  python -c "import megatron; print(\"megatron ok\")"'
fi

echo
echo "=== slime env build finished ==="
"$CONDA_BIN" env list
