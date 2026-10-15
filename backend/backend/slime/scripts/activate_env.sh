#!/bin/bash
# ============================================================================
# Demo: 激活 slime 训练环境（conda, Python 3.12 / PyTorch 2.11+cu129）
#
# 用法（两种）：
#   1) source scripts/activate_env.sh        # 在当前 shell 激活，可继续用 conda/python
#   2) bash scripts/activate_env.sh          # 子 shell 里激活并打印信息（退出后不保留）
#
# 等价的手工命令：
#   source $CONDA_ROOT/etc/profile.d/conda.sh
#   conda activate slime
# ============================================================================
set -uo pipefail

CONDA_ROOT="${CONDA_ROOT:-}"
CONDA_ENV="${CONDA_ENV:-slime}"

# 1) 把环境 bin 放到 PATH 最前（ray/sglang/torch 等可执行文件都从这里拿）
export PATH="$CONDA_ROOT/envs/$CONDA_ENV/bin:$PATH"

# 2) source conda.sh 定义 conda / activate 函数
#    非交互 shell 下 .bashrc 的 conda hook eval 不可靠（会报 "Run 'conda init' before 'conda activate'"）
if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
else
    echo "ERROR: conda.sh not found under $CONDA_ROOT/etc/profile.d/" >&2
    return 1 2>/dev/null || exit 1
fi

# 3) 激活（set -u 下 source 可能报 PS1 unbound，故临时关掉 nounset）
set +u
conda activate "$CONDA_ENV"
set -u

# 4) 与构建时一致的架构环境变量（JIT / 原生扩展匹配 H100/H200）
export CUDA_HOME="$CONDA_PREFIX"
export TORCH_CUDA_ARCH_LIST=9.0
export NVTE_CUDA_ARCHS=9.0

echo "==> slime env activated: $(which python3)"
echo "    torch $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '?')"
