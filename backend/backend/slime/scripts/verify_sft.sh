#!/bin/bash
# ============================================================================
# Demo: 单脚本快速验证 slime 跑 SFT 可用（环境预检 + 前置检查 + 端到端冒烟）
#
# 在 GPU 节点上执行：
#   bash scripts/verify_sft.sh
#
# 三段式：
#   1) 环境预检  torch/CUDA/GPU + megatron/slime/TE/sglang 导入链
#               失败立即退出（避免在坏环境上白跑一次训练）
#   2) 前置检查  冒烟数据 + torch_dist 权重是否存在（缺失时给出修复命令）
#   3) 端到端冒烟 launch_sft_smoke.sh：8 条合成数据 x 1 epoch = 1 个训练步 + 存 ckpt
#               覆盖 数据加载→loss mask→forward→sft loss→backward→optimizer→ckpt save
#
# 输出：ckpt 目录 $CKPT_DIR（含 SMOKE_PASSED 标记）
# 退出码：0=通过；1=环境/前置失败；2=冒烟训练失败
#
# 调试辅助：
#   SKIP_GPU_CHECK=1  跳过 GPU 检测（仅用于在非 GPU 节点上预检前置条件，正常不要用）
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"

# ---------------- CONFIG：改这里（默认值 = 之前跑通的值） ----------------
WORKSPACE="${WORKSPACE:-}"
DATA_FILE="${DATA_FILE:-$WORKSPACE/training_scripts/data/smoke_sft.parquet}"
TORCH_DIST_DIR="${TORCH_DIST_DIR:-}"
CKPT_DIR="${CKPT_DIR:-$WORKSPACE/training_scripts/ckpt/Qwen3-4B-Base_sft_smoke}"
# ----------------------------------------------------------------------------

echo "========== [1/3] 环境预检 =========="
if ! bash ./precheck_env.sh; then
    echo "[FAIL] 环境预检未通过（导入链 / GPU 异常），中止 —— 先修环境再重跑本脚本" >&2
    exit 1
fi

echo
echo "========== [2/3] 前置检查 =========="
if [ ! -f "$DATA_FILE" ]; then
    echo "[FAIL] 冒烟数据缺失: $DATA_FILE" >&2
    echo "  生成：python $WORKSPACE/training_scripts/prepare_smoke_data.py --dst $DATA_FILE --n 8" >&2
    exit 1
fi
if [ ! -d "$TORCH_DIST_DIR" ]; then
    echo "[FAIL] torch_dist 权重缺失: $TORCH_DIST_DIR" >&2
    echo "  转换（Qwen3-4B-Base）：" >&2
    echo "  MODEL_DIR=<repo>/public/models/Qwen/Qwen3-4B-Base \\" >&2
    echo "  SAVE_DIR=$TORCH_DIST_DIR \\" >&2
    echo "  MODEL_SPEC=$WORKSPACE/slime/scripts/models/qwen3-4B.sh \\" >&2
    echo "  bash scripts/convert_hf_to_torch_dist.sh" >&2
    exit 1
fi
echo "[OK] 冒烟数据: $DATA_FILE"
echo "[OK] torch_dist: $TORCH_DIST_DIR"

echo
echo "========== [3/3] 端到端 SFT 冒烟（1 步 + 存 ckpt）=========="
LOG_DIR="${LOG_DIR:-$WORKSPACE/training_scripts/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/verify_sft_$(date +%Y%m%d_%H%M%S).log}"
echo "==> 完整日志: $LOG_FILE"
if bash ./launch_sft_smoke.sh 2>&1 | tee "$LOG_FILE"; then
    echo
    echo "========== PASS: slime SFT 冒烟通过 =========="
    echo "ckpt: $CKPT_DIR（含 SMOKE_PASSED 标记）"
    exit 0
else
    echo "[FAIL] 冒烟训练未通过" >&2
    echo "      完整日志: $LOG_FILE" >&2
    echo "      定位报错: grep -nE 'ERROR|Traceback|error|ray job' '$LOG_FILE' | tail -30" >&2
    exit 2
fi
