#!/bin/bash
# ============================================================================
# Demo: HF 权重 → Megatron torch_dist 格式（单卡一次，约 10-30 分钟）
#
# 为什么必须先转换：训练脚本的 --load 必须指向 torch_dist 产物，不能直接指
# HF 目录。hf_to_megatron 只按 ETP 切专家，256 专家张量塞不进 [num_local_experts]
# 参数；torch_dist 加载时按训练时的 TP/EP/PP 自动重分片。
#
# 在 GPU 节点上执行：
#   bash scripts/convert_hf_to_torch_dist.sh
#
# 换模型：
#   MODEL_DIR=<HF 权重目录> SAVE_DIR=<输出目录> bash scripts/convert_hf_to_torch_dist.sh
#   （模型架构需与 slime_plugins.models.qwen3_5 spec 匹配，或改下面 source 的 model 脚本）
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"

# ---------------- CONFIG：改这里（默认值 = 之前跑通的值） ----------------
WORKSPACE="${WORKSPACE:-}"
MODEL_DIR="${MODEL_DIR:-}"
SAVE_DIR="${SAVE_DIR:-}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/training_scripts/logs}"
GPU_ID="${GPU_ID:-0}"
# 官方模型 spec（提供 MODEL_ARGS：层数/隐藏维度/专家数等）
MODEL_SPEC="${MODEL_SPEC:-}"
# ----------------------------------------------------------------------------

[ -f "$MODEL_SPEC" ] || { echo "ERROR: 模型 spec 不存在: $MODEL_SPEC" >&2; exit 1; }
mkdir -p "$LOG_DIR"

# 激活环境
source ./activate_env.sh

# Megatron 仅能从仓库根导入（PYTHONPATH 指向）
export PYTHONPATH="$WORKSPACE/Megatron-LM"
# 单卡转换
export CUDA_VISIBLE_DEVICES=$GPU_ID
export WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=23456

# source 官方 spec 拿到 MODEL_ARGS
source "$MODEL_SPEC"

echo "==> 开始转换: $MODEL_DIR -> $SAVE_DIR"
echo "==> 日志: $LOG_DIR/convert_hf_to_torch_dist.log"
cd "$WORKSPACE/slime"
nohup python -u tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint "$MODEL_DIR" \
    --save "$SAVE_DIR" \
    > "$LOG_DIR/convert_hf_to_torch_dist.log" 2>&1 &

echo "launched pid $! (tail -f $LOG_DIR/convert_hf_to_torch_dist.log 看进度)"
