#!/bin/bash
# ============================================================================
# Demo: 观测训练进度
#
#   1) 实时日志：tail -f 训练日志（每步会打 train/loss、perf/step_time 等）
#   2) ray dashboard：http://<节点IP>:8265（训练期间可看 actor 状态）
#
# 用法：
#   bash scripts/watch_training.sh [日志路径]
#   默认日志 = $OUTPUT_DIR/train.log（与 launch_sft.sh 同款默认值）
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"

OUTPUT_DIR="${OUTPUT_DIR:-}"
LOG_FILE="${1:-$OUTPUT_DIR/train.log}"

if [ ! -f "$LOG_FILE" ]; then
    echo "日志还不存在: $LOG_FILE"
    echo "训练启动后会出现在这里（launch_sft.sh 的 OUTPUT_DIR）"
    echo "如果训练还没开始，先确认 GPU 节点上已运行 scripts/launch_sft.sh"
    exit 1
fi

echo "==> tail -f $LOG_FILE  (Ctrl-C 退出)"
# 只看关键行：训练步指标 / 保存 ckpt / 错误
tail -f "$LOG_FILE" | grep --line-buffered -E "step [0-9]+:|saving checkpoint|successfully saved|ERROR|Traceback|Traceback \(most recent|SMOKE|训练结束|Exception|error"
