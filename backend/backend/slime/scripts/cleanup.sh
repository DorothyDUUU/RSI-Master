#!/bin/bash
# ============================================================================
# Demo: 清理残留的 ray 集群（重跑训练前必做）
#
# 真实训练脚本开头也会做同样的事。如果上一次训练异常退出 / ray dashboard
# 起不来 / 端口被占，先跑这个。
#
#   bash scripts/cleanup.sh
# ============================================================================
set -uo pipefail

# slime 环境 bin 放最前，保证用的是环境里的 ray
SLIME_ENV="${SLIME_ENV:-}"   # slime conda env; see README
[ -n "$SLIME_ENV" ] && export PATH="$SLIME_ENV/bin:$PATH"

echo "==> ray stop --force（只清本框架 ray 集群, 不 pkill 节点其它 python）..."
ray stop --force 2>/dev/null || true

echo "==> 如需强杀残留: 先 ps -ef | grep <关键字> 查清 PID 再 kill -9 <PID>（禁止 pkill -9 python）"

echo "==> 清理完成（可用 bash scripts/launch_sft.sh 重新启动）"
