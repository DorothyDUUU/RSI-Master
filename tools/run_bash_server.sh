#!/bin/bash
# Bash Backend Server 启动脚本
# 带资源限制配置

set -e

# 默认配置
PORT="${PORT:-40059}"
HOST="${HOST:-0.0.0.0}"
WORKSPACE="${WORKSPACE:-}"

# 资源限制配置
MEMORY_LIMIT="${MEMORY_LIMIT:-4g}"          # 内存限制: 4GB
CPU_LIMIT="${CPU_LIMIT:-600}"               # CPU 时间限制: 600秒 (10分钟)
CPU_DEVICES="${CPU_DEVICES:-0-63}"           # CPU 核心绑定 (默认使用 3/4 核心: 0-95)
MAX_OUTPUT="${MAX_OUTPUT:-50000}"          # 最大输出长度: 100K 字符
MAX_CONCURRENT="${MAX_CONCURRENT:-30}"      # 最大并发进程数

# 日志目录
DATA_DIR="${DATA_DIR:-}"

# 额外白名单路径 (用空格分隔)
EXTRA_WHITELIST="${EXTRA_WHITELIST:-}"

# 构建命令
CMD="python -m tools.bash_backend"
CMD="$CMD --port $PORT"
CMD="$CMD --host $HOST"

# 工作目录
if [ -n "$WORKSPACE" ]; then
    CMD="$CMD --workspace $WORKSPACE"
fi

# 资源限制参数
if [ -n "$MEMORY_LIMIT" ]; then
    CMD="$CMD --memory-limit $MEMORY_LIMIT"
fi

if [ -n "$CPU_LIMIT" ] && [ "$CPU_LIMIT" != "0" ]; then
    CMD="$CMD --cpu-limit $CPU_LIMIT"
fi

if [ -n "$CPU_DEVICES" ]; then
    CMD="$CMD --cpu-devices $CPU_DEVICES"
fi

CMD="$CMD --max-output $MAX_OUTPUT"
CMD="$CMD --max-concurrent $MAX_CONCURRENT"

# 日志目录
if [ -n "$DATA_DIR" ]; then
    CMD="$CMD --data-dir $DATA_DIR"
fi

# 额外白名单
if [ -n "$EXTRA_WHITELIST" ]; then
    for path in $EXTRA_WHITELIST; do
        CMD="$CMD --whitelist $path"
    done
fi

echo "=========================================="
echo "Starting Bash Backend Server"
echo "=========================================="
echo "Port:           $PORT"
echo "Host:           $HOST"
echo "Workspace:      ${WORKSPACE:-'(not set)'}"
echo "Memory Limit:   $MEMORY_LIMIT"
echo "CPU Limit:      ${CPU_LIMIT}s"
echo "CPU Devices:    ${CPU_DEVICES:-'(all)'}"
echo "Max Output:     $MAX_OUTPUT chars"
echo "Max Concurrent: $MAX_CONCURRENT"
echo "Data Dir:       ${DATA_DIR:-'(default)'}"
echo "=========================================="

# 启动服务
exec $CMD