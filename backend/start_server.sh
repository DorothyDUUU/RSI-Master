#!/bin/bash
# Start GPU Backend Server
#
# Usage:
#   ./start_server.sh                          # defaults: 0.0.0.0:8989
#   BACKEND_PORT=8000 ./start_server.sh        # custom port
#   BACKEND_LOG_LEVEL=debug ./start_server.sh  # verbose logging

set -e

HOST="${BACKEND_HOST:-0.0.0.0}"
PORT="${BACKEND_PORT:-8989}"
LOG_LEVEL="${BACKEND_LOG_LEVEL:-info}"
LOG_DIR="${BACKEND_LOG_DIR:-}"
RELOAD="${BACKEND_RELOAD:-}"

mkdir -p "$LOG_DIR"

echo "=== GPU Backend Server ==="
echo "  Host:      $HOST"
echo "  Port:      $PORT"
echo "  Log level: $LOG_LEVEL"
echo "  Log dir:   $LOG_DIR"
echo "=========================="

# Build uvicorn args
UVICORN_ARGS=(
    --host "$HOST"
    --port "$PORT"
    --log-level "$LOG_LEVEL"
)

if [ -n "$RELOAD" ]; then
    UVICORN_ARGS+=(--reload)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

python -m uvicorn backend.server.main:app "${UVICORN_ARGS[@]}"
