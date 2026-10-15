#!/usr/bin/env bash
set -euo pipefail

# ============================================
# HF Search CLI – Server Management Script
# ============================================
#
# Usage:
#   ./start_server.sh              # Start server
#   ./start_server.sh stop         # Stop server
#   ./start_server.sh restart      # Restart server
#   ./start_server.sh status       # Check status
#
# Environment variables (optional):
#   HF_SANDBOX_PORT        Server port (default 8899)
#   HF_SANDBOX_HOST        Listen address (default 0.0.0.0)
#   HF_ENDPOINT            HF mirror URL (default https://huggingface.co)
#   HF_TOKEN               HuggingFace token
#   HF_RATE_LIMIT_RPM      Max requests per minute (default 30)
#   HF_MAX_CONCURRENT      Max concurrent requests (default 3)
#   HF_SEARCH_CACHE_TTL    Search cache TTL in seconds (default 600)
#   PYTHON                 Python interpreter path

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_DIR="${SCRIPT_DIR}/.pids"
LOG_DIR="${SCRIPT_DIR}/.logs"

PORT="${HF_SANDBOX_PORT:-8899}"
HOST="${HF_SANDBOX_HOST:-0.0.0.0}"
PYTHON="${PYTHON:-python3}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

mkdir -p "${PID_DIR}" "${LOG_DIR}"

start() {
    local pid_file="${PID_DIR}/server.pid"
    if [[ -f "${pid_file}" ]]; then
        local pid
        pid=$(cat "${pid_file}")
        if kill -0 "${pid}" 2>/dev/null; then
            log_info "Server already running (PID: ${pid}, port: ${PORT})"
            return 0
        fi
    fi

    log_info "Starting HF Search Server (${HOST}:${PORT}) ..."

    # Start server in background
    nohup "${PYTHON}" -m huggingface_search_cli.server.server \
        > "${LOG_DIR}/server.log" 2>&1 &
    local pid=$!
    echo "${pid}" > "${pid_file}"

    # Wait briefly and check if process is still alive
    sleep 2
    if kill -0 "${pid}" 2>/dev/null; then
        log_info "Server started (PID: ${pid}, port: ${PORT})"
        log_info "  Health: http://${HOST}:${PORT}/health"
        log_info "  Log:    ${LOG_DIR}/server.log"
    else
        log_error "Server failed to start. Check log: ${LOG_DIR}/server.log"
        rm -f "${pid_file}"
        return 1
    fi
}

stop() {
    local pid_file="${PID_DIR}/server.pid"
    if [[ -f "${pid_file}" ]]; then
        local pid
        pid=$(cat "${pid_file}")
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
            # Wait for process to die
            local count=0
            while kill -0 "${pid}" 2>/dev/null && [[ ${count} -lt 10 ]]; do
                sleep 1
                count=$((count + 1))
            done
            if kill -0 "${pid}" 2>/dev/null; then
                kill -9 "${pid}" 2>/dev/null || true
                log_warn "Force-killed server (PID: ${pid})"
            else
                log_info "Server stopped (PID: ${pid})"
            fi
        else
            log_warn "Server process ${pid} not running (stale PID)"
        fi
        rm -f "${pid_file}"
    else
        log_warn "No PID file found"
    fi
}

status() {
    echo "============================================"
    echo "HF Search Server Status"
    echo "============================================"
    local pid_file="${PID_DIR}/server.pid"
    if [[ -f "${pid_file}" ]]; then
        local pid
        pid=$(cat "${pid_file}")
        if kill -0 "${pid}" 2>/dev/null; then
            echo -e "  ${GREEN}●${NC} Server: running (PID: ${pid}, port: ${PORT})"
        else
            echo -e "  ${RED}○${NC} Server: not running (stale PID: ${pid})"
        fi
    else
        echo -e "  ${RED}○${NC} Server: not running"
    fi
    echo "============================================"
    echo ""
    echo "  Endpoint: http://${HOST}:${PORT}/health"
    echo "  Log:      ${LOG_DIR}/server.log"
    echo ""

    # Try health check
    if command -v curl &>/dev/null; then
        local health
        health=$(curl -s --max-time 3 "http://localhost:${PORT}/health" 2>/dev/null || echo "")
        if [[ -n "${health}" ]]; then
            echo "  Health response:"
            echo "  ${health}" | python3 -m json.tool 2>/dev/null || echo "  ${health}"
        fi
    fi
}

case "${1:-start}" in
    start)   start ;;
    stop)    stop ;;
    restart) stop; sleep 2; start ;;
    status)  status ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac
