#!/bin/bash
# Unified eval_all launcher — evaluate a trained model on any benchmark suite bench.
#
# Unlike the per-benchmark eval_* modules, this delegates scoring entirely to the
# benchmark suite unified framework (framework/run_eval.py). Supply a bench_id and
# the bench's own config governs prompt/scoring.
#
# Usage:
#   bash eval.sh --bench-id aime2025                    # default: limit 500, model from MODEL_NAME
#   bash eval.sh --bench-id gpqa --limit 100 --split default
#
# Config via env (mirrors sibling eval.sh modules):
#   MODEL_NAME       model name registered in the vLLM endpoint
#   OPENAI_BASE_URL  OpenAI-compatible base URL (default http://0.0.0.0:8101/v1)
#   BENCHMARK_ROOT  benchmark suite repo root (default $BENCHMARK_ROOT)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_URL="${OPENAI_BASE_URL:-http://0.0.0.0:8101/v1}"
API_KEY="${OPENAI_API_KEY:-sk-123}"
MODEL_NAME="${MODEL_NAME:-}"

MAX_TOKENS="${MAX_TOKENS:-0}"
MAX_WORKERS="${MAX_WORKERS:-0}"
TIMEOUT="${TIMEOUT:-0}"
LIMIT="${LIMIT:-500}"

RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-}}"

# ---- required -------------------------------------------------------------
BENCH_ID="${BENCH_ID:-${1:-}}"
if [[ -z "${BENCH_ID}" ]]; then
    echo "usage: bash eval.sh <bench_id> [extra eval_all.py flags...]" >&2
    exit 2
fi

mkdir -p "${OUTPUT_DIR}"

echo ""
echo "======================================"
echo "  Unified eval_all Evaluation"
echo "======================================"
echo "  Bench id:   ${BENCH_ID}"
echo "  Model:      ${MODEL_NAME}"
echo "  Base URL:   ${BASE_URL}"
echo "  Max tokens: ${MAX_TOKENS}"
echo "  Max workers:${MAX_WORKERS}"
echo "  Limit:      ${LIMIT}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "======================================"
echo ""

cd "${SCRIPT_DIR}"

python eval_all.py \
    --model "${MODEL_NAME}" \
    --base-url "${BASE_URL}" \
    --api-key "${API_KEY}" \
    --bench-id "${BENCH_ID}" \
    --max-tokens "${MAX_TOKENS}" \
    --max-workers "${MAX_WORKERS}" \
    --timeout "${TIMEOUT}" \
    --limit "${LIMIT}" \
    --run-dir "${OUTPUT_DIR}" \
    "$@"

echo ""
echo "Done. Results in ${OUTPUT_DIR}/ (metrics.json + reward.json + samples.jsonl)"
