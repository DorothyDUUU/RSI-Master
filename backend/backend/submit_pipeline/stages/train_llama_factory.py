"""llama-factory training bash fragments (stage-1 + ckpt).

原 ``pipeline_llama_factory_gsm8k.py`` 的逐字节切片。``$$VAR`` = bash 变量,
``${VAR}`` = Template 占位符 (render 时 substitute 一次替换)。
"""

from __future__ import annotations

# --- pipeline 头部变量 (llama-factory 路径) ---------------------------------
FRAGMENT_LF_VARS = r"""EXPERIMENT_ID="${EXPERIMENT_ID}"
WORKER_ID="${WORKER_ID}"
BENCHMARK="${BENCHMARK}"
TRAINING_MODE="${TRAINING_MODE}"
TRAINING_CONFIG_PATH="${TRAINING_CONFIG_PATH}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH}"
WORKSPACE="${WORKSPACE}"
CHECKPOINT_DIR="${CHECKPOINT_DIR}"
TENSORBOARD_DIR="${TENSORBOARD_DIR}"
EVAL_DIR="${EVAL_DIR}"
RESULT_JSON="${RESULT_JSON}"
PIPELINE_LOG="${PIPELINE_LOG}"
VLLM_PORT="${VLLM_PORT}"
# stage-2 vLLM serve 参数 (任务 #79): 默认 gpu-memory-utilization=0.6 + 充足的
# max-model-len, 避免共享卡上 serve 必 OOM / eval max_tokens 被截断。
# 可用 config ``evaluation.eval_all.serve_gpu_memory_utilization`` /
# ``serve_max_model_len`` 覆盖 (值透传到 SubmitRequest.eval_all)。
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN}"
EVAL_SCRIPT_DIR="${EVAL_SCRIPT_DIR}"
EVAL_PY="${EVAL_PY}"
METRIC_KEY="${METRIC_KEY}"
MAX_TOKENS="${MAX_TOKENS}"
MAX_WORKERS="${MAX_WORKERS}"
LIMIT="${LIMIT}"
# 训练数据集目录 —— 指向本次 run 在 workspace/public 里的 llama-factory 副本.
# 自定义数据集必须注册到这里, 绝不允许写 source tree.
DATASET_DIR="${DATASET_DIR}"
# stage-0 数据预检脚本 (backend/backend/llama_factory/scripts/...), 在 stage-1
# 占住 GPU 之前拦截 dataset_info.json 列名注册错误.
PREFLIGHT_SCRIPT="${PREFLIGHT_SCRIPT}"
# stage-0 去污 (decontamination): heavy python + 入口脚本 + 评测目标。
# BENCH_ID/BMK_ROOT 来自提交参数 (benchmark suite format/<id>/)。
DECONTAM_PYTHON="${DECONTAM_PYTHON}"
DECONTAM_SCRIPT="${DECONTAM_SCRIPT}"
BENCH_ID="${BENCH_ID}"
BMK_ROOT="${BMK_ROOT}"

mkdir -p "$$WORKSPACE" "$$CHECKPOINT_DIR" "$$TENSORBOARD_DIR" "$$EVAL_DIR"
"""

# --- stage 1: llama-factory SFT (+ LATEST_CKPT) -------------------------------
FRAGMENT_STAGE1_LF = r"""
# --- stage 1: llama-factory SFT ---------------------------------------------
log "stage-1: llama-factory SFT starts"
# shellcheck disable=SC1090
source "${LLAMA_FACTORY_ENV}"
cd "${LLAMA_FACTORY_CWD}"
log "stage-1: using $$(which python) / $$(which llamafactory-cli)"

# 通过 CLI 的 key=value 语法覆盖 YAML 中的落盘位置和数据集路径 (见
# backend/backend/llama_factory/documents/getting_started/sft.rst). 训练 YAML
# 里剩下的字段 (learning_rate, epochs, ...) 全部保持不动, 由 agent 通过
# ``training_config_path`` 事先落盘决定。
# ``dataset_dir`` 强制指向本次 run 的 llama-factory 副本 (workspace/public/...),
# 让 agent 注册在 run 副本 data/ 里的自定义数据集生效, 同时挡住对 source tree
# (backend/backend/llama_factory/llama_factory/data) 的写入路径。
STAGE1_STATUS=0
llamafactory-cli train "$$TRAINING_CONFIG_PATH" \
    output_dir="$$CHECKPOINT_DIR" \
    logging_dir="$$TENSORBOARD_DIR" \
    report_to=tensorboard \
    overwrite_output_dir=true \
    dataset_dir="$$DATASET_DIR" \
    2>&1 | tee -a "$$STAGE_LOG" || STAGE1_STATUS=$$?
if [[ "$$STAGE1_STATUS" -ne 0 ]]; then
    FAILURE_REASON="stage-1 (llama-factory train) exited with code $$STAGE1_STATUS"
    log "$$FAILURE_REASON"
    exit "$$STAGE1_STATUS"
fi
log "stage-1: SFT finished, checkpoint under $$CHECKPOINT_DIR"

# 挑最新的 checkpoint 目录 (无 checkpoint-* 时回落到 output_dir).
LATEST_CKPT="$$(find "$$CHECKPOINT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | awk '{print $$2}')"
if [[ -z "$$LATEST_CKPT" ]]; then
    LATEST_CKPT="$$CHECKPOINT_DIR"
fi

# LoRA 已下线 (2026-08-20): 全参 SFT 直接可服务, SERVE_DIR 恒等于 LATEST_CKPT.
SERVE_DIR="$$LATEST_CKPT"
"""
