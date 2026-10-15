"""Bash pipeline fragments + per-framework composers.

每个 FRAGMENT_* 是 ``string.Template`` 片段:
    - ``$$VAR``       -> bash 变量 (Template 转义后原样保留为字面 $VAR)
    - ``${VAR}``      -> Template 占位符, render 时 ``substitute()`` **一次**替换
    - python heredoc 正文同样会被 Template 扫描, 片段内已避免出现 ``$``。

``compose_llama_factory()`` / ``compose_slime()`` 各自把片段拼成一个 **单个**
Template, 再交给 render.py 做单次 substitute —— 与旧单文件的组装方式等价。

共享片段 (banner / helpers) 是原 ``pipeline_llama_factory_gsm8k.py``
``_PIPELINE_TEMPLATE`` 的逐字节切片 (banner 仅标题行与 Written-by 两处外观改动)。
"""

from __future__ import annotations

from string import Template

from .stages.preflight import (
    FRAGMENT_STAGE0_LF,
    FRAGMENT_STAGE0_LF_TEMPLATE_EOS,
    FRAGMENT_STAGE0_SLIME,
)
from .stages.serve_eval import FRAGMENT_STAGE2, FRAGMENT_STAGE3
from .stages.train_llama_factory import FRAGMENT_LF_VARS, FRAGMENT_STAGE1_LF
from .stages.train_slime import (
    FRAGMENT_SLIME_EPILOGUE,
    FRAGMENT_SLIME_VARS,
    FRAGMENT_STAGE15_SLIME,
    FRAGMENT_STAGE1_SLIME,
)

# --- 共享片段 ----------------------------------------------------------------

FRAGMENT_BANNER = r"""#!/usr/bin/env bash
# ============================================================================
# Auto-generated training pipeline (expos.backends.pipelines).
# Experiment: ${EXPERIMENT_ID}  (worker=${WORKER_ID})
# Written by: expos.backends.pipelines
# ============================================================================
set -eEuo pipefail

"""

FRAGMENT_HELPERS = r"""
VLLM_PID=""
STATUS="failed"
SCORE="null"
EVAL_STATUS="failed"
FAILURE_REASON=""
STAGE_LOG="$$WORKSPACE/pipeline.stage.log"

# --- helpers -----------------------------------------------------------------
log() {
    local msg="[$$(date '+%F %T')] $$*"
    echo "$$msg"
    echo "$$msg" >> "$$STAGE_LOG"
}

write_result_json() {
    # 用 python 生成 result.json, 保证格式跟 pydantic ExperimentResult 一致.
    /usr/bin/env python3 - "$$RESULT_JSON" <<'PYEOF'
import json, os, sys, time
result = {
    "schema_version": "rsi_master.result.v1",
    "experiment_id": os.environ.get("EXPERIMENT_ID", ""),
    "worker_id": os.environ.get("WORKER_ID", ""),
    "status": os.environ.get("STATUS", "failed"),
    "eval_status": os.environ.get("EVAL_STATUS", "failed"),
    "benchmark": os.environ.get("BENCHMARK", ""),
    "training_mode": os.environ.get("TRAINING_MODE", ""),
    "score": None if os.environ.get("SCORE", "null") == "null" else float(os.environ["SCORE"]),
    "data_summary": "",
    "effective_config": {"training_config_path": os.environ.get("TRAINING_CONFIG_PATH", "")},
    "artifacts": [
        {"type": "checkpoint", "path": os.environ.get("CHECKPOINT_DIR", ""), "sha256": "", "summary": "llama-factory output_dir", "metadata": {}},
        {"type": "tensorboard", "path": os.environ.get("TENSORBOARD_DIR", ""), "sha256": "", "summary": "tensorboard logs", "metadata": {}},
        {"type": "eval", "path": os.environ.get("EVAL_DIR", ""), "sha256": "", "summary": "benchmark eval outputs", "metadata": {}},
        {"type": "gpu_log", "path": os.environ.get("PIPELINE_LOG", ""), "sha256": "", "summary": "gpu_backend script log", "metadata": {}},
    ],
    "submit_result": {
        "status": os.environ.get("STATUS", "failed"),
        "score": None if os.environ.get("SCORE", "null") == "null" else float(os.environ["SCORE"]),
        "benchmark": os.environ.get("BENCHMARK", ""),
        "training_mode": os.environ.get("TRAINING_MODE", ""),
        "trial_path": os.environ.get("WORKSPACE", ""),
        "train_data_path": os.environ.get("TRAIN_DATA_PATH", ""),
        "checkpoint_dir": os.environ.get("CHECKPOINT_DIR", ""),
        "tensorboard_dir": os.environ.get("TENSORBOARD_DIR", ""),
        "eval_dir": os.environ.get("EVAL_DIR", ""),
        "result_json_path": os.environ.get("RESULT_JSON", ""),
        "pipeline_script_path": os.environ.get("PIPELINE_SCRIPT_PATH", ""),
        "gpu_log_path": os.environ.get("PIPELINE_LOG", ""),
    },
    "failure_reason": os.environ.get("FAILURE_REASON", ""),
    "notes": "",
    "created_at": time.time(),
}
with open(sys.argv[1], "w", encoding="utf-8") as fh:
    json.dump(result, fh, indent=2, ensure_ascii=False)
PYEOF
}

cleanup() {
    if [[ -n "$$VLLM_PID" ]] && kill -0 "$$VLLM_PID" 2>/dev/null; then
        log "cleanup: killing vllm PID $$VLLM_PID"
        kill "$$VLLM_PID" 2>/dev/null || true
        # 给它一点关闭时间, 再强制 SIGKILL 收尾
        for _ in 1 2 3 4 5; do
            kill -0 "$$VLLM_PID" 2>/dev/null || break
            sleep 1
        done
        kill -9 "$$VLLM_PID" 2>/dev/null || true
    fi
    # 无论怎么退出, 都写一次最新的 result.json
    STATUS="$$STATUS" EVAL_STATUS="$$EVAL_STATUS" SCORE="$$SCORE" \
        FAILURE_REASON="$$FAILURE_REASON" \
        EXPERIMENT_ID="$$EXPERIMENT_ID" WORKER_ID="$$WORKER_ID" \
        BENCHMARK="$$BENCHMARK" TRAINING_MODE="$$TRAINING_MODE" \
        TRAINING_CONFIG_PATH="$$TRAINING_CONFIG_PATH" \
        WORKSPACE="$$WORKSPACE" TRAIN_DATA_PATH="$$TRAIN_DATA_PATH" \
        CHECKPOINT_DIR="$$CHECKPOINT_DIR" TENSORBOARD_DIR="$$TENSORBOARD_DIR" \
        EVAL_DIR="$$EVAL_DIR" RESULT_JSON="$$RESULT_JSON" \
        PIPELINE_LOG="$$PIPELINE_LOG" \
        PIPELINE_SCRIPT_PATH="$${BASH_SOURCE[0]}" \
        write_result_json
    log "cleanup: result.json written -> $$RESULT_JSON  status=$$STATUS score=$$SCORE"
}
trap cleanup EXIT
trap 'FAILURE_REASON="pipeline error at line $$LINENO (cmd: $$BASH_COMMAND)"; STATUS="failed"; exit 1' ERR
"""


# --- 组装: 每个框架一个 Template, render 时单次 substitute -------------------

def compose_llama_factory(
    *,
    include_template_preflight: bool = True,
) -> Template:
    """llama-factory SFT + vLLM serve + eval 完整管线 (与旧单文件等价).

    - ``include_template_preflight=False``: 跳过 stage-0b 模板-eos 预检.
      0819 后该预检已是咨询性 (不再拒绝 qwen3/qwen3_nothink 等 replace_eos 模板,
      只打印 turn-ender vs base eos 比对), 一般无需关闭; 保留此开关作兜底逃生口.
    """
    parts = [
        FRAGMENT_BANNER,
        FRAGMENT_LF_VARS,
        FRAGMENT_HELPERS,
        FRAGMENT_STAGE0_LF,
    ]
    if include_template_preflight:
        parts.append(FRAGMENT_STAGE0_LF_TEMPLATE_EOS)
    parts.append(FRAGMENT_STAGE1_LF)
    parts.extend([FRAGMENT_STAGE2, FRAGMENT_STAGE3])
    return Template("".join(parts))


def compose_slime(*, include_eval: bool = True) -> Template:
    """slime SFT + torch_dist->HF 转换 + (默认) vLLM serve + eval.

    - ``include_eval=True`` (默认): stage-1 后接 stage-1.5 转换 + 与 llama-factory
      共用的 FRAGMENT_STAGE2/FRAGMENT_STAGE3 (serve + eval_all), 让 slime 训练
      产物像 LF 一样出真分。
    - ``include_eval=False``: 只训练 (不转换不 serve), 走 FRAGMENT_SLIME_EPILOGUE
      (eval_status=skipped)。对应 slime_config ``skip_eval: true`` —— 完全恢复
      旧行为, 也省掉 35a3b 那 ~10-30min 的转换开销。
    """
    parts = [
        FRAGMENT_BANNER,
        FRAGMENT_SLIME_VARS,
        FRAGMENT_HELPERS,
        FRAGMENT_STAGE0_SLIME,
        FRAGMENT_STAGE1_SLIME,
    ]
    if include_eval:
        parts.extend([FRAGMENT_STAGE15_SLIME, FRAGMENT_STAGE2, FRAGMENT_STAGE3])
    else:
        parts.append(FRAGMENT_SLIME_EPILOGUE)
    return Template("".join(parts))
