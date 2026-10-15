"""Stage-0 data-preflight bash fragments (per training framework).

每个 FRAGMENT_* 都是 ``string.Template`` 片段: ``$$VAR`` = bash 变量 (Template
转义后原样保留为字面 $VAR), ``${VAR}`` = 占位符 (render 时 substitute 一次替换)。

``FRAGMENT_STAGE0_LF`` 是原 ``pipeline_llama_factory_gsm8k.py`` 的逐字节切片,
唯一例外是 source 行: 原文件写 ``$${LLAMA_FACTORY_ENV}`` (Template 转义后引用
**未声明**的 bash 变量, 在 ``set -uo pipefail`` 下必崩, 只因 gpu_backend 环境里
恰好 export 了同名变量才没暴露); 这里统一改成 ``${LLAMA_FACTORY_ENV}`` 占位符,
与 stage-1/1.5/3 的写法一致, 由 render 替换为绝对路径, 不再依赖环境变量。
"""

from __future__ import annotations

# --- 共享: stage-0 去污 + 数据集注册 (先于各自预检, 用 heavy python) ---------
# 本仓库是 serverless CLI 且 launch env (.venv) 无 pyarrow, 所以去污在管线内用
# heavy python (LF: llama_factory_env / slime: slime conda env) 跑 decontam_stage.py:
# 读原始 TRAIN_DATA_PATH, 剔除命中评测 query 的行, 写净化副本 + report。
# benchmark query 源不可用 -> decontam_stage 降级 pass-through (report 记录 skipped),
# 不卡死整轮。DECONTAM_PYTHON / DECONTAM_SCRIPT / BENCH_ID / BMK_ROOT 由 render 替换。
FRAGMENT_STAGE0_DECONTAM_LF = r"""
# --- stage 0: 去污 + 数据集注册 (llama_factory) ------------------------------
mkdir -p "$$WORKSPACE/artifacts" "$$DATASET_DIR"
DECONTAM_STATUS=0
"$$DECONTAM_PYTHON" "$$DECONTAM_SCRIPT" \
    --mode llama_factory \
    --data "$$TRAIN_DATA_PATH" \
    --train-config "$$TRAINING_CONFIG_PATH" \
    --bench-id "$$BENCH_ID" \
    --bmk-root "$$BMK_ROOT" \
    --benchmark "$$BENCHMARK" \
    --out-jsonl "$$DATASET_DIR/decontaminated_train.jsonl" \
    --registry-dir "$$DATASET_DIR" \
    --report "$$WORKSPACE/artifacts/decontamination_report.json" \
    2>&1 | tee -a "$$STAGE_LOG" || DECONTAM_STATUS=$$?
if [[ "$$DECONTAM_STATUS" -ne 0 ]]; then
    FAILURE_REASON="stage-0 (decontamination) failed (exit $$DECONTAM_STATUS)"
    log "$$FAILURE_REASON"
    exit "$$DECONTAM_STATUS"
fi
log "stage-0: decontamination OK -> $$DATASET_DIR/decontaminated_train.jsonl"
"""

FRAGMENT_STAGE0_DECONTAM_SLIME = r"""
# --- stage 0: 去污 (slime) ---------------------------------------------------
mkdir -p "$$WORKSPACE/artifacts"
DECONTAM_STATUS=0
"$$SLIME_PYTHON" "$$DECONTAM_SCRIPT" \
    --mode slime \
    --data "$$TRAIN_DATA_PATH" \
    --bench-id "$$BENCH_ID" \
    --bmk-root "$$BMK_ROOT" \
    --benchmark "$$BENCHMARK" \
    --out-parquet "$$SLIME_DATA_FILE" \
    --report "$$WORKSPACE/artifacts/decontamination_report.json" \
    2>&1 | tee -a "$$STAGE_LOG" || DECONTAM_STATUS=$$?
if [[ "$$DECONTAM_STATUS" -ne 0 ]]; then
    FAILURE_REASON="stage-0 (decontamination) failed (exit $$DECONTAM_STATUS)"
    log "$$FAILURE_REASON"
    exit "$$DECONTAM_STATUS"
fi
log "stage-0: decontamination OK -> $$SLIME_DATA_FILE"
"""

# --- llama-factory: 训练数据预检 (GPU 入队前) --------------------------------
FRAGMENT_STAGE0_LF = r"""
# --- stage 0: 训练数据预检 (GPU 入队前) -----------------------------------
# 任务 #113/#114: 用真实 llamafactory DatasetAttr 解析 dataset_info.json, 在
# stage-1 占住一张卡之前拦截列名注册错误 —— 最常见的是 ``columns`` 写成行字段名
# (instruction/input/output 而非 prompt/query/response), 或映射字段在数据行里
# 不存在 (mode 2 KeyError)。预检失败立刻 exit, 不烧 GPU。
# 脚本缺失 = 集群 checkout 未更新 -> 降级 warning 放行, 不卡死整轮.
if [[ -f "$${PREFLIGHT_SCRIPT}" ]]; then
    log "stage-0: data preflight ($${PREFLIGHT_SCRIPT})"
    # shellcheck disable=SC1090
    source "${LLAMA_FACTORY_ENV}"
    PREFLIGHT_STATUS=0
    python "$${PREFLIGHT_SCRIPT}" \
        --train-config "$${TRAINING_CONFIG_PATH}" \
        --dataset-dir "$${DATASET_DIR}" \
        2>&1 | tee -a "$${STAGE_LOG}" || PREFLIGHT_STATUS=$$?
    if [[ "$${PREFLIGHT_STATUS}" -ne 0 ]]; then
        FAILURE_REASON="stage-0 (data preflight) rejected training data before GPU launch (exit $${PREFLIGHT_STATUS})"
        log "$${FAILURE_REASON}"
        exit "$${PREFLIGHT_STATUS}"
    fi
    log "stage-0: data preflight OK"
else
    log "stage-0: preflight script not found ($${PREFLIGHT_SCRIPT}), skipping"
fi
"""

# --- llama-factory: 模板-eos 预检 (咨询性, 数据预检之后, GPU 入队前) ----------
# 2026-08-19 六模板对比实证: 模板 turn-ender != base eos 的"分叉"
# (qwen3/qwen3_nothink = <|im_end|> 151645 vs Qwen3-Base <|endoftext|> 151643)
# 在当前 serve 路径 (vllm_env_0819 = vllm 0.27 + `--generation-config vllm`)
# 下**无害**: vllm 无条件把 tokenizer.eos 加入 stop set (`--generation-config vllm`
# 只中和采样默认值, gen_config 的 eos ids 仍被 union, 只是多几个 stop token),
# 模型发出 tokenizer.eos 即停 (full-SFT 三模板全 200/200 stop, 0.80-0.83).
# 因此该预检现在是**咨询性**: 只打印 turn-ender 与 base eos 的比对说明,
# 对齐报 OK、分叉报预期分裂 (指向训后自检 check_eos_alignment.py), **不拒绝**
# 任何模板; 仍拦的只有配置/环境错误 (exit 2). 真正决定 stop 的是训练容量:
# LoRA 已下线 (2026-08-20); 0819 实证 full-SFT 三模板 200/200 stop (0.80-0.83),
# 数据/epochs/容量不足的模型在长 few-shot 里不发 stop token -> length loop.
FRAGMENT_STAGE0_LF_TEMPLATE_EOS = r"""
# --- stage 0b: 模板-eos 预检 (咨询性: 打印 turn-ender vs base eos 比对) --------
# 与数据预检同款降级策略: 脚本缺失 = 集群 checkout 未更新 -> warning 放行.
# 注意: 预检本身 exit 0 (不拦模板); 只有脚本报配置错误才 exit 2 -> 依旧 exit.
TEMPLATE_PREFLIGHT="${TEMPLATE_PREFLIGHT}"
if [[ -f "$${TEMPLATE_PREFLIGHT}" ]]; then
    log "stage-0: template-eos preflight ($${TEMPLATE_PREFLIGHT})"
    # shellcheck disable=SC1090
    source "${LLAMA_FACTORY_ENV}"
    TEMPLATE_PREFLIGHT_STATUS=0
    python "$${TEMPLATE_PREFLIGHT}" \
        --train-config "$${TRAINING_CONFIG_PATH}" \
        2>&1 | tee -a "$${STAGE_LOG}" || TEMPLATE_PREFLIGHT_STATUS=$$?
    if [[ "$${TEMPLATE_PREFLIGHT_STATUS}" -ne 0 ]]; then
        FAILURE_REASON="stage-0 (template-eos preflight) config error before GPU launch (exit $${TEMPLATE_PREFLIGHT_STATUS})"
        log "$${FAILURE_REASON}"
        exit "$${TEMPLATE_PREFLIGHT_STATUS}"
    fi
    log "stage-0: template-eos preflight OK"
else
    log "stage-0: template preflight script not found ($${TEMPLATE_PREFLIGHT}), skipping"
fi
"""

# --- slime: 训练数据预检 (数据格式校验, GPU 入队前) --------------------------
# slime 训练数据是 ``messages`` 列的 parquet (OpenAI chat 格式)。污染/leakage
# 检查在 WorkerService.submit 里已先做 (decontaminate_parquet), 这里只兜底
# **数据格式校验** —— 用 slime conda env 的解释器跑 slime_data_preflight.py,
# 在占住 GPU 前拦截 非 parquet / messages 列缺失 / 空样本 等错误, 快速失败。
FRAGMENT_STAGE0_SLIME = r"""
# --- stage 0: slime 数据预检 (数据格式校验, GPU 入队前) ---------------------
STAGE0_STATUS=0
"$$SLIME_PYTHON" "$$SLIME_PREFLIGHT" \
    --data-file "$$SLIME_DATA_FILE" \
    2>&1 | tee -a "$$STAGE_LOG" || STAGE0_STATUS=$$?
if [[ "$$STAGE0_STATUS" -ne 0 ]]; then
    FAILURE_REASON="stage-0 (slime data preflight) rejected training data before GPU launch (exit $$STAGE0_STATUS)"
    log "$$FAILURE_REASON"
    exit "$$STAGE0_STATUS"
fi
log "stage-0: slime data preflight OK"
"""
