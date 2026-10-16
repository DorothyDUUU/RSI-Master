"""slime SFT training bash fragments (stage-1 + stage-1.5 转换 + skip 收尾).

设计约束 (用户指定): **只在配置层做改动** —— 训练委托
``backend/backend/slime/scripts/launch_sft.sh``, 全部超参/数据/ckpt 路径通过
环境变量注入, 不重实现训练, 不碰 slime 安装/源码。

``$$VAR`` = bash 变量, ``${VAR}`` = Template 占位符 (render 时 substitute 一次)。
worker 不设 ``CUDA_VISIBLE_DEVICES`` —— GPU 卡由 gpu_backend 调度注入。

EOS 对齐不变式 (H6): 2026-08-23 slime 已接 serve/eval (stage-2 vLLM serve +
stage-3 eval_all, 复用 llama-factory 路径), 沿用同一不变式 —— 训练 turn-ender
必须等于 serve 侧 vLLM 认的 stop token。转换产物从 slime_config.model_dir
(HF base) 复制 tokenizer/config, 但 Qwen3 chat 模板训练模型终止发 <|im_end|>
而 base 的 eos 是 <|endoftext|> —— 只认 base eos 会复读不停 (实测 slime-8b-think
50/50 finish_reason=length, 平均 6930 chars)。stage-1.5 转换后 patch eos_token_id
同时包含 <|im_end|> (从 tokenizer_config 动态找 id) + base eos, 对齐 llama-factory
(generation_config eos_token_id=[151645,151643], 50/50 stop)。非 Qwen3 模型无
<|im_end|> token, patch 天然跳过。
"""

from __future__ import annotations

# --- pipeline 头部变量 (slime 路径) -----------------------------------------
# 与 llama-factory 路径共享 helper 所需的公共变量 (RESULT_JSON / WORKSPACE / ...),
# 另加 launch_sft.sh 的 env 覆盖项。``SLIME_*`` 全部来自 ``slime_config``。
FRAGMENT_SLIME_VARS = r"""EXPERIMENT_ID="${EXPERIMENT_ID}"
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
# slime 训练 (backend/backend/slime): 预检解释器 + launch_sft.sh + 数据/模型路径.
# 这些 SLIME_* 是 render.py 一次性替换的占位符; 片段里用 $$SLIME_* 引用为 bash 变量.
SLIME_PYTHON="${SLIME_PYTHON}"
# Fail loudly rather than silently falling back to PATH's python3: the whole
# slime stack (torch/megatron/sglang) lives only in that env. Build it with
# backend/backend/slime/docs/env_construct/, then export SLIME_ENV.
: "$${SLIME_PYTHON:?SLIME_ENV is not set — see backend/backend/slime/docs/env_construct/README.md}"
SLIME_PREFLIGHT="${SLIME_PREFLIGHT}"
SLIME_LAUNCH_SFT="${SLIME_LAUNCH_SFT}"
SLIME_DATA_FILE="${SLIME_DATA_FILE}"
SLIME_MODEL_DIR="${SLIME_MODEL_DIR}"
SLIME_TORCH_DIST_DIR="${SLIME_TORCH_DIST_DIR}"
SLIME_MODEL_SPEC="${SLIME_MODEL_SPEC}"
SLIME_OUTPUT_DIR="${SLIME_OUTPUT_DIR}"
SLIME_CKPT_DIR="${SLIME_CKPT_DIR}"
SLIME_NUM_EPOCH="${SLIME_NUM_EPOCH}"
SLIME_NUM_GPUS="${SLIME_NUM_GPUS}"
SLIME_TP_SIZE="${SLIME_TP_SIZE}"
SLIME_EP_SIZE="${SLIME_EP_SIZE}"
SLIME_HAS_NVLINK="${SLIME_HAS_NVLINK}"
SLIME_ROLLOUT_BATCH_SIZE="${SLIME_ROLLOUT_BATCH_SIZE}"
SLIME_GLOBAL_BATCH_SIZE="${SLIME_GLOBAL_BATCH_SIZE}"
SLIME_SEQ_LENGTH="${SLIME_SEQ_LENGTH}"
SLIME_MAX_TOKENS_PER_GPU="${SLIME_MAX_TOKENS_PER_GPU}"
SLIME_CP_SIZE="${SLIME_CP_SIZE}"
# stage-1.5 torch_dist -> HF 转换 (vLLM 只认 HF safetensors).
SLIME_HF_DIR="${SLIME_HF_DIR}"
SLIME_CONVERT_TO_HF="${SLIME_CONVERT_TO_HF}"
: "$${SLIME_CONVERT_TO_HF:?SLIME_CONVERT_TO_HF is not set — see backend/backend/slime/docs/env_construct/README.md}"
SLIME_MEGATRON_LM="${SLIME_MEGATRON_LM}"
SLIME_VOCAB_ARGS="${SLIME_VOCAB_ARGS}"
SLIME_ADD_MISSING="${SLIME_ADD_MISSING}"
# ray 集群隔离: temp-dir 后缀取自 WORKSPACE (每实验唯一, 并行任务不冲突);
# GCS/dashboard 随机端口由 launch_sft.sh 每次启动时自选 (>20000, 避开 worker_ports
# 区间 10002-19999)。stage-1.5 按同一 RAY_TMPDIR 精确清理 —— ray stop 无地址限定
# 会误杀并行任务, 禁用。
RAY_TAG="$$(printf '%s' "$${WORKSPACE}" | sha1sum | cut -c1-10)"
RAY_TMPDIR="/tmp/ray_$${RAY_TAG}"
# stage-2/3 (与 llama-factory 路径共用) —— 由 render.py 一次性 substitute.
VLLM_PORT="${VLLM_PORT}"
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN}"
EVAL_SCRIPT_DIR="${EVAL_SCRIPT_DIR}"
EVAL_PY="${EVAL_PY}"
METRIC_KEY="${METRIC_KEY}"
MAX_TOKENS="${MAX_TOKENS}"
MAX_WORKERS="${MAX_WORKERS}"
LIMIT="${LIMIT}"

mkdir -p "$$WORKSPACE" "$$CHECKPOINT_DIR" "$$TENSORBOARD_DIR" "$$EVAL_DIR"
"""

# --- stage 1: slime SFT (launch_sft.sh, env 覆盖) ---------------------------
FRAGMENT_STAGE1_SLIME = r"""
# --- stage 1: slime SFT (launch_sft.sh) -------------------------------------
# 委托 backend/backend/slime/scripts/launch_sft.sh: conda 激活 -> ray start ->
# ray job submit train_async.py。训练超参/数据/ckpt 全部通过环境变量注入, 脚本
# 内部 set -uo pipefail 并回传退出码, 由本管线 EXIT trap 捕获写 result.json。
# torch_dist ckpt 落在 $$CKPT_DIR (slime_config.ckpt_dir)。
log "stage-1: slime SFT starts ($${SLIME_LAUNCH_SFT})"
export DATA_FILE="$$SLIME_DATA_FILE"
export MODEL_DIR="$$SLIME_MODEL_DIR"
export TORCH_DIST_DIR="$$SLIME_TORCH_DIST_DIR"
export MODEL_SPEC="$$SLIME_MODEL_SPEC"
export OUTPUT_DIR="$$SLIME_OUTPUT_DIR"
export CKPT_DIR="$$SLIME_CKPT_DIR"
export NUM_EPOCH="$$SLIME_NUM_EPOCH"
export NUM_GPUS="$$SLIME_NUM_GPUS"
export TP_SIZE="$$SLIME_TP_SIZE"
export EP_SIZE="$$SLIME_EP_SIZE"
export HAS_NVLINK="$$SLIME_HAS_NVLINK"
export ROLLOUT_BATCH_SIZE="$$SLIME_ROLLOUT_BATCH_SIZE"
export GLOBAL_BATCH_SIZE="$$SLIME_GLOBAL_BATCH_SIZE"
export SEQ_LENGTH="$$SLIME_SEQ_LENGTH"
export MAX_TOKENS_PER_GPU="$$SLIME_MAX_TOKENS_PER_GPU"
export CP_SIZE="$$SLIME_CP_SIZE"
# ray 集群隔离: 唯一 temp-dir 传给 launch_sft.sh (并行 slime 任务各占独立集群)。
export RAY_TMPDIR="$${RAY_TMPDIR}"
# --- ESTALE 硬化 (2026-08-24): 拷到 run-local 再执行 --------------------------
# launch_sft.sh 在共享 NFS 上: 若训练期间被改写 (给下一 run 优化), 运行中 bash 进程
# 持有旧 inode, teardown 读脚本尾部 -> Stale file handle -> 训练成功被误标 failed
# (2026-08-23 平台 4 pod 全踩雷, memory nfs-estale-running-script)。把整个 scripts/
# 拷到 run 本地 (WORKSPACE/scripts, 每 run 专属、永不重写) 再执行 —— 执行进程持有
# run-local 文件, 对并发编辑免疫。launch_sft.sh 内部 cd 到自己目录并相对 source
# ./activate_env.sh, 故**整目录**拷贝 (只拷 launch_sft.sh 会断 activate_env 链接)。
SLIME_RUN_LOCAL_DIR="$$WORKSPACE/scripts"
rm -rf "$${SLIME_RUN_LOCAL_DIR}" && mkdir -p "$${SLIME_RUN_LOCAL_DIR}"
cp -r "$$(dirname "$${SLIME_LAUNCH_SFT}")/." "$${SLIME_RUN_LOCAL_DIR}/"
SLIME_RUN_LOCAL_LAUNCH="$${SLIME_RUN_LOCAL_DIR}/launch_sft.sh"
log "stage-1: run-local launch_sft.sh = $${SLIME_RUN_LOCAL_LAUNCH} (copied from $${SLIME_LAUNCH_SFT})"
STAGE1_STATUS=0
bash "$${SLIME_RUN_LOCAL_LAUNCH}" 2>&1 | tee -a "$$STAGE_LOG" || STAGE1_STATUS=$$?
if [[ "$$STAGE1_STATUS" -ne 0 ]]; then
    FAILURE_REASON="stage-1 (slime SFT via run-local launch_sft.sh) exited with code $$STAGE1_STATUS"
    log "$$FAILURE_REASON"
    exit "$$STAGE1_STATUS"
fi
log "stage-1: slime SFT finished, ckpt under $$SLIME_CKPT_DIR"
"""

# --- stage 1.5: torch_dist -> HF 转换 (vLLM 只认 HF safetensors) --------------
# slime 训练产物是 Megatron torch_dist (iter_*/__*.distcp), vLLM 不能加载, 必须
# 离线转成 HF safetensors。转换工具 slime/tools/convert_torch_dist_to_hf.py, 用
# slime conda env 解释器 (slime / megatron_core 都是该 env 的 editable 安装, 任意
# cwd 可 import)。纯 CPU 张量操作, 不占 GPU。--origin-hf-dir 提供 tokenizer/config/
# generation_config 并从 config.json 推断架构; --vocab-size 去 Megatron embedding
# padding (传真实 HF vocab, 无 padding 时是 no-op); -a 只补 ckpt 缺失的权重
# (EP 分片专家/共享参数的安全网, 已训练权重优先); -f 覆盖旧输出。
#
# 转换前先停本实验的 ray 集群: launch_sft.sh 只在开头清 Ray、结尾不停, head 进程
# 持续占满 NUM_GPUS 张卡 —— 不先释放, stage-2 vLLM serve 必然撞 Ray 占卡 OOM。
# 并行任务各占独立集群, 必须按 RAY_TMPDIR (WORKSPACE 哈希) 精确 pkill ——
# ray stop 无地址限定会杀掉节点上所有 ray 集群 (实测误杀并行任务的集群)。
FRAGMENT_STAGE15_SLIME = r"""
# --- stage 1.5: torch_dist -> HF 转换 ----------------------------------------
pkill -9 -f "$${RAY_TMPDIR}" 2>/dev/null || true
SLIME_INPUT_CKPT=""
LATEST_ITER=$$(cat "$$SLIME_CKPT_DIR/latest_checkpointed_iteration.txt" 2>/dev/null | tr -d '[:space:]')
# iter_ 目录 padding 宽度随 save 配置可能是 7 位或 8 位 (megatron %07d / %08d),
# 都试一下; 都找不到就按名字排序取最后一个 iter_* 兜底.
if [ -n "$$LATEST_ITER" ]; then
    for w in 8 7; do
        d="$$SLIME_CKPT_DIR/iter_$$(printf "%0$${w}d" "$$LATEST_ITER")"
        if [ -d "$$d" ]; then
            SLIME_INPUT_CKPT="$$d"
            break
        fi
    done
fi
[ -z "$$SLIME_INPUT_CKPT" ] && \
    SLIME_INPUT_CKPT=$$(find "$$SLIME_CKPT_DIR" -maxdepth 1 -type d -name 'iter_*' 2>/dev/null | sort | tail -1)
if [ -z "$$SLIME_INPUT_CKPT" ]; then
    FAILURE_REASON="stage-1.5: no iter_* ckpt under $$SLIME_CKPT_DIR"
    log "$$FAILURE_REASON"
    exit 1
fi
rm -rf "$$SLIME_HF_DIR" && mkdir -p "$$SLIME_HF_DIR"
STAGE15_STATUS=0
(
    export PYTHONPATH="$$SLIME_MEGATRON_LM"
    export WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=23457
    "$$SLIME_PYTHON" "$$SLIME_CONVERT_TO_HF" \
        --input-dir "$$SLIME_INPUT_CKPT" \
        --output-dir "$$SLIME_HF_DIR" \
        --origin-hf-dir "$$SLIME_MODEL_DIR" \
        $$SLIME_VOCAB_ARGS $$SLIME_ADD_MISSING -f
) >> "$$STAGE_LOG" 2>&1 || STAGE15_STATUS=$$?
if [[ "$$STAGE15_STATUS" -ne 0 ]]; then
    FAILURE_REASON="stage-1.5 (torch_dist->HF convert) exited with code $$STAGE15_STATUS"
    log "$$FAILURE_REASON"
    exit "$$STAGE15_STATUS"
fi
log "stage-1.5: converted HF model at $$SLIME_HF_DIR"
# --- EOS 对齐 patch (H6 不变式) ----------------------------------------------
# 转换产物继承 base 的 eos (<|endoftext|>, 151643), 但 Qwen3 chat 模板训练模型
# 终止发 <|im_end|> (151645)。serve 侧 vLLM --generation-config vllm 只认 base eos,
# 模型发 im_end 不停 -> 复读撞 max_tokens (2026-08-23 实测 slime-8b-think 50/50
# length)。从 tokenizer_config 动态找 <|im_end|> id 并进 eos_token_id (同时保留
# base eos), 与 llama-factory 对齐。非 Qwen3 无 im_end token -> 跳过。
# patch 失败只 WARN 不阻断 (训练已烧卡, 非致命; 复读只浪费 eval token, 不影响出分)。
SLIME_EOS_STATUS=0
"$$SLIME_PYTHON" - "$$SLIME_HF_DIR" <<'SLIME_EOS_PY' >> "$$STAGE_LOG" 2>&1 || SLIME_EOS_STATUS=$$?
import json, sys
from pathlib import Path
hf = Path(sys.argv[1])
tok = json.loads((hf / "tokenizer_config.json").read_text())
im_end_id = None
for tid, info in (tok.get("added_tokens_decoder") or {}).items():
    if info.get("content") == "<|im_end|>":
        im_end_id = int(tid)
        break

def _tok_id(name):
    # 从 tokenizer_config 解析特殊 token 的 id (AddedToken dict 或 str)。
    t = tok.get(name)
    if not t:
        return None
    content = t.get("content") if isinstance(t, dict) else t
    for tid, info in (tok.get("added_tokens_decoder") or {}).items():
        if info.get("content") == content:
            return int(tid)
    return None

gc_path = hf / "generation_config.json"
# 35B base (Qwen3.5-35A3B-Base) 无 generation_config.json, 转换产物随之缺失
# (2026-08-27 实测 FileNotFoundError -> WARN 不 patch, 训练模型发 im_end 不停 -> 复读
# 撞 max_tokens 出 0 分)。缺失时从 config.json / tokenizer 派生 base eos 再落盘。
if gc_path.exists():
    gc = json.loads(gc_path.read_text())
    base_eos = gc.get("eos_token_id")
else:
    gc = json.loads((hf / "config.json").read_text())
    base_eos = gc.get("eos_token_id") or _tok_id("eos_token")
if base_eos is None:
    raise SystemExit("no eos_token_id in generation_config/config/tokenizer; skip patch")
eos_list = [int(e) for e in (base_eos if isinstance(base_eos, list) else [base_eos]) if e is not None]
if im_end_id is not None and im_end_id not in eos_list:
    eos_list.append(im_end_id)
gc["eos_token_id"] = eos_list
gc_path.write_text(json.dumps(gc, indent=2))
print(f"eos align: eos_token_id={eos_list} im_end={im_end_id}")
SLIME_EOS_PY
if [[ "$$SLIME_EOS_STATUS" -ne 0 ]]; then
    log "WARN stage-1.5: eos align patch failed (rc=$$SLIME_EOS_STATUS), serve may not stop on <|im_end|>"
fi
SERVE_DIR="$$SLIME_HF_DIR"
"""

# --- epilogue: skip_eval 模式 (slime_config.skip_eval=true) -------------------
FRAGMENT_SLIME_EPILOGUE = r"""
# --- epilogue: skip_eval 模式 (slime_config.skip_eval=true) --------------------
# torch_dist ckpt 已落盘 $$CKPT_DIR; 无 eval -> eval_status=skipped, score=null.
STATUS="completed"
EVAL_STATUS="skipped"
SCORE="null"
log "slime: training completed; ckpt=$$SLIME_CKPT_DIR  eval deferred (EVAL_STATUS=skipped)"
exit 0
"""
