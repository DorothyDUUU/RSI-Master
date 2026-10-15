#!/bin/bash
# ============================================================================
# Demo: SFT 端到端冒烟测试（先证明全链路，再上正式训练）
#
# 模型：Qwen3-4B-Base（稠密，无 MoE / 无 --spec）
#   架构参数来自 slime 官方 spec 脚本 $WORKSPACE/slime/scripts/models/qwen3-4B.sh
#
# 用 8 条合成样本 x 1 epoch == 1 个真实训练步 + 存 ckpt，覆盖：
#   数据加载 → sft loss → forward → backward → optimizer step → checkpoint save
#
# 用法（GPU 节点）：
#   NUM_GPUS=1 TP_SIZE=1 EP_SIZE=1 HAS_NVLINK=0 bash scripts/launch_sft_smoke.sh
#
# torch_dist 权重（首次必须先转换一次，之后可复用）：
#   MODEL_DIR=<repo>/public/models/Qwen/Qwen3-4B-Base \
#   SAVE_DIR=${MODEL_DIR}_torch_dist \
#   MODEL_SPEC=$SLIME_ROOT/scripts/models/qwen3-4B.sh \
#   bash scripts/convert_hf_to_torch_dist.sh
#
# 数据生成：
#   python $SLIME_WORKSPACE/training_scripts/prepare_smoke_data.py \
#     --dst $SLIME_WORKSPACE/training_scripts/data/smoke_sft.parquet --n 8
#
# 为什么 8 条 / gbs=4：
#   (1) world_size = TP×EP（稠密模型 EP=1），Megatron 断言 gbs % (mbs*DP) == 0；
#   (2) 新 load 起 start_rollout_id=1，driver 跑 range(1, num_rollout)。
#       8 条 / rollout_batch_size=4 → num_rollout=2 → range(1,2)=1 个 iteration。
#       （4 条会得到 range(1,1) 空循环 → 0 步，静默白跑）
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"

# ---------------- CONFIG ----------------
WORKSPACE="${WORKSPACE:-}"
MODEL_SPEC="${MODEL_SPEC:-$WORKSPACE/slime/scripts/models/qwen3-4B.sh}"
MODEL_DIR="${MODEL_DIR:-}"
# 约定：torch_dist 转换产物 = ${MODEL_DIR}_torch_dist（convert_hf_to_torch_dist.sh 的 SAVE_DIR 默认值）
TORCH_DIST_DIR="${TORCH_DIST_DIR:-${MODEL_DIR}_torch_dist}"
DATA_FILE="${DATA_FILE:-$WORKSPACE/training_scripts/data/smoke_sft.parquet}"
CKPT_DIR="${CKPT_DIR:-$WORKSPACE/training_scripts/ckpt/Qwen3-4B-Base_sft_smoke}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
# 并行度（全部手动设置，不做自动检测；稠密模型默认单卡）
NUM_GPUS="${NUM_GPUS:-1}"        # ray 分配卡数 / actor 每节点卡数
TP_SIZE="${TP_SIZE:-1}"          # --tensor-model-parallel-size（GQA 约束只能 1 或 2）
EP_SIZE="${EP_SIZE:-1}"          # --expert-model-parallel-size（稠密模型固定 1；MoE 才需整除专家数）
HAS_NVLINK="${HAS_NVLINK:-0}"    # NCCL_NVLS_ENABLE：1=启用 NVLink 集合通信（NVLink 节点设 1）
# ----------------------------------------------------------------------------

# ---- 并行度校验（PP=CP=1，需 TP×EP = world_size = NUM_GPUS）----
# 放在清理进程之前：配置配错立刻退出，避免 pkill 误杀节点上其他任务
WORLD_SIZE=$((TP_SIZE * EP_SIZE))
[ "$WORLD_SIZE" -eq "$NUM_GPUS" ] || {
    echo "ERROR: TP_SIZE($TP_SIZE) × EP_SIZE($EP_SIZE) = $WORLD_SIZE ≠ NUM_GPUS($NUM_GPUS)" >&2
    exit 1
}
case "$TP_SIZE" in 1|2) ;; *) echo "WARN: TP_SIZE=$TP_SIZE 可能破坏 GQA(num-query-groups=2)，建议 1 或 2" >&2;; esac
echo "parallel: NUM_GPUS=$NUM_GPUS  TP=$TP_SIZE × EP=$EP_SIZE × PP=1 × CP=1  HAS_NVLINK=$HAS_NVLINK"

[ -f "$DATA_FILE" ]  || { echo "ERROR: $DATA_FILE missing（先跑 prepare_smoke_data.py --n 8）" >&2; exit 1; }
[ -f "$MODEL_SPEC" ] || { echo "ERROR: 模型 spec 脚本不存在: $MODEL_SPEC" >&2; exit 1; }
[ -d "$TORCH_DIST_DIR" ] || {
    echo "ERROR: $TORCH_DIST_DIR missing（训练 --load 必须指 torch_dist 产物，先转换一次）" >&2
    echo "  MODEL_DIR=$MODEL_DIR SAVE_DIR=$TORCH_DIST_DIR MODEL_SPEC=$MODEL_SPEC bash scripts/convert_hf_to_torch_dist.sh" >&2
    exit 1
}

# ---- 1) 激活环境 ----
source ./activate_env.sh

# ray 在 slime conda 环境里（$SHARED_ROOT/.../envs/slime/bin/ray），激活后必须能找到
command -v ray >/dev/null 2>&1 || {
    echo "ERROR: 未找到 ray。确认 slime 环境已激活后重试：" >&2
    echo "  source "$SLIME_ENV/../etc/profile.d/conda.sh" && conda activate slime && which ray" >&2
    exit 1
}

# ---- 2) 清理残留 (精确, 禁止 pkill -9 python 无差别清节点) ----
# 只用 ray 自己的优雅关闭清理本框架的集群, 不碰节点上其它 python 进程。
# 确需强杀残留时: 先 `ps -ef | grep <关键字>` 查清 PID, 再 `kill -9 <PID>`。
ray stop --force 2>/dev/null
sleep 3

set -ex
export PYTHONUNBUFFERED=1
export no_proxy="127.0.0.1,${MASTER_ADDR}"
export MASTER_ADDR

# ---- 模型架构（稠密 Qwen3-4B：source 官方 spec 拿 MODEL_ARGS，不含 --spec / MoE 参数）----
source "$MODEL_SPEC"

# ---- 训练参数：8 条 / 1 epoch → 恰好 1 个训练步 ----
CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}
   --load ${TORCH_DIST_DIR}
   --save ${CKPT_DIR}/
   --save-interval 1
   --no-save-optim
   --no-save-rng
)
SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data ${DATA_FILE}
   --input-key messages
   --rollout-shuffle
   --num-epoch 1
   --rollout-batch-size 4
   --global-batch-size 4
   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --log-probs-chunk-size 4096
   --debug-train-only
)
PERF_ARGS=(
   --tensor-model-parallel-size ${TP_SIZE} --sequence-parallel
   --pipeline-model-parallel-size 1 --context-parallel-size 1
   --expert-model-parallel-size ${EP_SIZE} --expert-tensor-parallel-size 1
   --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
   --use-dynamic-batch-size --max-tokens-per-gpu 4096 --seq-length 4096
)
OPTIMIZER_ARGS=(
   --optimizer adam --lr 1e-5 --lr-decay-style cosine --min-lr 1e-6
   --lr-warmup-fraction 0.1 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98
   --use-distributed-optimizer --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer --main-grads-dtype bf16
)
MISC_ARGS=(
   --attention-dropout 0.0 --hidden-dropout 0.0
   --attention-softmax-in-fp32 --attention-backend flash
)

rm -rf "${CKPT_DIR}"; mkdir -p "${CKPT_DIR}"

# ---- 启动 ray + 提交 ----
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} \
  --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${WORKSPACE}/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NVSHMEM_DISABLE_NCCL\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"no_proxy\": \"${no_proxy}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\",
    \"SLIME_MEM_DIAG\": \"1\"
  }
}"

# train_async.py 无可执行权限，entrypoint 必须用 python3 前缀（shell 直接跑裸路径会 exit 126）
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 ${WORKSPACE}/slime/train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${SFT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${MISC_ARGS[@]}

# ray job submit 返回 0 仅当训练完成；标记冒烟通过
touch "${CKPT_DIR}/SMOKE_PASSED"
echo "SMOKE PASSED -> ${CKPT_DIR}/SMOKE_PASSED"
