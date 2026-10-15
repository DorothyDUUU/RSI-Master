#!/bin/bash
# ============================================================================
# Demo: 一键启动 Qwen3.6-35B-A3B SFT 训练（slime / Megatron-LM，8×H20）
#
# 设计目标：在「相同磁盘挂载 && 未装配环境」的 GPU 节点上一条命令启动训练：
#   conda 初始化（幂等）→ 激活 slime → 启动单节点 ray cluster → 提交 train_async.py
#
# 用法（GPU 节点，如 gpu-node-0 / gpu-node）：
#   bash scripts/launch_sft.sh
#
# 换数据 / 换超参（全部支持环境变量覆盖，默认值 = 之前跑通的值）：
#   DATA_FILE=/path/to/xxx.parquet NUM_EPOCH=2 \
#   OUTPUT_DIR=/somewhere/output \
#   bash scripts/launch_sft.sh
#
# 看进度：
#   tail -f $OUTPUT_DIR/train.log
#   ray dashboard: 启动时日志打印 (随机端口, 见 "ray cluster:" 行)
#
# ⚠️ 耐心点, 别误杀: ray 集群启动 + Megatron 模型初始化 + rollout 数据准备很慢,
#   从提交到第一条 loss 日志实测 8-12 分钟甚至更久 (8B/35B 都这样)。期间 train.log
#   停在 "RolloutManager ... import ... generate_rollout" 是正常的, 不是卡死 ——
#   判断还活着看 train.log 行数在增长 / gpu_backend script-status 是 running。
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"

# ---------------- CONFIG：改这里（默认值 = 之前跑通的值） ----------------
# slime conda 环境 (仓库之外, 必须自己提供): 留空则用 PATH 上的 python3,
# LD_LIBRARY_PATH 也不注入。
SLIME_ENV="${SLIME_ENV:-}"
WORKSPACE="${WORKSPACE:-}"      # slime 源码 / Megatron-LM 所在
OUTPUT_DIR="${OUTPUT_DIR:-$WORKSPACE/backend_slime_demo/output}"  # 日志 + ckpt 产物目录

MODEL_DIR="${MODEL_DIR:-}"
TORCH_DIST_DIR="${TORCH_DIST_DIR:-}"
DATA_FILE="${DATA_FILE:-$WORKSPACE/data/xyz_aquila_sft.parquet}"
CKPT_DIR="${CKPT_DIR:-$OUTPUT_DIR/ckpt}"

# 模型架构 spec 脚本 (slime/scripts/models/*.sh)。设置时 source 它拿 MODEL_ARGS
# (稠密模型, 无 --spec / 无 MoE 参数); 留空则用下方内置的 35B MoE 默认架构。
MODEL_SPEC="${MODEL_SPEC:-}"

# 关键训练超参（默认值 = 正式跑通：3412 条 / 3 epoch / 77 步 / ~11h）
NUM_EPOCH="${NUM_EPOCH:-3}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-128}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
SEQ_LENGTH="${SEQ_LENGTH:-81920}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-81920}"
SAVE_INTERVAL="${SAVE_INTERVAL:-45}"
# the scheduler exports a job-wide MASTER_ADDR that points every pod at rank 0.  Each
# gpu-backend job owns an independent single-node Ray cluster, so bind Ray to
# the pod that is actually executing this script instead.
MASTER_ADDR="${RAY_NODE_IP:-$(hostname -I | awk '{print $1}')}"
[ -n "$MASTER_ADDR" ] || MASTER_ADDR="127.0.0.1"
# 并行度（全部手动设置，不做自动检测；默认值 = 8×H20 跑通值）
NUM_GPUS="${NUM_GPUS:-8}"        # ray 分配卡数 / actor 每节点卡数
TP_SIZE="${TP_SIZE:-2}"          # --tensor-model-parallel-size（GQA 约束只能 1 或 2）
EP_SIZE="${EP_SIZE:-4}"          # --expert-model-parallel-size（需整除专家数 256）
CP_SIZE="${CP_SIZE:-1}"          # --context-parallel-size（seq_length>4096 时建议开 CP）
HAS_NVLINK="${HAS_NVLINK:-0}"    # NCCL_NVLS_ENABLE：1=启用 NVLink 集合通信（NVLink 节点设 1）
# torch dist init_process_group 集结超时（分钟）。Co-Placement 下多 pod 同节点
# 抢 CPU, actor 导入可能被拖到 >10 分钟: rank 0 先到 init_process_group, 等 rank 1
# 加入, 默认 10 分钟 (megatron 默认 distributed_timeout_minutes=10) 会在 601s 超时
# (2026-08-27 35a3b-2gpu 实证: DistStoreError "1/2 clients joined")。默认抬到 60。
DISTRIBUTED_TIMEOUT_MINUTES="${DISTRIBUTED_TIMEOUT_MINUTES:-60}"
# optional per-experiment LR override file (worker drops into OUTPUT_DIR before stage-1).
# 只读本实验 OUTPUT_DIR —— 2026-08-28 移除 worker_0017 硬编码兜底路径: 那是 0826
# healthbench 调试遗留, 会把 SFT_LR=3e-6 泄漏到所有共享主机上的 slime run
# (L2 验证 run 031134 实测 --lr 3e-6, 而正式跑通 0.94/0.96 全用默认 1e-5)。
# 无 override 时走默认 --lr 1e-5, 与已验证配置一致。
for _f in "${OUTPUT_DIR}/sft_lr.env"; do
  if [ -f "$_f" ]; then . "$_f"; echo "[launch_sft] loaded LR override from $_f: SFT_LR=${SFT_LR:-unset} SFT_MIN_LR=${SFT_MIN_LR:-unset} SFT_WARMUP_FRAC=${SFT_WARMUP_FRAC:-unset}"; break; fi
done
# ----------------------------------------------------------------------------

# ---- ray 集群隔离（槽位化, 并行 slime 任务各占独立集群）-----------------------
# 每个实验一个独立 ray 集群 + 唯一 temp-dir。并行任务不能共享端口/ temp-dir,
# 否则第二个 ray start 会挂到第一个集群上。管线会传 RAY_TMPDIR (取自 WORKSPACE
# 哈希, 每实验唯一); 手动跑没传就用 PID 兜底。
RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_slime_$$}"
#
# 2026-08-31 重写: 从「随机端口 + pgrep 探测」改为「flock 槽位 + 确定性端口派生」。
# 背景 (HLE 0830 exp_0006 实证): 旧实现随机 8 固定端口 [20000,55000) + 随机 worker
# 区间 [MAX_FIXED+2000, +2000], worker 区间互不互斥 —— exp_0009 [49639,51639] 与
# exp_0005 [49806,51806] 重叠, exp_0006 随机端口 50926 落进 exp_0009 的区间, 整组
# 重选 5 轮全失败 → "ERROR: 多次重试后仍无法选到 8 个空闲 ray 端口" → ray 起不来。
# 且 pgrep 读其他集群命令行存在竞态 (进程 cmdline 未必即时可见)。两集群并发碰撞
# 概率 ~37%, 三集群必挂 —— 靠概率不可能支撑稳定并发。
#
# 新设计: /tmp/ray_slots 下 RAY_SLOT_COUNT=12 个 flock 锁文件, 每个集群占一个槽位
# 并持锁到脚本退出 (集群存活期独占) → 拿到的槽位 s 之后端口全部**确定性派生**,
# 槽位之间由构造保证互斥, 无随机、无 pgrep 竞态:
#   固定端口块: [20000 + 800*s, 20000 + 800*(s+1))    8 端口 @ +20 + i*100, 被外部
#     进程占用时在窗口内 +1 步进 (块内专属, 不会跨界); 块尾 29600 < worker 起点。
#   worker 区间: [30000 + 2500*s, 30000 + 2500*(s+1))  2500 宽, 槽位间无缝不重叠,
#     且固定端口块 (<=29600) 永远够不着任何 worker 区间 (>=30000)。
# 最大并行 12 集群 —— 远超 GPU 上限 (8 卡 / 每 35B 4 卡 = 2 并发), 留足余量。
# /tmp 只读时 acquire 返回 2 → 降级为旧随机+探测 (pick_ports)。
#
# ==== ray_slots BEGIN ====
# (本块为纯函数+变量默认值定义, 供 test/infra_test/test_ray_slots.sh 按标记抽取 source)
RAY_SLOT_COUNT="${RAY_SLOT_COUNT:-12}"
RAY_SLOT_DIR="${RAY_SLOT_DIR:-/tmp/ray_slots}"
RAY_SLOT=""    # 当前持有槽位 (空 = 未持有)

port_free() {
    # $1=端口号; 0=空闲, 1=占用。bind 探测 (SO_REUSEADDR 关, 防误判)。
    python3 -c 'import socket, sys
p = int(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
try:
    s.bind(("0.0.0.0", p))
    sys.exit(0)
except OSError:
    sys.exit(1)
finally:
    s.close()' "$1" 2>/dev/null
}

acquire_ray_slot() {
    # 抢占一个空闲槽位: fd 9 对该槽位锁文件持 flock 到脚本退出, 集群存活期独占。
    # 返回: 0=占槽成功 (RAY_SLOT=槽号); 1=RAY_SLOT_COUNT 槽全忙; 2=RAY_SLOT_DIR 不可写。
    local s
    if ! mkdir -p "$RAY_SLOT_DIR" 2>/dev/null; then
        echo "WARN: 无法创建 $RAY_SLOT_DIR (只读 /tmp?), 降级为随机端口+探测" >&2
        return 2
    fi
    for s in $(seq 0 $((RAY_SLOT_COUNT - 1))); do
        # 关键: exec 的重定向对 shell 是持久性的, 直接把 `2>/dev/null` 并到 exec 行
        # 会把整个脚本(含训练子进程)的 stderr 永久吞掉 (2026-08-31 仿真暴露)。
        # 必须用 group command 把 stderr 降噪作用域化: fd 9 保持打开, 但 stderr 恢复。
        if { exec 9>"$RAY_SLOT_DIR/slot_${s}.lock"; } 2>/dev/null && flock -n 9; then
            RAY_SLOT="$s"
            echo "==> ray slot: $RAY_SLOT ($RAY_SLOT_DIR)" >&2
            return 0
        fi
        { exec 9>&-; } 2>/dev/null || true   # 释放本次未加锁成功的 fd, 试下一个槽
    done
    echo "ERROR: $RAY_SLOT_COUNT 个 ray 槽位全部被占 (当前并发集群数 >= $RAY_SLOT_COUNT?)" >&2
    return 1
}

release_ray_slot() {
    # 显式释放槽位 (关闭 fd 9 = flock 释放)。脚本正常退出时 fd 自动关闭, 无需 EXIT trap。
    if [ -n "$RAY_SLOT" ]; then
        echo "[ray_slots] 释放槽位 $RAY_SLOT" >&2
        { exec 9>&-; } 2>/dev/null || true   # 同 acquire: stderr 降噪作用域化, 不吞脚本 stderr
        RAY_SLOT=""
    fi
}

derive_ports_from_slot() {
    # 从槽位 RAY_SLOT 确定性派生 8 个固定端口 + worker 区间。槽位间天然互斥:
    #   固定端口: 块内 base+20+i*100 为主选, 被外部进程占用时在窗口内 +1 步进;
    #   步进窗口 = 本端口槽位到下一个端口槽位 (i=7 时到块尾), 越界即放弃该槽位。
    #   worker 区间: [30000+2500*s, +2500), 各槽位无缝不重叠。
    local base i p end found ports
    base=$(( 20000 + RAY_SLOT * 800 ))
    RAY_GCS_PORT=""
    RAY_DASH_PORT=""
    RAY_AGENT_PORT=""
    RAY_CLIENT_PORT=""
    RAY_OBJ_PORT=""
    RAY_NODE_PORT=""
    RAY_AGENT_GRPC_PORT=""
    RAY_RUNTIME_ENV_PORT=""
    ports=""
    for (( i = 0; i < 8; i++ )); do
        p=$(( base + 20 + i * 100 ))
        end=$(( i == 7 ? base + 800 : base + 20 + (i + 1) * 100 ))
        found=0
        for (( ; p < end; p++ )); do
            if port_free "$p"; then
                found=1
                break
            fi
        done
        if [ "$found" -ne 1 ]; then
            echo "WARN: 槽位 $RAY_SLOT 固定端口窗口 [$(( base + 20 + i * 100 )), $end) 无空闲端口, 放弃该槽位" >&2
            return 1
        fi
        ports="$ports $p"
    done
    read -r RAY_GCS_PORT RAY_DASH_PORT RAY_AGENT_PORT RAY_CLIENT_PORT \
           RAY_OBJ_PORT RAY_NODE_PORT RAY_AGENT_GRPC_PORT RAY_RUNTIME_ENV_PORT <<<"$ports"
    RAY_MIN_WORKER=$(( 30000 + RAY_SLOT * 2500 ))
    RAY_MAX_WORKER=$(( RAY_MIN_WORKER + 2500 - 1 ))
    return 0
}

pick_ports() {
    # 降级路径 (RAY_SLOT_DIR 不可写时用): 选 8 个互异且当前未被占用的随机固定端口。
    # $1 = 整组重选轮数 (任一端口被占就整组重选, 不再逐个补)。
    local tries="$1" used p ok
    while [ "$tries" -gt 0 ]; do
        used=" "
        ok=1
        for _ in $(seq 1 8); do
            while :; do
                p=$(( 20000 + RANDOM % 35000 ))
                case "$used" in *" $p "*) continue;; esac
                break
            done
            if ! port_free "$p"; then
                echo "WARN: ray 端口 $p 被占用, 整组重选 ..." >&2
                ok=0; break
            fi
            used="$used$p "
        done
        if [ "$ok" -eq 1 ]; then
            read -r RAY_GCS_PORT RAY_DASH_PORT RAY_AGENT_PORT RAY_CLIENT_PORT \
                   RAY_OBJ_PORT RAY_NODE_PORT RAY_AGENT_GRPC_PORT RAY_RUNTIME_ENV_PORT <<<"$used"
            # worker 区间起点必须高于所有固定端口, 否则 [min, min+2000] 会包住某个固定端口。
            MAX_FIXED=$(printf '%s\n' "$RAY_GCS_PORT" "$RAY_DASH_PORT" "$RAY_AGENT_PORT" "$RAY_CLIENT_PORT" \
                        "$RAY_OBJ_PORT" "$RAY_NODE_PORT" "$RAY_AGENT_GRPC_PORT" "$RAY_RUNTIME_ENV_PORT" | sort -n | tail -1)
            RAY_MIN_WORKER=$(( MAX_FIXED + 2000 + RANDOM % 2000 ))
            RAY_MAX_WORKER=$(( RAY_MIN_WORKER + 2000 ))
            return 0
        fi
        tries=$((tries - 1))
        sleep 1
    done
    return 1
}
# ==== ray_slots END ====

# ---- 并行度校验（PP=1, 需 TP×EP×CP = world_size = NUM_GPUS）----
# 放在清理进程之前：配置配错立刻退出，避免 pkill 误杀节点上其他任务
WORLD_SIZE=$((TP_SIZE * EP_SIZE * CP_SIZE))
[ "$WORLD_SIZE" -eq "$NUM_GPUS" ] || {
    echo "ERROR: TP_SIZE($TP_SIZE) × EP_SIZE($EP_SIZE) × CP_SIZE($CP_SIZE) = $WORLD_SIZE ≠ NUM_GPUS($NUM_GPUS)" >&2
    exit 1
}
case "$TP_SIZE" in 1|2) ;; *) echo "WARN: TP_SIZE=$TP_SIZE 可能破坏 GQA(num-query-groups=2)，建议 1 或 2" >&2;; esac
echo "parallel: NUM_GPUS=$NUM_GPUS  TP=$TP_SIZE × EP=$EP_SIZE × CP=$CP_SIZE × PP=1  HAS_NVLINK=$HAS_NVLINK"

# ---- 0) 前置依赖检查（提前暴露，避免跑一半才报错）----
[ -f "$DATA_FILE" ]      || { echo "ERROR: 训练数据不存在: $DATA_FILE" >&2; exit 1; }
[ -d "$TORCH_DIST_DIR" ] || { echo "ERROR: torch_dist 权重不存在: $TORCH_DIST_DIR（先跑 convert_hf_to_torch_dist.sh）" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR"

# ---- 1) 激活 slime 环境（内部已处理 conda 初始化）----
source ./activate_env.sh

# The shared slime environment has repeatedly been observed mid-update on
# shared storage (missing transformers model packages / ESTALE).  Use a validated,
# versioned Transformers overlay so training never depends on that mutable
# package tree.  The overlay is built once by aliyun_submit/env_smoke_0918.
SLIME_TRANSFORMERS_OVERLAY="${SLIME_TRANSFORMERS_OVERLAY:-}"
[ -f "$SLIME_TRANSFORMERS_OVERLAY/READY" ] || {
    echo "ERROR: validated Transformers overlay is unavailable: $SLIME_TRANSFORMERS_OVERLAY" >&2
    exit 1
}
export PYTHONPATH="$SLIME_TRANSFORMERS_OVERLAY:$WORKSPACE/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}"

# Fail before Ray actors reserve GPUs if the exact import path used by slime
# is unhealthy.  This also prevents NumPy 2.x from reaching Megatron's actors.
python3 - <<'PY'
import numpy as np
if not np.__version__.startswith("1."):
    raise RuntimeError(f"Megatron requires NumPy 1.x, got {np.__version__}")
import transformers
from transformers import AutoConfig
import slime.backends.megatron_utils.initialize  # noqa: F401
print(f"[env-preflight] numpy={np.__version__} transformers={transformers.__version__} imports=ok")
PY

# ---- 2) 重跑卫生：清理残留 ray 集群（精确, 禁止 pkill -9 python 无差别清节点）----
# 只清本实验的集群 (temp-dir 唯一)。ray stop 无地址限定, 会杀掉节点上所有并行
# 集群 —— 并行任务必须用 pkill -f <temp-dir> 精确清理 (ray 进程 cmdline 含 temp-dir)。
echo "==> 清理残留 ray 集群 (${RAY_TMPDIR}) ..."
pkill -9 -f "${RAY_TMPDIR}" 2>/dev/null || true
sleep 3

set -ex
export PYTHONUNBUFFERED=1
export no_proxy="127.0.0.1,${MASTER_ADDR}"
export MASTER_ADDR

# ---- 4) 模型架构参数 ----
# 显式给了 MODEL_SPEC (slime/scripts/models/*.sh, 稠密模型) → source 它拿官方
# MODEL_ARGS (无 --spec / 无 MoE 参数)。留空 → 内置 35B MoE 默认 (原 qwen3.5 spec)。
if [ -n "$MODEL_SPEC" ]; then
   [ -f "$MODEL_SPEC" ] || { echo "ERROR: 模型 spec 脚本不存在: $MODEL_SPEC" >&2; exit 1; }
   echo "==> 使用模型架构 spec: $MODEL_SPEC"
   source "$MODEL_SPEC"
else
NLAYERS=40
arr=()
for ((i=0; i<NLAYERS; i++)); do arr+=(1); done
printf -v MOE_LAYER_FREQ "[%s]" "$(IFS=', '; echo "${arr[*]}")"

MODEL_ARGS=(
   --spec "slime_plugins.models.qwen3_5" "get_qwen3_5_spec"

   --disable-bias-linear
   --qk-layernorm
   --group-query-attention
   --num-attention-heads 16
   --num-query-groups 2
   --kv-channels 256
   --num-layers 40
   --hidden-size 2048
   --ffn-hidden-size 512
   --use-gated-attention

   --normalization RMSNorm
   --apply-layernorm-1p
   --position-embedding-type rope
   --norm-epsilon 1e-6
   --rotary-percent 0.25
   --swiglu
   --untie-embeddings-and-output-weights
   --vocab-size 248320

   --rotary-base 10000000

   # moe (256 experts, 8 topk)
   --moe-ffn-hidden-size 512
   --moe-shared-expert-intermediate-size 512
   --moe-router-score-function softmax
   --moe-token-dispatcher-type alltoall
   --moe-router-topk 8
   --moe-layer-freq "$MOE_LAYER_FREQ"
   --num-experts 256
   --moe-grouped-gemm
   --moe-token-drop-policy probs
   --moe-router-dtype fp32
   --moe-permute-fusion
   --moe-aux-loss-coeff 0

   # qwen3.5/3.6 specific
   --attention-output-gate
   --moe-shared-expert-gate
)
fi

# ---- 5) checkpoint 参数 ----
CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}
   --load ${TORCH_DIST_DIR}        # torch_dist 加载时按 TP/EP/PP 自动重分片
   --save ${CKPT_DIR}/
   --save-interval ${SAVE_INTERVAL}
   --no-save-optim
   --no-save-rng
)

# ---- 6) SFT 参数（--debug-train-only 是离线 SFT 路径：只跳过 SGLang 引擎，不跳过训练）----
SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data ${DATA_FILE}
   --input-key messages
   --metadata-key metadata
   --rollout-shuffle
   --num-epoch ${NUM_EPOCH}
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
   --global-batch-size ${GLOBAL_BATCH_SIZE}

   --loss-type sft_loss
   --loss-mask-type qwen3_5
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --log-probs-chunk-size 1024
   --debug-train-only
)

# ---- 7) 并行度（默认 TP=2 / EP=4；用 TP_SIZE / EP_SIZE 环境变量覆盖）----
PERF_ARGS=(
   --tensor-model-parallel-size ${TP_SIZE}
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size ${CP_SIZE}
   --expert-model-parallel-size ${EP_SIZE}
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU}
   --seq-length ${SEQ_LENGTH}
)

# ---- 8) 优化器（内存修复：bf16 梯度缓冲 / CPU offload）----
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr ${SFT_LR:-1e-5}
   --lr-decay-style cosine
   --min-lr ${SFT_MIN_LR:-1e-6}
   --lr-warmup-fraction ${SFT_WARMUP_FRAC:-0.1}
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --use-distributed-optimizer
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
   --main-grads-dtype bf16
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --attention-softmax-in-fp32
   --attention-backend flash
)

# ---- 9) 启动单节点 ray head，提交训练 ----
# 35B MoE full-SFT (Adam fp32 + CPU offload) 4 卡峰值 ~843GiB RAM, 本 pod cgroup
# 上限 880GiB (944892805120B)。Ray 默认 95% (836GiB) 就杀 worker, 即使实际用量仍在
# cgroup 硬限内 —— 放宽到 99% (871GiB) 让收尾 save_model 不被误杀 (2026-08-23 实证)。
export RAY_memory_usage_threshold=0.99
# ray start 重试: 槽位端口已探测空闲, 但检查到启动之间有极小 TOCTOU 窗口 (并发集群
# worker 刚好在窗口内绑定我们选的端口)。GCS 被占 → ray start 非零退出; dashboard 被占
# → ray start 静默换端口但 exit 0 → 后续 job submit 打到错误地址。两种情况都清理本
# 集群 (pkill 精确按 temp-dir, 禁止裸 ray stop --force —— 它会杀掉同 pod 所有并行
# 集群) + 释放槽位换新槽重试 (换槽即换整组互斥端口)。探活以 ray job list 连上
# RAY_DASH_PORT 为准。
start_ok=0
degraded=0
for attempt_ray in 1 2 3; do
  if [ "$degraded" -eq 1 ]; then
    # 降级路径 (RAY_SLOT_DIR 不可写): 沿用旧随机+探测
    if ! pick_ports 5; then
      echo "ERROR: 多次重试后仍无法选到 8 个空闲 ray 端口 (pod 端口被占满?)" >&2
      exit 1
    fi
  else
    release_ray_slot   # 上一轮失败释放; 第一轮无持有 → no-op
    # acquire 返回非零时 (槽满=1 / 不可写=2) 必须在 if 上下文捕获 —— set -e 下裸
    # 调用会在 `rc=$?` 赋值前退出, 降级/耗尽分支会变死代码 (2026-08-31 测试暴露)。
    if acquire_ray_slot; then
      acquire_rc=0
    else
      acquire_rc=$?
    fi
    if [ "$acquire_rc" -eq 2 ]; then
      echo "==> WARN: $RAY_SLOT_DIR 不可写, 降级为随机端口+探测"
      degraded=1
      if ! pick_ports 5; then
        echo "ERROR: 多次重试后仍无法选到 8 个空闲 ray 端口 (pod 端口被占满?)" >&2
        exit 1
      fi
    elif [ "$acquire_rc" -ne 0 ]; then
      echo "ERROR: ray 槽位耗尽 (${RAY_SLOT_COUNT} 个并发集群已占满?)" >&2
      exit 1
    elif ! derive_ports_from_slot; then
      echo "WARN: 槽位 $RAY_SLOT 固定端口块无空闲, 释放换新槽重试 ..." >&2
      release_ray_slot
      continue
    fi
  fi
  echo "==> ray cluster (try $attempt_ray): gcs=${RAY_GCS_PORT} dashboard=${RAY_DASH_PORT} agent=${RAY_AGENT_PORT} client=${RAY_CLIENT_PORT} obj=${RAY_OBJ_PORT} node=${RAY_NODE_PORT} grpc=${RAY_AGENT_GRPC_PORT} rt=${RAY_RUNTIME_ENV_PORT} worker=${RAY_MIN_WORKER}-${RAY_MAX_WORKER} tempdir=${RAY_TMPDIR}"
  if ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} \
      --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=${RAY_DASH_PORT} \
      --port=${RAY_GCS_PORT} --temp-dir="${RAY_TMPDIR}" \
      --dashboard-agent-listen-port=${RAY_AGENT_PORT} \
      --ray-client-server-port=${RAY_CLIENT_PORT} \
      --min-worker-port=${RAY_MIN_WORKER} --max-worker-port=${RAY_MAX_WORKER} \
      --object-manager-port=${RAY_OBJ_PORT} \
      --node-manager-port=${RAY_NODE_PORT} \
      --dashboard-agent-grpc-port=${RAY_AGENT_GRPC_PORT} \
      --runtime-env-agent-port=${RAY_RUNTIME_ENV_PORT}; then
    # start 成功也要确认 job agent 在 RAY_DASH_PORT 可达 (并行集群刚起时 agent 未
    # 注册会报 HTTP 500, 等最多 60s); 一直连不上说明 dashboard 端口被抢占 (TOCTOU),
    # 换端口重试。
    dash_ok=0
    for _ in $(seq 1 30); do
      if ray job list --address="http://127.0.0.1:${RAY_DASH_PORT}" >/dev/null 2>&1; then
        dash_ok=1
        break
      fi
      sleep 2
    done
    if [ "$dash_ok" -eq 1 ]; then
      start_ok=1
      break
    fi
    echo "==> WARN: ray dashboard ${RAY_DASH_PORT} 探测不可达 (端口被并发集群抢占?), 清理 + 换槽重试 ..."
  else
    echo "==> WARN: ray start --head 失败 (GCS 端口被占用?), 清理 + 换槽重试 ..."
  fi
  pkill -9 -f "${RAY_TMPDIR}" 2>/dev/null || true
  sleep 3
done
if [ "$start_ok" -ne 1 ]; then
  echo "ERROR: ray 集群启动失败 (3 次尝试, 端口持续被占)" >&2
  exit 1
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${SLIME_TRANSFORMERS_OVERLAY}:${WORKSPACE}/Megatron-LM/\",
    \"LD_LIBRARY_PATH\": \"${SLIME_ENV:+$SLIME_ENV/lib:}${LD_LIBRARY_PATH:-}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NVSHMEM_DISABLE_NCCL\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"no_proxy\": \"${no_proxy}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\",
    \"SLIME_MEM_DIAG\": \"1\"
  }
}"

echo "==> 训练数据: $DATA_FILE"
echo "==> 输出 ckpt: $CKPT_DIR/"
echo "==> 训练日志: $OUTPUT_DIR/train.log"

# train_async.py 无可执行权限，entrypoint 必须用 python3 前缀（shell 直接跑裸路径会 exit 126）
# ray job submit 有竞态: 并行集群刚起时 job agent 未注册, 会报 HTTP 500 "No available
# agent to submit job" (2026-08-23 两个 8B 并行时实证)。job agent 就绪探活已在上面
# ray start 重试里做掉 (dash_ok 探测 RAY_DASH_PORT), 这里只留提交级重试。
#
# 2026-08-26 修复 (WebSocket 1006 缺口, gpqa-35b 0826 exp_0004 实证): 旧实现
# `ray job submit` 默认流式跟日志 —— log-tail 的 WebSocket 在 pod 资源紧张/网络闪断时
# close 1006 (abnormal closure), submit 进程带着非零退出码结束, 而 ray job 本身还在
# 正常训练。旧逻辑把 submit 的 exit code 当成训练结果, 见 train.log 有 "Running
# entrypoint" 就 break → exit 1 → stage-1 判失败 → 集群被清掉, 白烧 27min。新实现:
#   1) 提交加 --no-wait: submit 只做 HTTP 提交、立刻返回 job id, 不流式跟日志 → 1006
#      在提交阶段无从发生;
#   2) 训练日志用 ray SDK 快照式拉取 (单次 HTTP, 同样 1006 免疫), 写回 train.log;
#   3) 训练成败由 ray job 终态 status + driver_exit_code 判定, 与日志流断连完全解耦。

submit_sft() {
  # --no-wait: 提交后立刻返回并拿到 job id, 不流式跟日志 (log-tail WebSocket 1006 免疫)
  local submit_out rc
  submit_out=$(ray job submit --no-wait \
     --address="http://127.0.0.1:${RAY_DASH_PORT}" \
     --runtime-env-json="${RUNTIME_ENV_JSON}" \
     -- ${SLIME_ENV:+$SLIME_ENV/bin/}python3 ${WORKSPACE}/slime/train_async.py \
     --actor-num-nodes 1 \
     --actor-num-gpus-per-node ${NUM_GPUS} \
     --distributed-timeout-minutes ${DISTRIBUTED_TIMEOUT_MINUTES} \
     ${MODEL_ARGS[@]} \
     ${CKPT_ARGS[@]} \
     ${SFT_ARGS[@]} \
     ${OPTIMIZER_ARGS[@]} \
     ${PERF_ARGS[@]} \
     ${MISC_ARGS[@]} 2>&1)
  rc=$?
  printf '%s\n' "$submit_out" | tee -a "$OUTPUT_DIR/train.log"
  if [ "$rc" -ne 0 ]; then
    return "$rc"
  fi
  # 成功输出形如 "Job 'raysubmit_xxx' submitted successfully"
  RAY_JOB_ID=$(printf '%s\n' "$submit_out" | grep -oE 'raysubmit_[A-Za-z0-9_-]+' | head -1)
  if [ -z "$RAY_JOB_ID" ]; then
    echo "ERROR: ray job submit 成功但解析不到 job id (输出: $submit_out)" >&2
    return 2
  fi
  echo "$RAY_JOB_ID" > "$OUTPUT_DIR/ray_job_id"
  return 0
}

poll_job() {
  # 轮询 ray job 终态 + 快照式拉日志 (ray JobSubmissionClient SDK, 纯 HTTP 无 WebSocket):
  # 训练成败看终态 status / driver_exit_code, 日志流断连不会误杀仍在训练的任务。
python3 - "${RAY_DASH_PORT}" "${RAY_JOB_ID}" "$OUTPUT_DIR/train.log" <<'PYEOF'
import sys, time
from ray.job_submission import JobSubmissionClient
port, job_id, log_path = sys.argv[1], sys.argv[2], sys.argv[3]
TERMINAL = {"SUCCEEDED", "FAILED", "STOPPED"}
final = None
last_code = None
empty_streak = 0
client = None
for _ in range(7200):          # 7200 × 15s ≈ 30h 上限, 覆盖超长训练
    try:
        # 构造函数也要进 try: JobSubmissionClient 初始化会 GET /api/version 做连接
        # 检查, dashboard 瞬时不达同样抛异常 —— 一并按 empty_streak 重试, 不误杀
        if client is None:
            client = JobSubmissionClient(f"http://127.0.0.1:{port}")
        info = client.get_job_info(job_id)
        empty_streak = 0
    except Exception as e:
        empty_streak += 1
        if empty_streak >= 60:  # 连续 ~15min 不可达 (dashboard 崩了), 判失败
            print(f"WARN: ray job status 连续 60 次不可达: {e}", file=sys.stderr)
            final = "FAILED"
            break
        time.sleep(15)
        continue
    status = info.status or "UNKNOWN"
    last_code = info.driver_exit_code
    try:  # 快照式拉日志 (单次 HTTP, 失败静默等下一轮)
        logs = client.get_job_logs(job_id)
        if logs:
            with open(log_path, "w") as f:
                f.write(logs)
    except Exception:
        pass
    if status in TERMINAL:
        final = status
        break
    time.sleep(15)
if final is None:
    print("WARN: 轮询超时未到终态, 判训练失败", file=sys.stderr)
    final = "FAILED"
print(f"==> ray job {job_id} 终态: {final} (driver_exit_code={last_code})")
sys.exit(0 if final == "SUCCEEDED" else 1)
PYEOF
}

status=1
for attempt in 1 2 3; do
  echo "==> ray job submit (attempt $attempt)"
  submit_sft
  rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "==> ray job ${RAY_JOB_ID} 提交成功, 轮询训练状态 ..."
    if poll_job; then
      status=0
    else
      status=1
    fi
    break
  fi
  if [ "$rc" -eq 2 ]; then
    # job 已提交但解析不到 id —— 绝不能重试, 会重复提交第二份训练
    echo "ERROR: job 已提交但解析不到 job id, 放弃重试避免重复提交" >&2
    status=1
    break
  fi
  status=$rc
  echo "==> 提交级失败 (exit $status, job agent 未就绪?), 5s 后重试 ..."
  sleep 5
done
echo "==> 训练结束 (exit $status)"
exit "$status"
