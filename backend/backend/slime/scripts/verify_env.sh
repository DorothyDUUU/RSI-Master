#!/bin/bash
# ============================================================================
# Demo: 验证 slime 环境（激活 + 导入链 + GPU）
#
# 在 GPU 节点上执行：
#   bash scripts/verify_env.sh
#
# 覆盖：conda 环境 / torch+cu129 / sglang+TE+flash-attn+apex+TMS /
#       megatron+slime 导入链 / GPU 可见性。
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"

# 复用激活逻辑
source ./activate_env.sh

echo
echo "==> 1) torch / CUDA"
python3 - <<'PY'
import torch
print("    torch:", torch.__version__)
print("    cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("    gpu count:", torch.cuda.device_count())
    print("    gpu 0:", torch.cuda.get_device_name(0))
PY

echo
echo "==> 2) 训练导入链 (megatron / slime / transformer_engine / sglang)"
python3 - <<'PY'
import torch  # 必须先 import torch（fake_int4_quant_cuda 依赖）
import megatron, slime, transformer_engine, sglang
print("    megatron:", getattr(megatron, "__version__", "?"))
print("    slime:   ", getattr(slime, "__version__", "?"))
print("    TE:      ", transformer_engine.__version__)
print("    sglang:  ", sglang.__version__)
print("    OK: import chain works")
PY

echo
echo "==> 3) GPU 拓扑 (NVLink 检测，影响 NCCL_NVLS_ENABLE)"
nvidia-smi topo -m 2>/dev/null | head -20 || echo "    (nvidia-smi topo 不可用)"

echo
echo "==> 环境验证完成"
