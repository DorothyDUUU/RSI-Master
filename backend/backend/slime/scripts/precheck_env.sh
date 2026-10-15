#!/bin/bash
# ============================================================================
# Demo: slime 环境预检（导入链 + GPU）
#
# 在 GPU 节点上执行：
#   bash scripts/precheck_env.sh
#
# 覆盖：conda slime 环境 / torch+cu129 / megatron+slime+TE+sglang 导入链 / GPU 可见性。
# 退出码：0=全部通过；1=任一检查失败。
# 调试：SKIP_GPU_CHECK=1 跳过 GPU 检测（仅用于在非 GPU 节点上预检，正常不要用）
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"

source ./activate_env.sh

python3 - <<'PY' || { echo "[FAIL] 环境预检未通过（导入链 / GPU 异常）" >&2; exit 1; }
import importlib.util, os, sys

missing = [m for m in ("torch", "megatron", "slime", "transformer_engine", "sglang")
           if importlib.util.find_spec(m) is None]
if missing:
    print(f"[FAIL] 缺少模块: {missing}", file=sys.stderr)
    sys.exit(1)

import torch
print(f"    torch: {torch.__version__}  cuda: {torch.version.cuda}")

if not torch.cuda.is_available():
    if os.environ.get("SKIP_GPU_CHECK") == "1":
        print("    [warn] SKIP_GPU_CHECK=1，跳过 GPU 检测（仅调试用）")
    else:
        print("[FAIL] torch 检测不到 CUDA/GPU —— 必须在 GPU 节点跑本脚本", file=sys.stderr)
        sys.exit(1)
else:
    print(f"    gpu x{torch.cuda.device_count()}: {torch.cuda.get_device_name(0)}")

print("[OK] 导入链 + GPU 预检通过")
PY
