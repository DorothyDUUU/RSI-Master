"""GPU-side self-contained loader for the canonical decontamination module.

去污逻辑单一来源 = ``tools/ExpOS/expos/service/decontamination.py``(controller
侧 ``WorkerService.submit`` 与 serverless GPU stage-0 共用同一份; 本仓库不再
维护逻辑 fork)。本文件只是把 canonical 以**独立文件**方式按路径加载
(importlib spec, 不触发 ``expos/__init__`` 的 service+backends 重 import 链),
供同目录 ``decontam_stage.py`` 在 GPU 节点上用 heavy python (llama_factory_env
/ slime env, 有 pyarrow) 直接跑文件路径、不依赖 repo 在 PYTHONPATH 上。

canonical 模块仅用 stdlib (pyarrow 在调用时才惰性 import), 可安全独立 exec。
公开 API (8 个符号) 与旧 fork 保持一致, 语义为 canonical 的超集
(extra bench 源 healthbench_professional/bbeh/medxpertqa/arenahardwriting/
humaneval + stale-symlink 容错 + healthbench_easy 修好的活动源)。
"""

import importlib.util
import sys
from pathlib import Path

# <repo>/backend/backend/submit_pipeline/decontaminate.py -> parents[3] = <repo>。
# 与 decontam_stage.py 用 dirname(__file__) 找同目录模块是同一类"同树自定位":
# GPU 侧 stage 本就要求全量 repo (llama_factory / slime 脚本 / datasets) 挂在该根上。
_CANONICAL = (
    Path(__file__).resolve().parents[3]
    / "tools"
    / "ExpOS"
    / "expos"
    / "service"
    / "decontamination.py"
)
if not _CANONICAL.is_file():
    raise ImportError(
        f"canonical decontamination module not found: {_CANONICAL}. "
        "Expected the shared RSI-Master tree (tools/ExpOS under the repo root)."
    )

_spec = importlib.util.spec_from_file_location(
    "_decontamination_canonical_gpu", _CANONICAL
)
assert _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
# 先注册再 exec: canonical 顶层用 @dataclass, 需要 sys.modules[name] 就位才能解析。
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

DecontaminationReport = _mod.DecontaminationReport
normalize_query = _mod.normalize_query
training_query = _mod.training_query
load_benchmark_queries = _mod.load_benchmark_queries
decontaminate_jsonl = _mod.decontaminate_jsonl
read_parquet_rows = _mod.read_parquet_rows
training_query_from_messages = _mod.training_query_from_messages
decontaminate_parquet = _mod.decontaminate_parquet

__all__ = [
    "DecontaminationReport",
    "normalize_query",
    "training_query",
    "load_benchmark_queries",
    "decontaminate_jsonl",
    "read_parquet_rows",
    "training_query_from_messages",
    "decontaminate_parquet",
]
