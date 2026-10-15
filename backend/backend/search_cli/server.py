"""FastAPI server exposing the search_cli operations over HTTP.

``ops.execute`` 的薄 HTTP 壳: ``POST /search`` 收 ``{op, params}``, 在
线程池里跑同步 op, 返回 ``{"ok": True, "result": ...}``。未知 op → 400,
op 意外抛异常 → 500。

启动方式::

    python search_cli/main.py server start --port 8991
    # 或直接:
    uvicorn search_cli.server:app --host 0.0.0.0 --port 8991
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .ops import execute
from .req_log import log_request_record

logger = logging.getLogger(__name__)

app = FastAPI(
    title="search_cli Backend",
    description="Unified search backend (HuggingFace / web / github / scholar)",
    version="0.1.0",
)


# ── Request-level latency logging (traj_analysis T8) ─────────────────
# 每个非 /health POST 记一行 {ts, endpoint, method, op, query, latency_ms,
# status_code}。latency_ms 含 server 侧排队 + 实际 ops.execute —— 是
# "agent 等搜索结果多久" 的权威来源。落点由 SEARCH_REQ_LOG env 控制
# (submit 脚本设 SEARCH_REQ_LOG=${POD_LOG}/search_requests.jsonl)。
# CLI 本地回退路径 (server 不可达) 由 cli.py::_dispatch 记 via=local 行。


@app.middleware("http")
async def _request_logging(request, call_next):
    path = request.url.path
    if request.method != "POST" or path == "/health":
        return await call_next(request)
    ts = time.time()
    op = ""
    query = ""
    dataset_id = ""
    try:
        body = await request.body()  # Starlette caches; sync endpoint re-reads fine
        if body:
            data = json.loads(body)
            op = str(data.get("op", ""))[:80]
            params = data.get("params") or {}
            query = str(
                params.get("query") or params.get("keyword") or params.get("url") or ""
            )[:200]
            dataset_id = str(
                params.get("dataset_id") or params.get("owner") or ""
            )[:120]
    except Exception:  # noqa: BLE001 — 记日志绝不影响请求
        pass
    response = await call_next(request)
    log_request_record(
        {
            "ts": round(ts, 3),
            "endpoint": path,
            "method": request.method,
            "op": op,
            "query": query,
            "dataset_id": dataset_id,
            "latency_ms": round((time.time() - ts) * 1000, 1),
            "status_code": response.status_code,
            "response_bytes": int(response.headers.get("content-length") or -1),
        }
    )
    return response


class SearchRequest(BaseModel):
    """Body of ``POST /search``."""

    op: str = Field(description="Operation name, e.g. 'search_datasets'.")
    params: dict[str, Any] = Field(default_factory=dict)


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": "search-cli", "version": "0.1.0"}


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "healthy", "service": "search-cli", "version": "0.1.0"}


@app.post("/search")
def search(req: SearchRequest) -> dict[str, Any]:
    """Run a search op. Sync endpoint → FastAPI runs it in a threadpool."""
    try:
        result = execute(req.op, req.params)
    except ValueError as exc:
        # 未知 op 等参数性问题 → 客户端可判定的 4xx。写 op (download/materialize) 的
        # 相对 output_dir 由 ops.hf._write_root 在此抛 ValueError → 同样落 400
        # (不变量: 写路径绝不按 server 自身 cwd=backend/backend 解析, 见 _write_root)。
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — 把意外失败兜住, 不外泄 traceback
        logger.exception("search op %r failed", req.op)
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    return {"ok": True, "result": result}
