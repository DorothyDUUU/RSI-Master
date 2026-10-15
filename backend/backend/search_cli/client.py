"""Synchronous HTTP client for the search_cli server (front/back separation).

CLI 端通过这个 client 对 server 的 API 端口发请求。``health_check()`` 是探测
语义: 任何失败都返回 ``None`` (而不是抛异常), 方便 CLI 判定是否回退到本地执行。
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class SearchError(RuntimeError):
    """Raised when the search server HTTP call fails outright (network/HTTP 5xx)."""


# 瞬态连接级错误: 服务端中途断连 / 连接被重置 / 远端主动关流。这类错误重试一次
# 大概率能撞上恢复的连接 (轨迹里 6 次 "Server disconnected without sending a
# response", 全部是搜索服务进程瞬时抖动, 重试即过)。**不包含超时** —— 请求级
# 超时重试会翻倍成本, 不是瞬态。
_TRANSIENT_DISCONNECT_SIGS = (
    "server disconnected without sending a response",
    "connection reset",
    "remote end closed connection",
    "connection closed unexpectedly",
)


class SearchClient:
    """HTTP client for the search_cli server."""

    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # base_url must be on the httpx.Client so relative paths like "/health"
        # resolve against it — otherwise every request fails with "missing
        # http:// protocol" and health_check() would always report "down".
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "SearchClient":  # pragma: no cover
        return self

    def __exit__(self, *_: Any) -> None:  # pragma: no cover
        self.close()

    # ── health probe ────────────────────────────────────────────────────────

    def health_check(self, timeout: float | None = None) -> dict[str, Any] | None:
        """Probe ``GET /health``.

        Returns the response body on success, ``None`` on any failure
        (connection refused, timeout, non-2xx). Semantics = "probe failed".
        """
        try:
            resp = self._client.get("/health", timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            logger.debug("health probe failed for %s: %s", self.base_url, exc)
            return None

    # ── search ──────────────────────────────────────────────────────────────

    def search(self, op: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Fire ``POST /search`` and return the server body.

        Returns ``{"ok": True, "result": <ops result dict>}`` on success;
        raises :class:`SearchError` on network/HTTP failure so the caller can
        decide whether to fall back.

        瞬态连接级错误 (服务端中途断连等) 重试一次; HTTP 错误 / 超时不上送重试。
        """
        body = {"op": op, "params": params or {}}
        last_err: httpx.RequestError | None = None
        for attempt in range(2):
            try:
                resp = self._client.post("/search", json=body)
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as exc:
                logger.error("HTTP %s /search: %s", exc.response.status_code, exc.response.text)
                raise SearchError(exc.response.text[:400]) from exc
            except httpx.RequestError as exc:
                logger.error("request error /search: %s", exc)
                last_err = exc
                if attempt == 0 and _is_transient_disconnect(exc):
                    time.sleep(0.5)  # 等服务端恢复; 1 次足够, 不拖慢正常路径
                    continue
                break
        assert last_err is not None
        raise SearchError(str(last_err)) from last_err


def _is_transient_disconnect(exc: httpx.RequestError) -> bool:
    msg = str(exc).lower()
    return any(sig in msg for sig in _TRANSIENT_DISCONNECT_SIGS)
