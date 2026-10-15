"""Shared Serper HTTP helper for ``ops.web`` and ``ops.scholar``.

Both web search (``google.serper.dev``) and Google Scholar
(``google.serper.dev/scholar``) plus page scraping (``scrape.serper.dev``)
hit the same Serper API, so the key resolution and the request/timeout/status
handling are consolidated here instead of being duplicated per module.
Key read from ``SERPER_API_KEY`` env var.
"""

from __future__ import annotations

import http.client
import json
import os
from typing import Any


def require_key() -> str:
    key = os.getenv("SERPER_API_KEY", "")
    if not key:
        raise ValueError("SERPER_API_KEY environment variable is required")
    return key


def serper_request(
    host: str,
    path: str,
    payload: dict[str, Any],
    *,
    timeout: int = 20,
) -> dict[str, Any]:
    """POST ``payload`` to Serper ``host``/``path`` and return parsed JSON.

    Enforces an HTTP timeout (Serper previously was called with no timeout at
    all, so a hung upstream could block forever) and raises on non-2xx
    responses instead of blindly ``json.loads``-ing the body. The connection
    is always closed.
    """
    conn = http.client.HTTPSConnection(host, timeout=timeout)
    try:
        headers = {
            "X-API-KEY": require_key(),
            "Content-Type": "application/json",
        }
        conn.request("POST", path, json.dumps(payload), headers)
        resp = conn.getresponse()
        body = resp.read().decode("utf-8")
        if resp.status >= 400:
            raise ValueError(f"Serper {host} HTTP {resp.status}: {body[:300]}")
        return json.loads(body)
    finally:
        conn.close()
