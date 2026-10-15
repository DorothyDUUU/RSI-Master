"""Web search operations (Serper) — google_search / web_parse.

从 ``backend/huggingface_search_cli/mcp_servers/search_web.py`` 抽出 (去掉
FastMCP 装饰器, 返回解析后的 dict 而不是原始 JSON 字符串)。密钥读
``SERPER_API_KEY`` 环境变量, 请求统一走 ``ops._serper`` (带超时 + 状态码检查)。

``web_parse`` 在 Serper 抓不到时**降级为直接 ``requests`` GET**: 轨迹里 8 次
``search_web_web_parse`` 有 7 次失败 —— JSON content-type URL (HF API /
datasets-server) 被 Serper 400 拒, GitHub raw / tree 页 500 "Scraping failed",
大 parquet ``/resolve/`` 直接读超时。这些内容 Serper 抓不了, 但 ``requests``
GET 都能拿 (JSON / 纯文本 / 明确报二进制不可解析)。
"""

from __future__ import annotations

from typing import Any

from ._serper import serper_request

# 浏览器 UA —— 部分站点对裸 requests 返回 403 (Cloudflare 等)。
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

# 已知二进制 content-type: 命中即不整段下载 (parquet / 图片 / 压缩包)。
_BINARY_CTYPES = {
    "application/octet-stream",
    "application/zip",
    "application/gzip",
    "application/x-tar",
    "application/pdf",
    "application/x-parquet",
    "application/vnd.apache.parquet",
}

# 直接 GET 的读取上限: 文本页面足够, 又能挡掉大文件拖垮工具子进程。
_MAX_READ_BYTES = 16 * 1024 * 1024
# 返回给 agent 的文本上限 (trajectory 空间有限, 全文会撑爆响应)。
_MAX_TEXT_CHARS = 6000


def google_search(query: str) -> dict[str, Any]:
    """General web search via Google (Serper). Returns parsed JSON results."""
    return serper_request("google.serper.dev", "/search", {"q": query})


def web_parse(url: str) -> dict[str, Any]:
    """Fetch and extract the full text content of a specific web page.

    优先走 Serper 页面抓取 (能渲染 JS)。失败时 (JSON content-type 被拒 / GitHub
    raw 抓取 500 / 大文件超时) 降级为直接 ``requests`` GET, 按 content-type 返回
    JSON / 文本, 或明确报"二进制不可解析" —— 不再让 agent 面对 400/500 盲重试。
    """
    try:
        return serper_request("scrape.serper.dev", "/", {"url": url, "includeMarkdown": True})
    except Exception as serper_err:  # noqa: BLE001 — 任何 Serper 失败都走降级
        return _web_parse_get_fallback(url, serper_err)


def _web_parse_get_fallback(url: str, serper_err: Exception) -> dict[str, Any]:
    """Serper 失败后直接 GET ``url``, 按 content-type 决定怎么给。

    - JSON content-type → 解析成 ``{"json": {...}}`` (HF API / datasets-server);
    - 文本 → ``{"text": ...}`` 截断 (GitHub raw / HTML 页);
    - 二进制 (parquet / 图片 / zip) → ``{"error": ...}``, 不整段下载;
    - GET 本身也挂 → 错误里同时带 Serper 和 GET 的原因, 帮 agent 判断换源。
    """
    try:
        import requests

        resp = requests.get(
            url,
            headers={"User-Agent": _USER_AGENT},
            timeout=20,
            allow_redirects=True,
            stream=True,
        )
    except Exception as get_err:  # noqa: BLE001 — DNS / 连接 / TLS 都归这里
        return {
            "error": f"web_parse failed: Serper: {serper_err}; "
            f"direct GET: {type(get_err).__name__}: {get_err}"
        }

    ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    try:
        if resp.status_code >= 400:
            return {
                "error": f"web_parse failed: Serper: {serper_err}; "
                f"direct GET HTTP {resp.status_code} ({ctype})"
            }
        if not ctype or ctype in _BINARY_CTYPES or ctype.startswith("image/"):
            return {
                "error": f"URL is a binary file (Content-Type: {ctype or 'unknown'}); "
                f"web_parse can't extract text (Serper: {serper_err})"
            }
        # 用 iter_content 读前 16MB —— 自动解压 (resp.raw 是原始字节, gzip 流会
        # 被当二进制); 同时挡掉 parquet 等大文件拖垮工具子进程。
        head = b""
        for chunk in resp.iter_content(chunk_size=65536):
            head += chunk
            if len(head) >= _MAX_READ_BYTES:
                break
    except Exception as read_err:  # noqa: BLE001
        return {
            "error": f"reading {url} failed: {type(read_err).__name__}: {read_err} "
            f"(Serper: {serper_err})"
        }
    finally:
        resp.close()

    # 二进制垃圾 (raw 流被当文本) 的兜底: 前 4KB 含 null 字节多半是 parquet / zip
    text = head.decode("utf-8", errors="replace")
    if "\x00" in text[:4096]:
        return {
            "error": f"URL is a binary file (Content-Type: {ctype or 'unknown'}); "
            f"web_parse can't extract text (Serper: {serper_err})"
        }
    if "json" in ctype:
        import json as _json

        try:
            return {
                "json": _json.loads(text),
                "note": "Serper scraping failed; returned direct GET JSON",
            }
        except Exception:
            pass  # content-type 标 json 但解析失败 → 当纯文本给
    return {
        "text": text[:_MAX_TEXT_CHARS],
        "note": "Serper scraping failed; returned direct GET content",
        "truncated": len(text) > _MAX_TEXT_CHARS,
    }
