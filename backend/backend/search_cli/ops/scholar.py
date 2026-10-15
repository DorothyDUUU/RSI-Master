"""Scholarly search operations — arxiv / google_scholar / dblp.

从 ``backend/huggingface_search_cli/mcp_servers/search_scholar.py`` 移植 (去掉
FastMCP 装饰器, 把 arxiv 和 dblp 分别收敛成带 ``by``/``kind`` 参数的函数)。
Serper 密钥读 ``SERPER_API_KEY`` 环境变量 (请求走 ``ops._serper``)。

dblp 用同步 httpx (带指数退避重试) —— 之前用 ``asyncio.run``, 在已运行的
事件循环里被调用会抛 ``RuntimeError``。
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Union

import httpx

from ._serper import serper_request


# ── arxiv ────────────────────────────────────────────────────────────

def _format_arxiv_results(results: Any) -> List[Dict]:
    formatted = []
    for result in results:
        formatted.append(
            {
                "标题": result.title,
                "作者": [a.name for a in result.authors],
                "发表日期": result.published.strftime("%Y-%m-%d"),
                "摘要开头": result.summary.replace("\n", " "),
                "URL": result.entry_id,
                "pdf": str(result.entry_id).replace("abs", "pdf"),
            }
        )
    return formatted


def arxiv_search(query: str, by: str = "content", max_results: int = 5) -> List[Dict]:
    """Search arXiv by content (default) or by author.

    by="content": query_string (title/abstract/subject keywords)
    by="author":  author name, e.g. "Geoffrey Hinton"
    """
    try:
        import arxiv  # noqa: PLC0415 — optional dep: only needed for this verb
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ValueError(
            "the 'arxiv' package is not installed; run scholar-arxiv from an env "
            "that has it (e.g. the repo .venv)"
        ) from exc
    client = arxiv.Client()
    if by == "author":
        search = arxiv.Search(
            query=f"au:{query}",
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )
    else:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
    return _format_arxiv_results(client.results(search))


# ── google scholar ───────────────────────────────────────────────────

def google_scholar_search(query: str) -> Dict:
    """Perform a google scholar search (Serper /scholar). Returns parsed JSON."""
    return serper_request("google.serper.dev", "/scholar", {"q": query})


# ── dblp ─────────────────────────────────────────────────────────────


def _safe_str(val: Union[str, List, None]) -> str:
    """Helper to handle DBLP's inconsistent XML-to-JSON list/string conversion."""
    if val is None:
        return ""
    if isinstance(val, list):
        return str(val[0]) if val else ""
    return str(val)


def _fetch_dblp(api_url: str, query: str, max_results: int) -> List[Dict]:
    """Fetch DBLP JSON with exponential backoff retry (sync).

    原来是 async 函数 + ``asyncio.run`` 调用; 改成同步 httpx 后, 无论从同步
    (CLI/threadpool) 还是异步上下文调用都不会炸 ``asyncio.run`` 的 loop 冲突。
    """
    max_retries = 3
    base_delay = 1.0
    params = {"q": query, "h": max_results, "format": "json"}

    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=20.0) as client:
                response = client.get(api_url, params=params)
                response.raise_for_status()
                data = response.json()
                try:
                    return data["result"]["hits"]["hit"]
                except (KeyError, TypeError):
                    return []
        except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)

    return []


def _format_dblp_papers(hits: List[Dict], query: str) -> str:
    if not hits:
        return f"No papers found for query: '{query}'"

    formatted_output = []
    for hit in hits:
        info = hit.get("info", {})
        url_link = info.get("url", "No URL")

        title = _safe_str(info.get("title", "Unknown Title"))
        year = _safe_str(info.get("year", "Unknown Year"))
        doi_link = _safe_str(info.get("ee", url_link))
        pub_type = _safe_str(info.get("type", "Unknown Type"))
        venue_raw = info.get("venue")
        if venue_raw:
            venue_str = f"{_safe_str(venue_raw)} ({year})"
        else:
            venue_str = f"{pub_type} ({year})"

        authors_data = info.get("authors", {}).get("author", [])
        names = []
        if isinstance(authors_data, dict):
            names.append(authors_data.get("text", ""))
        elif isinstance(authors_data, list):
            for a in authors_data:
                names.append(a.get("text", "") if isinstance(a, dict) else str(a))
        authors_str = ", ".join(names) if names else "Unknown Authors"

        entry = (
            f"[Paper] {title}\n"
            f"Authors: {authors_str}\n"
            f"Venue: {venue_str}\n"
            f"Type: {pub_type}\n"
            f"Link: {doi_link}"
        )
        formatted_output.append(entry)

    return "\n\n---\n\n".join(formatted_output)


def _format_dblp_authors(hits: List[Dict], query: str) -> str:
    if not hits:
        return f"No authors found for query: '{query}'"

    formatted_output = []
    for hit in hits:
        info = hit.get("info", {})
        author_name = _safe_str(info.get("author", "Unknown Name"))
        url_link = info.get("url", "No URL")

        notes_raw = info.get("notes", {}).get("note", [])
        notes_list = (
            [notes_raw]
            if isinstance(notes_raw, dict)
            else (notes_raw if isinstance(notes_raw, list) else [])
        )
        affiliations = [
            n.get("text", "") for n in notes_list if isinstance(n, dict) and "text" in n
        ]

        notes_str = f"Context: {'; '.join(affiliations)}\n" if affiliations else ""

        entry = f"[Author] {author_name}\n{notes_str}Profile: {url_link}"
        formatted_output.append(entry)

    return "\n\n---\n\n".join(formatted_output)


def _format_dblp_venues(hits: List[Dict], query: str) -> str:
    if not hits:
        return f"No venues found for query: '{query}'"

    formatted_output = []
    for hit in hits:
        info = hit.get("info", {})
        url_link = info.get("url", "No URL")

        venue_name = _safe_str(info.get("venue", "Unknown Venue"))
        acronym = _safe_str(info.get("acronym", ""))
        venue_type = _safe_str(info.get("type", "Conference/Journal"))

        if acronym and acronym not in venue_name:
            display_name = f"{venue_name} ({acronym})"
        else:
            display_name = venue_name

        entry = f"[Venue] {display_name}\nType: {venue_type}\nLink: {url_link}"
        formatted_output.append(entry)

    return "\n\n---\n\n".join(formatted_output)


def dblp_search(query: str, kind: str = "papers", max_results: int = 5) -> str:
    """Search DBLP. kind ∈ {papers, authors, venues}."""
    kind_url = {
        "papers": "https://dblp.org/search/publ/api",
        "authors": "https://dblp.org/search/author/api",
        "venues": "https://dblp.org/search/venue/api",
    }
    if kind not in kind_url:
        raise ValueError(f"unknown dblp kind: {kind!r} (expected papers/authors/venues)")

    hits = _fetch_dblp(kind_url[kind], query, max_results)

    if kind == "papers":
        return _format_dblp_papers(hits, query)
    if kind == "authors":
        return _format_dblp_authors(hits, query)
    return _format_dblp_venues(hits, query)


__all__ = [
    "arxiv_search",
    "google_scholar_search",
    "dblp_search",
]
