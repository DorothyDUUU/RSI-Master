"""CLI entry-point for the search_cli backend.

Usage::

    python search_cli/main.py search "math word problems" --limit 20
    python search_cli/main.py inspect openai/gsm8k
    python search_cli/main.py materialize openai/gsm8k --config main --split train --max-rows 2048
    python search_cli/main.py web-search "LLM post-training"
    python search_cli/main.py web-parse https://example.com
    python search_cli/main.py github-repos "llm post-training"
    python search_cli/main.py scholar-arxiv "RLHF" --by content --max-results 5
    python search_cli/main.py scholar-dblp "Diffusion Model" --kind papers
    python search_cli/main.py health
    python search_cli/main.py server start --port 8991

前后端分离: 每个搜索 verb 先探测 server ``/health``; server 活着则把请求代理到
API 端口 (``POST /search``), 否则在 CLI 进程内本地直接执行 (``ops.execute``)。
所有搜索输出都是 stdout 上的单个 JSON 对象::

    {"ok": true, "result": {...}}

配置读 ``config.yaml`` (包目录), ``--url`` 可临时覆盖 server 地址。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List

from .client import SearchClient, SearchError
from .config import (
    get_probe_timeout,
    get_request_timeout,
    get_search_url,
)
from .ops import execute
from .req_log import log_request_record


# ── Front/back separation: health probe → proxy or local fallback ──────────


def _try_server(url: str, probe_timeout: float) -> SearchClient | None:
    """Probe ``/health``; return a client when the server is alive, else None."""
    client = SearchClient(url, timeout=get_request_timeout())
    try:
        if client.health_check(timeout=probe_timeout) is not None:
            return client
    except SearchError:
        pass
    return None


def _dispatch(verb: str, params: Dict[str, Any], url: str | None = None) -> Dict[str, Any]:
    """Run a search verb: proxy over HTTP when the server is up, else local."""
    base_url = url or get_search_url()
    client = _try_server(base_url, get_probe_timeout())
    if client is not None:
        try:
            return client.search(verb, params)  # {"ok": True, "result": ...}
        except SearchError as exc:
            # server 活着但请求失败 → 直接上报, 不回退本地 (同一网络故障)
            return {"ok": False, "error": str(exc)}
    # health 探测失败 → 本地直接执行 (T8: 也记一行 via=local, 覆盖 server 不可达的路径)
    ts = time.time()
    try:
        result = execute(verb, params)
        ok = True
        error = ""
    except Exception as exc:  # noqa: BLE001 — surface local op failures
        result = None
        ok = False
        error = f"{type(exc).__name__}: {exc}"
    log_request_record(
        {
            "ts": round(ts, 3),
            "endpoint": "local",
            "method": "CLI",
            "op": verb[:80],
            "via": "local",
            "latency_ms": round((time.time() - ts) * 1000, 1),
            "ok": ok,
            "error": error[:200],
        }
    )
    return {"ok": ok, "result": result} if ok else {"ok": False, "error": error}


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


# 写 verb: 相对 ``output_dir`` 必须先在 CLI 边界按**本进程 cwd** (= 调用方工作目录,
# cc/v2 agent 都是自己的 workspace) 绝对化。否则透传给 cwd=backend/backend 的 server,
# 相对路径会被按 server 的错 cwd 解析 → 数据写进 repo 树 (真实泄漏 backend/backend/data/)。
_WRITE_VERBS = frozenset({"download_dataset", "materialize_dataset"})


def _resolve_write_paths(verb: str, params: Dict[str, Any]) -> None:
    """把写 verb 的 ``output_dir`` 收口成绝对路径 (就地改 ``params``)。

    已绝对者原样保留 (幂等); 缺省不管 → 由 ``ops`` 落到绝对共享缓存区。
    """
    if verb not in _WRITE_VERBS:
        return
    out = params.get("output_dir")
    if out and not os.path.isabs(out):
        params["output_dir"] = os.path.abspath(out)


# ── Command handlers ────────────────────────────────────────────────────────


def _make_search_cmd(verb: str, keys: List[str]) -> Callable[[argparse.Namespace], int]:
    """Build a handler that collects the listed argparse keys into params."""

    def cmd(args: argparse.Namespace) -> int:
        params = {k: getattr(args, k) for k in keys if getattr(args, k, None) is not None}
        _resolve_write_paths(verb, params)
        _print_json(_dispatch(verb, params, url=args.url))
        return 0

    return cmd


def cmd_health(args: argparse.Namespace) -> int:
    url = args.url or get_search_url()
    client = SearchClient(url, timeout=get_request_timeout())
    body = client.health_check(timeout=get_probe_timeout())
    if body is not None:
        _print_json({"status": "healthy", "via": "server", "health": body})
        return 0
    _print_json({"status": "unreachable", "via": "local", "error": f"server not reachable at {url}"})
    return 1


def cmd_server_start(args: argparse.Namespace) -> int:
    """Start the FastAPI server (convenience wrapper around uvicorn)."""
    import uvicorn

    from .config import load_config
    from .server import app

    cfg = load_config().get("server", {})
    host = args.host or cfg.get("host", "0.0.0.0")
    port = args.port or cfg.get("port", 8991)
    log_level = args.log_level or cfg.get("log_level", "info")

    print(f"Starting search_cli server on {host}:{port} (log-level={log_level})", file=sys.stderr)
    uvicorn.run(app, host=host, port=port, log_level=log_level)
    return 0


# ── Parser setup ────────────────────────────────────────────────────────────

_COMMON = argparse.ArgumentParser(add_help=False)
_COMMON.add_argument(
    "--url", default=None,
    help="Server URL override (default: config.yaml backend.host:port)",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="search-cli",
        description="Unified search CLI (HF / web / github / scholar). "
                    "Probes the server first; falls back to local execution when down.",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")
    sub.required = True

    # ── HF Hub verbs (1:1 with the hf_* agent tools) ────────────────────────
    p = sub.add_parser("search", parents=[_COMMON], help="Search HF datasets")
    p.add_argument("query")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--author", default=None)
    p.set_defaults(func=_make_search_cmd("search_datasets", ["query", "limit", "author"]))

    p = sub.add_parser("inspect", parents=[_COMMON], help="Inspect a dataset's metadata")
    p.add_argument("dataset_id")
    p.add_argument("--config", default=None)
    p.set_defaults(func=_make_search_cmd("inspect_dataset", ["dataset_id", "config"]))

    p = sub.add_parser("configs", parents=[_COMMON], help="List a dataset's configs")
    p.add_argument("dataset_id")
    p.set_defaults(func=_make_search_cmd("dataset_configs", ["dataset_id"]))

    p = sub.add_parser("splits", parents=[_COMMON], help="List a dataset's splits")
    p.add_argument("dataset_id")
    p.add_argument("--config", default=None)
    p.set_defaults(func=_make_search_cmd("dataset_splits", ["dataset_id", "config"]))

    p = sub.add_parser("readme", parents=[_COMMON], help="Get a dataset's README")
    p.add_argument("dataset_id")
    p.set_defaults(func=_make_search_cmd("dataset_readme", ["dataset_id"]))

    p = sub.add_parser("sample", parents=[_COMMON], help="Fetch sample records")
    p.add_argument("dataset_id")
    p.add_argument("--config", default=None)
    p.add_argument("--split", default=None)
    p.add_argument("--num-samples", type=int, default=None)
    p.set_defaults(func=_make_search_cmd("dataset_sample", ["dataset_id", "config", "split", "num_samples"]))

    p = sub.add_parser("download", parents=[_COMMON], help="Download a dataset's raw files")
    p.add_argument("dataset_id")
    p.add_argument("--output-dir", default=None)
    p.set_defaults(func=_make_search_cmd("download_dataset", ["dataset_id", "output_dir"]))

    p = sub.add_parser("materialize", parents=[_COMMON], help="Materialize a dataset to JSONL")
    p.add_argument("dataset_id")
    p.add_argument("--config", default=None)
    p.add_argument("--split", default=None)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    p.set_defaults(func=_make_search_cmd("materialize_dataset", ["dataset_id", "config", "split", "max_rows", "output_dir"]))

    # ── web ─────────────────────────────────────────────────────────────────
    p = sub.add_parser("web-search", parents=[_COMMON], help="Google web search (Serper)")
    p.add_argument("query")
    p.set_defaults(func=_make_search_cmd("web_search", ["query"]))

    p = sub.add_parser("web-parse", parents=[_COMMON], help="Extract a web page's text (Serper)")
    p.add_argument("url")
    p.set_defaults(func=_make_search_cmd("web_parse", ["url"]))

    # ── github ──────────────────────────────────────────────────────────────
    p = sub.add_parser("github-repos", parents=[_COMMON], help="Search GitHub repositories")
    p.add_argument("keyword")
    p.add_argument("--user", default=None)
    p.add_argument("--language", default=None)
    p.add_argument("--sort", default=None)
    p.add_argument("--order", default=None)
    p.add_argument("--per-page", type=int, default=None)
    p.set_defaults(func=_make_search_cmd("github_search_repositories", ["keyword", "user", "language", "sort", "order", "per_page"]))

    p = sub.add_parser("github-code", parents=[_COMMON], help="Search GitHub code")
    p.add_argument("keyword")
    p.add_argument("--repo", default=None)
    p.add_argument("--user", default=None)
    p.add_argument("--language", default=None)
    p.add_argument("--path", default=None)
    p.add_argument("--per-page", type=int, default=None)
    p.set_defaults(func=_make_search_cmd("github_search_code", ["keyword", "repo", "user", "language", "path", "per_page"]))

    p = sub.add_parser("github-issues", parents=[_COMMON], help="Search GitHub issues")
    p.add_argument("keyword")
    p.add_argument("--repo", default=None)
    p.add_argument("--user", default=None)
    p.add_argument("--state", default=None)
    p.add_argument("--per-page", type=int, default=None)
    p.set_defaults(func=_make_search_cmd("github_search_issues", ["keyword", "repo", "user", "state", "per_page"]))

    p = sub.add_parser("github-prs", parents=[_COMMON], help="Search GitHub pull requests")
    p.add_argument("keyword")
    p.add_argument("--repo", default=None)
    p.add_argument("--user", default=None)
    p.add_argument("--state", default=None)
    p.add_argument("--per-page", type=int, default=None)
    p.set_defaults(func=_make_search_cmd("github_search_prs", ["keyword", "repo", "user", "state", "per_page"]))

    p = sub.add_parser("github-users", parents=[_COMMON], help="Search GitHub users")
    p.add_argument("keyword")
    p.add_argument("--per-page", type=int, default=None)
    p.set_defaults(func=_make_search_cmd("github_search_users", ["keyword", "per_page"]))

    p = sub.add_parser("github-readme", parents=[_COMMON], help="Get a repo's README")
    p.add_argument("owner")
    p.add_argument("repo")
    p.set_defaults(func=_make_search_cmd("github_get_repository_readme", ["owner", "repo"]))

    # ── scholar ─────────────────────────────────────────────────────────────
    p = sub.add_parser("scholar-arxiv", parents=[_COMMON], help="Search arXiv")
    p.add_argument("query")
    p.add_argument("--by", choices=["content", "author"], default="content")
    p.add_argument("--max-results", type=int, default=None)
    p.set_defaults(func=_make_search_cmd("arxiv_search", ["query", "by", "max_results"]))

    p = sub.add_parser("scholar-google", parents=[_COMMON], help="Google Scholar search (Serper)")
    p.add_argument("query")
    p.set_defaults(func=_make_search_cmd("google_scholar_search", ["query"]))

    p = sub.add_parser("scholar-dblp", parents=[_COMMON], help="Search DBLP (papers/authors/venues)")
    p.add_argument("query")
    p.add_argument("--kind", choices=["papers", "authors", "venues"], default="papers")
    p.add_argument("--max-results", type=int, default=None)
    p.set_defaults(func=_make_search_cmd("dblp_search", ["query", "kind", "max_results"]))

    # ── system ──────────────────────────────────────────────────────────────
    p = sub.add_parser("health", parents=[_COMMON], help="Probe the server /health")
    p.set_defaults(func=cmd_health)

    p_srv = sub.add_parser("server", help="Server management commands")
    srv_sub = p_srv.add_subparsers(dest="server_command")
    srv_start = srv_sub.add_parser("start", help="Start the search_cli server")
    srv_start.add_argument("--host", default=None, help="Bind host (default: 0.0.0.0)")
    srv_start.add_argument("--port", type=int, default=None, help="Bind port (default: 8991)")
    srv_start.add_argument("--log-level", default=None, choices=["debug", "info", "warning", "error"])
    srv_start.add_argument("--reload", action="store_true", default=False, help="Enable auto-reload")
    srv_start.set_defaults(func=cmd_server_start)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
