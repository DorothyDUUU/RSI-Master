"""Search operations dispatcher — CLI 本地回退与 server 共用。

``execute(op, params)`` 把操作名分发给 ``ops.{hf,web,github,scholar}`` 里的具体
实现, 统一返回 JSON-safe 的 dict:

- 实现返回 dict → 原样透传 (如 materialize 的 ``{"local_path", "rows"}``)。
- 实现返回 list → 包成 ``{"result": [...]}``。
- 实现返回 str   → 包成 ``{"result": "..."}``。
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from . import github, hf, scholar, web

# op name -> 实现函数 (接收显式 kwargs)
REGISTRY: Dict[str, Callable[..., Any]] = {
    # HF Hub (8 个 hf_* 工具对应的 verb)
    "search_datasets": hf.search_datasets,
    "inspect_dataset": hf.inspect_dataset,
    "dataset_configs": hf.get_dataset_configs,
    "dataset_splits": hf.get_dataset_splits,
    "dataset_readme": hf.get_dataset_readme,
    "dataset_sample": hf.get_dataset_sample,
    "download_dataset": hf.download_dataset,
    "materialize_dataset": hf.materialize_dataset,
    # web
    "web_search": web.google_search,
    "web_parse": web.web_parse,
    # github
    "github_search_repositories": github.search_repositories,
    "github_search_code": github.search_code,
    "github_search_issues": github.search_issues,
    "github_search_prs": github.search_prs,
    "github_search_users": github.search_users,
    "github_get_repository_readme": github.get_repository_readme,
    # scholar
    "arxiv_search": scholar.arxiv_search,
    "google_scholar_search": scholar.google_scholar_search,
    "dblp_search": scholar.dblp_search,
}


def execute(op: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Dispatch a search operation by name. Always returns a JSON-safe dict."""
    func = REGISTRY.get(op)
    if func is None:
        raise ValueError(f"unknown search op: {op!r}")
    kwargs = {k: v for k, v in (params or {}).items() if v is not None}
    result = func(**kwargs)
    if isinstance(result, dict):
        return result
    return {"result": result}


__all__ = ["REGISTRY", "execute"]
