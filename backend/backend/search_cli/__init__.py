"""search_cli — 前后端分离的统一搜索 backend（HF / web / github / scholar）。

用法::

    python search_cli/main.py search "math word problems" --limit 20
    python search_cli/main.py inspect openai/gsm8k
    python search_cli/main.py materialize openai/gsm8k --config main --split train
    python search_cli/main.py web-search "LLM post-training"
    python search_cli/main.py github-repos "post-training"
    python search_cli/main.py scholar-arxiv "RLHF" --by content
    python search_cli/main.py scholar-dblp "Diffusion Model" --kind papers
    python search_cli/main.py health
    python search_cli/main.py server start --port 8991

前后端分离: CLI 先探 server ``/health``; server 活着则把请求代理到 API 端口,
否则在 CLI 进程内本地直接执行。所有输出都是 stdout JSON。

对外暴露的模块级符号::

    from search_cli import SearchClient, SearchError, get_search_url, execute
"""

from __future__ import annotations

from .client import SearchClient, SearchError
from .config import get_request_timeout, get_search_url
from .ops import execute

__all__ = [
    "SearchClient",
    "SearchError",
    "execute",
    "get_request_timeout",
    "get_search_url",
]
