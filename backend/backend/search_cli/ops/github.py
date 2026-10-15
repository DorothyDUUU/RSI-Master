"""GitHub search operations.

从 ``backend/huggingface_search_cli/mcp_servers/search_github.py`` 移植: 保留
``GitHubSearchClient`` 类 + 6 个格式化搜索函数 (去掉 FastMCP 装饰器)。密钥读
``GITHUB_TOKEN`` 环境变量, 缺失时抛 ``ValueError``。
"""

from __future__ import annotations

import base64
import os
from typing import Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class GitHubSearchClient:
    BASE_URL = "https://api.github.com"

    def __init__(self, token: str):
        self.headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
        }
        # 带 backoff 的重试 session (和 ops.hf 一致): GitHub 搜索 API 容易 429/5xx
        retry = Retry(
            total=2,
            connect=1,
            read=1,
            backoff_factor=0.5,
            status_forcelist=(408, 429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            raise_on_status=False,
        )
        self._session = requests.Session()
        self._session.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=8))

    # -------------------------
    # Internal helpers
    # -------------------------
    def _build_query(self, keyword: str, repo: str = None, user: str = None,
                     language: str = None, qualifiers: List[str] = None) -> str:
        parts = [keyword]

        if repo:
            parts.append(f"repo:{repo}")
        if user:
            parts.append(f"user:{user}")
        if language:
            parts.append(f"language:{language}")
        if qualifiers:
            parts.extend(qualifiers)

        return "+".join(parts)

    def _request(self, endpoint: str, params: Dict = None, timeout: int = 20) -> Dict:
        url = f"{self.BASE_URL}{endpoint}"
        resp = self._session.get(url, headers=self.headers, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    # -------------------------
    # Result Normalizer
    # -------------------------
    def _normalize(self, raw: Dict, result_type: str) -> Dict:
        items = []

        for it in raw.get("items", []):
            normalized = {
                "type": result_type,
                "name": it.get("name") or it.get("title"),
                "full_name": it.get("repository", {}).get("full_name") if "repository" in it else it.get("full_name"),
                "url": it.get("html_url"),
                "score": it.get("score"),
                "extra": {},
            }

            if result_type == "code":
                normalized["extra"] = {
                    "path": it.get("path"),
                    "repo": it["repository"]["full_name"],
                }
            elif result_type in ("issue", "pr"):
                normalized["extra"] = {
                    "state": it.get("state"),
                    "number": it.get("number"),
                    "repo": it.get("repository_url", "").replace("https://api.github.com/repos/", ""),
                }
            elif result_type == "repo":
                normalized["extra"] = {
                    "stars": it.get("stargazers_count"),
                    "language": it.get("language"),
                }
            elif result_type == "user":
                normalized["extra"] = {
                    "id": it.get("id"),
                    "avatar": it.get("avatar_url"),
                }

            items.append(normalized)

        return {"total": raw.get("total_count", 0), "items": items}

    # -------------------------
    # Search: Code
    # -------------------------
    def search_code(self, keyword: str, repo: str = None, user: str = None,
                    language: str = None, path: str = None,
                    per_page: int = 20) -> Dict:
        qualifiers = []
        if path:
            qualifiers.append(f"path:{path}")

        query = self._build_query(keyword, repo, user, language, qualifiers)

        raw = self._request("/search/code", {"q": query, "per_page": per_page})
        return self._normalize(raw, "code")

    # -------------------------
    # Search: Issues / PR
    # -------------------------
    def search_issues(self, keyword: str, repo: str = None, user: str = None,
                      language: str = None, state: str = None,
                      is_pr: bool = False, per_page: int = 20) -> Dict:
        qualifiers = []

        if state:
            qualifiers.append(f"state:{state}")
        qualifiers.append("is:pr" if is_pr else "is:issue")

        query = self._build_query(keyword, repo, user, language, qualifiers)

        raw = self._request("/search/issues", {"q": query, "per_page": per_page})
        return self._normalize(raw, "pr" if is_pr else "issue")

    def search_pr(self, keyword: str, repo: str = None, user: str = None,
                  language: str = None, state: str = None,
                  per_page: int = 20) -> Dict:
        return self.search_issues(
            keyword, repo, user, language, state, is_pr=True, per_page=per_page
        )

    # -------------------------
    # Search: Repositories
    # -------------------------
    def search_repos(self, keyword: str, user: str = None, language: str = None,
                     sort: str = None, order: str = None, per_page: int = 20) -> Dict:
        qualifiers = []

        if user:
            qualifiers.append(f"user:{user}")
        if language:
            qualifiers.append(f"language:{language}")

        query = self._build_query(keyword, qualifiers=qualifiers)

        params = {"q": query, "per_page": per_page}
        if sort:
            params["sort"] = sort
        if order:
            params["order"] = order

        raw = self._request("/search/repositories", params)
        return self._normalize(raw, "repo")

    # -------------------------
    # Search: Users
    # -------------------------
    def search_users(self, keyword: str, qualifiers: List[str] = None,
                     per_page: int = 20) -> Dict:
        query = self._build_query(keyword, qualifiers=qualifiers or [])
        raw = self._request("/search/users", {"q": query, "per_page": per_page})
        return self._normalize(raw, "user")

    # -------------------------
    # Get README.md
    # -------------------------
    def get_repo_readme(self, owner: str, repo: str) -> Dict:
        endpoint = f"/repos/{owner}/{repo}/readme"
        raw = self._request(endpoint)

        content_b64 = raw.get("content", "")
        markdown = base64.b64decode(content_b64).decode("utf-8")

        return {
            "name": raw.get("name"),
            "path": raw.get("path"),
            "html_url": raw.get("html_url"),
            "content": markdown,
        }


def get_github_client() -> GitHubSearchClient:
    """Initialize GitHub search client with token from environment."""
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError("GITHUB_TOKEN environment variable is required")
    return GitHubSearchClient(token)


# ── 格式化搜索函数 (返回 str, 供 LLM 直接消费) ──────────────────────────


def search_code(keyword: str, repo: Optional[str] = None, user: Optional[str] = None,
                language: Optional[str] = None, path: Optional[str] = None,
                per_page: int = 20) -> str:
    """Search for code across GitHub repositories with various filters."""
    try:
        client = get_github_client()
        result = client.search_code(
            keyword=keyword, repo=repo, user=user, language=language,
            path=path, per_page=per_page,
        )

        if not result["items"]:
            return f"No code found for keyword: {keyword}"

        output = [f"Found {result['total']} results for '{keyword}':\n"]
        for item in result["items"]:
            repo_name = item["full_name"] or "Unknown"
            file_path = item["extra"]["path"]
            file_url = item["url"]
            score = item.get("score", 0)
            output.append(f"Repository: {repo_name}")
            output.append(f"File: {file_path}")
            output.append(f"URL: {file_url}")
            output.append(f"Relevance Score: {score:.2f}")
            output.append("---")
        return "\n".join(output)
    except Exception as e:
        return f"Error searching code: {str(e)}"


def search_repositories(keyword: str, user: Optional[str] = None, language: Optional[str] = None,
                        sort: Optional[str] = None, order: Optional[str] = None,
                        per_page: int = 20) -> str:
    """Search for GitHub repositories with various sorting and filtering options."""
    try:
        client = get_github_client()
        result = client.search_repos(
            keyword=keyword, user=user, language=language,
            sort=sort, order=order, per_page=per_page,
        )

        if not result["items"]:
            return f"No repositories found for keyword: {keyword}"

        output = [f"Found {result['total']} repositories for '{keyword}':\n"]
        for item in result["items"]:
            name = item["name"] or "Unknown"
            full_name = item["full_name"] or "Unknown"
            stars = item["extra"]["stars"]
            language = item["extra"]["language"] or "Unknown"
            url = item["url"]
            output.append(f"Repository: {full_name}")
            output.append(f"Name: {name}")
            output.append(f"Language: {language}")
            output.append(f"Stars: {stars}")
            output.append(f"URL: {url}")
            output.append("---")
        return "\n".join(output)
    except Exception as e:
        return f"Error searching repositories: {str(e)}"


def search_users(keyword: str, qualifiers: Optional[List[str]] = None, per_page: int = 20) -> str:
    """Search for GitHub users based on keywords and qualifiers."""
    try:
        client = get_github_client()
        result = client.search_users(keyword=keyword, qualifiers=qualifiers, per_page=per_page)

        if not result["items"]:
            return f"No users found for keyword: {keyword}"

        output = [f"Found {result['total']} users for '{keyword}':\n"]
        for item in result["items"]:
            name = item["name"] or "Unknown"
            full_name = item["full_name"] or "Unknown"
            user_id = item["extra"]["id"]
            avatar = item["extra"]["avatar"]
            url = item["url"]
            output.append(f"User: {full_name}")
            output.append(f"User ID: {user_id}")
            output.append(f"Username: {name}")
            output.append(f"Avatar: {avatar}")
            output.append(f"Profile: {url}")
            output.append("---")
        return "\n".join(output)
    except Exception as e:
        return f"Error searching users: {str(e)}"


def search_issues(keyword: str, repo: Optional[str] = None, user: Optional[str] = None,
                  state: Optional[str] = None, per_page: int = 20) -> str:
    """Search for GitHub issues across repositories."""
    try:
        client = get_github_client()
        result = client.search_issues(
            keyword=keyword, repo=repo, user=user, state=state, is_pr=False, per_page=per_page,
        )

        if not result["items"]:
            return f"No issues found for keyword: {keyword}"

        output = [f"Found {result['total']} issues for '{keyword}':\n"]
        for item in result["items"]:
            title = item["name"] or "Unknown"
            number = item["extra"]["number"]
            repo_name = item["extra"]["repo"]
            state = item["extra"]["state"]
            url = item["url"]
            output.append(f"Issue #{number}: {title}")
            output.append(f"Repository: {repo_name}")
            output.append(f"State: {state}")
            output.append(f"URL: {url}")
            output.append("---")
        return "\n".join(output)
    except Exception as e:
        return f"Error searching issues: {str(e)}"


def search_prs(keyword: str, repo: Optional[str] = None, user: Optional[str] = None,
               state: Optional[str] = None, per_page: int = 20) -> str:
    """Search for GitHub pull requests across repositories."""
    try:
        client = get_github_client()
        result = client.search_pr(
            keyword=keyword, repo=repo, user=user, state=state, per_page=per_page,
        )

        if not result["items"]:
            return f"No pull requests found for keyword: {keyword}"

        output = [f"Found {result['total']} pull requests for '{keyword}':\n"]
        for item in result["items"]:
            title = item["name"] or "Unknown"
            number = item["extra"]["number"]
            repo_name = item["extra"]["repo"]
            state = item["extra"]["state"]
            url = item["url"]
            output.append(f"PR #{number}: {title}")
            output.append(f"Repository: {repo_name}")
            output.append(f"State: {state}")
            output.append(f"URL: {url}")
            output.append("---")
        return "\n".join(output)
    except Exception as e:
        return f"Error searching pull requests: {str(e)}"


def get_repository_readme(owner: str, repo: str) -> str:
    """Retrieve the README content from a GitHub repository."""
    try:
        client = get_github_client()
        readme_data = client.get_repo_readme(owner, repo)

        output = [
            f"README from {owner}/{repo}",
            f"File: {readme_data['path']}",
            f"URL: {readme_data['html_url']}",
            "\n--- Content ---\n",
            readme_data["content"],
        ]
        return "\n".join(output)
    except Exception as e:
        return f"Error retrieving README: {str(e)}"


__all__ = [
    "GitHubSearchClient",
    "get_github_client",
    "search_code",
    "search_repositories",
    "search_users",
    "search_issues",
    "search_prs",
    "get_repository_readme",
]
