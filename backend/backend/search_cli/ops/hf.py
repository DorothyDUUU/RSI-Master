"""HuggingFace operations — search, inspect, download, materialize.

从 ``backend/huggingface_search_cli/server/hf_ops.py`` 移植 (去掉对 MCP server
包内 config 模块的依赖, 直接读进程 env)。这些函数由 ``ops.execute`` 统一调用:
CLI 本地回退与 server 都走同一个实现。

所有函数要么返回 str (格式化文本), 要么返回 dict (materialize)。
"""

from __future__ import annotations

import json
import logging
import os
import re
import time  # noqa: F401 — 保留与上游一致的导入面
from math import ceil
from itertools import islice
from pathlib import Path
from typing import Any

import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..config import get_default_max_results

LOGGER = logging.getLogger(__name__)

# ── Session management ───────────────────────────────────────────────

_SESSION: requests.Session | None = None
SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _get_session() -> requests.Session:
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    session = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1.0,
        status_forcelist=(408, 429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    _SESSION = session
    return session


def _hf_endpoint() -> str:
    return os.getenv("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def _hf_token() -> str:
    return os.getenv("HF_TOKEN", "")


def _hf_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "User-Agent": "search-cli/1.0",
    }
    if _hf_token():
        headers["Authorization"] = f"Bearer {_hf_token()}"
    return headers


def _request_json(path: str, *, params: dict[str, Any] | None = None, timeout: int = 25) -> Any:
    url = f"{_hf_endpoint()}{path}"
    resp = _get_session().get(url, headers=_hf_headers(), params=params or {}, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _request_text(url: str, *, timeout: int = 25) -> str:
    resp = _get_session().get(url, headers=_hf_headers(), timeout=timeout)
    resp.raise_for_status()
    return resp.text


def _safe_source_filename(source_id: str) -> str:
    return SAFE_FILENAME_RE.sub("_", source_id).strip("._") or "dataset"


def _shared_cache_dir() -> str:
    return os.getenv(
        "SHARED_CACHE_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "hf_sandbox", "materialized"),
    )


def _write_root(output_dir: str, *, cache_leaf: str) -> Path:
    """把写 op (download/materialize) 的 ``output_dir`` 收口成**绝对**根路径。

    不变量: 写路径绝不按进程自身 cwd 解析。search_cli server 为了让包可解析, 以
    ``cwd = backend/backend`` 启动 (submit 模板 ``lib_pod_common.sh: start_search_backend``
    先 ``cd backend/backend``)。相对 output_dir 原样透传到 server 再 ``Path(output_dir)``
    会按**错的 cwd** 落到 repo 树 —— 轨迹真实发生过 ``--output-dir data/hf_raw/medical-o1``
    写进 ``backend/backend/data/hf_raw/...`` (287M 泄漏)。因此:

    - 显式给出 output_dir 时**必须是绝对路径** (调用方 CLI 分发前已按自身 cwd 绝对化);
      走到这里仍是相对 = 忘绝对化的调用方 (或直接打 server 的人) → 响亮失败
      (ValueError → server 既有 400 路径), 不许静默落错地方。
    - 缺省则落到**绝对**共享缓存区 (与 server cwd 无关), 调用方从返回的
      output_dir/local_path 读回实际落点。
    """
    if output_dir:
        path = Path(output_dir)
        if not path.is_absolute():
            raise ValueError(
                "output_dir must be an absolute path (a relative one would be resolved "
                f"against the search server's own cwd, not yours): got {output_dir!r}. "
                "Pass an absolute path, e.g. --output-dir $PWD/data/<name>"
            )
        return path
    return Path(_shared_cache_dir()) / cache_leaf


# ── 镜像绝对重定向兜底 (corner case: hf-mirror.com) ────────────────────────
#
# 某些 HF 镜像 (如 hf-mirror.com) 提供 Hub API (search/inspect/fs.ls 正常), 但对
# ``/resolve/`` 文件下载返回 **308 绝对重定向** 回 huggingface.co:
#
#   HEAD https://hf-mirror.com/datasets/{org}/{name}/resolve/main/README.md
#       → 308  Location: https://huggingface.co/datasets/...
#
# huggingface_hub 的 ``_httpx_follow_relative_redirects_with_backoff`` 只跟随
# *相对* 重定向 (见 utils/_http.py: ``follow_redirects=False`` + 手工判
# ``parsed_target.netloc == ""``), 跨主机绝对重定向一律不跟 → 每次未缓存下载都会
# 以 ``LocalEntryNotFoundError`` 失败。这在真实轨迹里表现为两类:
#   - materialize → "No rows materialized for ..."
#   - download   → 每个文件 "Failed to download ... cannot find the requested files"
# 而 `requests` 默认 ``allow_redirects=True`` 会跟完整重定向链, 所以原始 GET 没问题。
# 这里给 ``datasets.load_dataset`` / ``hf_hub_download`` 加一层"首次失败 → 换回
# 规范 endpoint 重试一次"的兜底, 对能正常出文件的镜像无副作用 (首次即成功)。

# 失败签名: 镜像重定向/网络导致的下载失败 (datasets 会把 LocalEntryNotFoundError
# 包进 FileNotFoundError/OSError 再抛出, 所以用消息特征而不是 isinstance)。
_MIRROR_DOWNLOAD_SIGS = (
    "LocalEntryNotFoundError",
    "cannot find the requested files in the local cache",
    "An error happened while trying to locate the file on the Hub",
)


def _is_mirror_download_failure(exc: BaseException) -> bool:
    """True 当异常像是"镜像把文件下载重定向回 canonical host 而 hf 不跟随"造成的。

    用消息签名而不是 ``isinstance(LocalEntryNotFoundError)`` 判断: 该异常类的导出
    位置随 huggingface_hub 版本漂移 (1.23 顶层导出 / 1.28 移到 ``utils``), 而
    ``datasets`` 还会把它包进 ``FileNotFoundError`` 再抛 —— 消息特征最稳。
    """
    if isinstance(exc, ConnectionError):
        return True
    msg = str(exc)
    return any(sig in msg for sig in _MIRROR_DOWNLOAD_SIGS)


_CANONICAL = "https://huggingface.co"

# ``datasets`` / ``huggingface_hub`` 在 import 时把 endpoint **快照**成字符串模板:
#   datasets/config.py:      HF_ENDPOINT            (动态读取, 够 patch)
#                            HUB_DATASETS_URL       = HF_ENDPOINT + "/datasets/..."
#   huggingface_hub/constants.py:
#                            ENDPOINT               (动态读取, 够 patch)
#                            HUGGINGFACE_CO_URL_TEMPLATE = ENDPOINT + "/{repo_id}/..."
# 只改动态常量不行 —— ``load_dataset`` 非流式下载 parquet 走 ``HUB_DATASETS_URL``,
# 模板里烤死的镜像 host 会把 "canonical 重试" 原样打回镜像。所以两个模板字符串
# 必须随 endpoint 一起换掉再恢复。
_CANONICAL_URL_TEMPLATES = {
    "datasets": {
        "HUB_DATASETS_URL": f"{_CANONICAL}/datasets/{{repo_id}}/resolve/{{revision}}/{{path}}",
    },
    "huggingface_hub": {
        "HUGGINGFACE_CO_URL_TEMPLATE": f"{_CANONICAL}/{{repo_id}}/resolve/{{revision}}/{{filename}}",
    },
}


def _hf_load_with_mirror_fallback(loader: Any, *args: Any, **kwargs: Any) -> Any:
    """Run ``loader(*args, **kwargs)``, retrying once against the canonical endpoint.

    仅当调用以镜像重定向/网络下载失败告终、且当前 endpoint 不是 canonical 时才会
    重试: patch ``huggingface_hub``/``datasets`` 两侧的动态 endpoint 常量 **以及**
    import 时快照的 URL 模板 → huggingface.co, 跑完恢复。其余异常原样上抛, 行为不变。

    **为什么四个都要 patch**: ``datasets`` 在 import 时把 ``HF_ENDPOINT`` 快照进
    自己的 ``datasets.config.HF_ENDPOINT`` 和 ``HUB_DATASETS_URL``; ``huggingface_hub``
    同样把 ``ENDPOINT`` 快照进 ``HUGGINGFACE_CO_URL_TEMPLATE``。``load.py`` 显式传
    ``endpoint=config.HF_ENDPOINT`` 给 ``HfApi``/``HfFileSystem``, 非流式下载 parquet
    走 ``HUB_DATASETS_URL`` —— 只 patch 动态常量, "canonical 重试" 仍打镜像, 未缓存
    parquet 依旧 ``LocalEntryNotFoundError``。真实轨迹里 materialize 全挂
    "No rows materialized" 就来自这个半修复 (download_dataset 走原始 requests 不受
    影响, 所以下载类轨迹看起来修好了)。
    """
    try:
        return loader(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 — 需按签名判断是否属于镜像下载失败
        if not _is_mirror_download_failure(exc):
            raise
        import huggingface_hub.constants as hf_const

        if hf_const.ENDPOINT in (None, _CANONICAL):
            raise

        # 记录旧值 → 设成 canonical; 恢复时逆序还原 (缺失属性也复原为缺失)。
        saved: list[tuple[Any, str, Any]] = []
        _set_or_record(hf_const, "ENDPOINT", _CANONICAL, saved)
        for name, value in _CANONICAL_URL_TEMPLATES["huggingface_hub"].items():
            _set_or_record(hf_const, name, value, saved)

        ds_config = None
        try:
            import datasets.config as _ds_config

            ds_config = _ds_config
            _set_or_record(ds_config, "HF_ENDPOINT", _CANONICAL, saved)
            for name, value in _CANONICAL_URL_TEMPLATES["datasets"].items():
                _set_or_record(ds_config, name, value, saved)
        except Exception:  # datasets 未安装/未导入时只 patch hub 侧
            pass
        try:
            return loader(*args, **kwargs)
        finally:
            for mod, name, old in reversed(saved):
                if old is None:
                    delattr(mod, name)
                else:
                    setattr(mod, name, old)


def _set_or_record(mod: Any, name: str, value: Any, saved: list) -> None:
    """Set ``mod.name = value``, remembering the old value (or absence) for restore."""
    if hasattr(mod, name):
        old = getattr(mod, name)
        if old == value:
            return
        saved.append((mod, name, old))
    else:
        saved.append((mod, name, None))
    setattr(mod, name, value)


def _hf_endpoint_download_url(dataset_id: str, relative_path: str) -> str:
    """``{HF_ENDPOINT}/datasets/{id}/resolve/main/{path}`` — 供原始 GET 下载用。

    ``requests`` 会跟随镜像的绝对重定向, 所以同一条 URL 在有/无镜像时都能下载。
    """
    return f"{_hf_endpoint()}/datasets/{dataset_id}/resolve/main/{relative_path}"


# ── Search ───────────────────────────────────────────────────────────

def search_datasets(query: str, limit: int | None = None, author: str | None = None) -> str:
    """Return formatted search results text.

    ``limit`` defaults to ``config.yaml defaults.max_results`` when omitted
    (agent tools always pass an explicit ``--limit``, so this only affects
    direct CLI/op calls).
    """
    if limit is None:
        limit = get_default_max_results()
    from huggingface_hub import list_datasets as _list_datasets

    try:
        payload = _request_json(
            "/api/datasets",
            params={
                "search": f"{query} author:{author}" if author else query,
                "limit": limit,
            },
            timeout=25,
        )
        results = payload if isinstance(payload, list) else []
        if not results:
            return f"No accessible datasets found for query: '{query}'"

        lines = [f"Found {len(results)} datasets for '{query}':", ""]
        for i, item in enumerate(results, 1):
            if not isinstance(item, dict):
                continue
            dataset_id = str(item.get("id") or item.get("name") or "unknown")
            author_name = item.get("author")
            if not author_name and "/" in dataset_id:
                author_name = dataset_id.split("/", 1)[0]
            downloads = item.get("downloads") or 0
            likes = item.get("likes") or 0
            lines.append(
                f"{i}. {dataset_id} | Author: {author_name or 'unknown'} | "
                f"Downloads: {downloads:,} | Likes: {likes} | "
                f"URL: https://huggingface.co/datasets/{dataset_id}"
            )
        return "\n".join(lines)
    except Exception as api_error:
        try:
            search_params = {
                "search": f"{query} author:{author}" if author else query,
                "limit": limit,
            }
            results = list(_list_datasets(**search_params))
            if not results:
                return f"No accessible datasets found for query: '{query}'"
            lines = [f"Found {len(results)} datasets for '{query}':", ""]
            for i, ds in enumerate(results, 1):
                lines.append(
                    f"{i}. {ds.id} | Author: {ds.author or 'unknown'} | "
                    f"Downloads: {ds.downloads or 0:,} | Likes: {ds.likes or 0} | "
                    f"URL: https://huggingface.co/datasets/{ds.id}"
                )
            return "\n".join(lines)
        except Exception as hub_error:
            return (
                f"Error searching datasets: REST API failed with {api_error}; "
                f"huggingface_hub fallback failed with {hub_error}"
            )


# ── Inspect ──────────────────────────────────────────────────────────

def inspect_dataset(dataset_id: str, config: str | None = None) -> str:
    """Return comprehensive dataset metadata as YAML."""
    from huggingface_hub import dataset_info as _dataset_info
    from datasets import load_dataset_builder, get_dataset_config_names

    hub_info = None
    try:
        from huggingface_hub import login
        token = _hf_token()
        if token:
            try:
                login(token=token)
            except Exception:
                pass
        hub_info = _dataset_info(dataset_id)
    except Exception:
        pass

    builder = None
    try:
        builder = load_dataset_builder(dataset_id, name=config)
    except Exception:
        pass

    configs = []
    try:
        configs = get_dataset_config_names(dataset_id)
    except Exception:
        pass

    hub_metadata = {}
    if hub_info:
        hub_metadata = {
            "id": hub_info.id,
            "author": hub_info.author,
            "created_at": hub_info.created_at.isoformat() if hub_info.created_at else None,
            "last_modified": hub_info.last_modified.isoformat() if hub_info.last_modified else None,
            "private": hub_info.private,
            "disabled": hub_info.disabled,
            "gated": hub_info.gated,
            "downloads": hub_info.downloads,
            "downloads_all_time": hub_info.downloads_all_time,
            "likes": hub_info.likes,
            "tags": hub_info.tags,
            "paperswithcode_id": hub_info.paperswithcode_id,
            "trending_score": hub_info.trending_score,
            "files": [s.rfilename for s in hub_info.siblings],
        }

    builder_info = {}
    if builder:
        info = builder.info
        builder_info = {
            "dataset_name": builder.dataset_name,
            "config_name": builder.name,
            "description": info.description,
            "citation": info.citation,
            "homepage": info.homepage,
            "license": info.license,
            "features": {k: str(v) for k, v in info.features.items()} if info.features else {},
            "splits": {k: str(v) for k, v in info.splits.items()} if info.splits else {},
            "configs": configs,
        }

    merged = {**hub_metadata, **builder_info}
    return yaml.dump(merged, allow_unicode=True, sort_keys=False, default_flow_style=False)


# ── Configs / Splits ─────────────────────────────────────────────────

def get_dataset_configs(dataset_id: str) -> str:
    """Return available configs as JSON or a default-config message."""
    from datasets import get_dataset_config_names as _get_configs
    try:
        configs = _get_configs(dataset_id)
        if not configs:
            return "This dataset uses the default configuration (no multiple configs)."
        return json.dumps(configs)
    except Exception as e:
        return f"Error listing configs for '{dataset_id}': {e}"


def get_dataset_splits(dataset_id: str, config: str | None = None) -> str:
    """Return available split names (cheap — does not load dataset rows).

    Previously this called ``load_dataset`` and stringified the whole dataset,
    which pulled every row into memory for what is meant to be a cheap
    discovery verb.
    """
    from datasets import get_dataset_split_names as _get_splits
    try:
        splits = list(_get_splits(dataset_id, config_name=config))
        if not splits:
            return f"No splits found for '{dataset_id}' (config: {config or 'default'})"
        lines = [
            f"Splits for '{dataset_id}' (config: {config or 'default'}):",
            f"Count: {len(splits)}",
            "",
        ]
        lines.extend(f"- {name}" for name in splits)
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing splits for '{dataset_id}': {e}"


# ── README ───────────────────────────────────────────────────────────

def get_dataset_readme(dataset_id: str) -> str:
    """Return the README.md or description for a dataset."""
    from huggingface_hub import dataset_info as _dataset_info, hf_hub_download as _download

    errors: list[str] = []

    # Try 1: Direct HTTP GET
    try:
        content = _request_text(
            f"{_hf_endpoint()}/datasets/{dataset_id}/resolve/main/README.md", timeout=25
        )
        if content.strip():
            return content
    except Exception as e:
        errors.append(f"direct_readme_get={e}")

    # Try 2: Hub API cardData
    try:
        payload = _request_json(f"/api/datasets/{dataset_id}", timeout=25)
        if isinstance(payload, dict):
            card = payload.get("cardData") or {}
            if isinstance(card, dict):
                desc = card.get("description", "") or (card.get("dataset_info", {}) or {}).get("description", "")
                if desc:
                    return desc
    except Exception as e:
        errors.append(f"api_dataset_info={e}")

    # Try 3: huggingface_hub dataset_info
    try:
        info = _dataset_info(dataset_id)
        desc = info.card_data.get("description", "") or info.card_data.get("dataset_info", {}).get("description", "")
        if desc:
            return desc
    except Exception as e:
        errors.append(f"huggingface_hub_dataset_info={e}")

    # Try 4: hf_hub_download
    try:
        readme_path = _download(repo_id=dataset_id, filename="README.md", repo_type="dataset")
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
        if content.strip():
            return content
    except Exception as e:
        errors.append(f"hf_hub_download={e}")

    return f"Error retrieving README for '{dataset_id}': {' | '.join(errors) if errors else 'not found'}"


# ── Sample ───────────────────────────────────────────────────────────

def get_dataset_sample(
    dataset_id: str,
    config: str | None = None,
    split: str | None = None,
    num_samples: int = 5,
) -> str:
    """Preview only N rows; never materialize a whole split for inspection."""
    from datasets import load_dataset as _load_dataset, get_dataset_split_names as _get_splits

    try:
        if num_samples <= 0:
            return "Error getting dataset sample: num_samples must be positive"
        if not split:
            split_names = _hf_load_with_mirror_fallback(
                _get_splits, dataset_id, config_name=config,
            )
            split = split_names[0] if split_names else "train"

        def _preview(*args: Any, **kwargs: Any) -> list[Any]:
            # Consume the lazy iterator inside the mirror-fallback boundary:
            # streaming network failures may happen during iteration, not load.
            ds = _load_dataset(*args, **kwargs)
            return list(islice(ds, num_samples))

        rows = _hf_load_with_mirror_fallback(
            _preview, dataset_id, name=config, split=split, streaming=True,
        )
        samples = []
        for example in rows:
            samples.append({
                k: v if isinstance(v, (str, int, float, bool, list, dict)) or v is None else str(v)
                for k, v in example.items()
            })

        if not samples:
            return f"No samples retrieved from '{dataset_id}' (split: '{split}')"

        columns = list(samples[0].keys())
        msgs = [
            f"Dataset Sample from '{dataset_id}'",
            f"Config: {config or 'default'} | Split: {split}",
            f"Columns ({len(columns)}): {', '.join(columns)}",
            "\nColumn Types:",
        ]
        msgs.extend(f"  {k}: {type(v).__name__}" for k, v in samples[0].items())
        for i, s in enumerate(samples, 1):
            msgs.append(f"\n[Sample {i}]")
            for k, v in s.items():
                msgs.append(f"  {k}: {v}")
        return "\n".join(msgs)
    except Exception as e:
        return f"Error getting dataset sample: {e}"


# ── Download ─────────────────────────────────────────────────────────

def download_dataset(dataset_id: str, output_dir: str = "") -> dict[str, Any]:
    """Download all raw files from a HuggingFace dataset repository.

    用**原始 ``requests`` 流式 GET** 下载字节, 不用 ``hf_hub_download``: 某些镜像
    (hf-mirror.com) 对 ``/resolve/`` 返回绝对重定向回 huggingface.co, 而
    huggingface_hub 在文件下载路径不跟随跨主机重定向 → ``hf_hub_download`` 必失败;
    ``requests`` 默认跟随重定向链, 同一份代码在有/无镜像的环境都能下载。

    返回结构化 dict (不再是格式化文本), 让 agent 一眼看到成败:
    ``{"output_dir", "downloaded_files": [...], "failed_files": [{"file", "error"}], "error": None|str}``
    """
    out_root = _write_root(output_dir, cache_leaf="downloaded_datasets")
    from huggingface_hub import HfFileSystem

    try:
        token = _hf_token() or None
        fs = HfFileSystem(token=token)
        all_files = list(fs.ls(f"datasets/{dataset_id}", recursive=True))
    except Exception as e:
        return {
            "output_dir": str(out_root),
            "downloaded_files": [],
            "failed_files": [{"file": "", "error": f"list files failed: {type(e).__name__}: {e}"[:200]}],
            "error": f"failed to list files for {dataset_id}: {type(e).__name__}: {e}"[:300],
        }

    prefix = f"datasets/{dataset_id}/"
    headers = _hf_headers()
    downloaded: list[str] = []
    failed: list[dict[str, str]] = []
    for fp in all_files:
        if fp.get("type") != "file":
            continue
        name = str(fp.get("name") or "")
        rel = name[len(prefix):] if name.startswith(prefix) else name
        if not rel:
            continue
        # 防路径穿越: 拒绝带 ".." 组件的相对路径 (文件名来自远端 repo, 不可信)
        if ".." in Path(rel).parts:
            failed.append({"file": rel, "error": "skipped: path contains '..' component"})
            continue
        dest = out_root / rel
        try:
            resp = _get_session().get(
                _hf_endpoint_download_url(dataset_id, rel),
                headers=headers,
                stream=True,
                timeout=180,
            )
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    fh.write(chunk)
            downloaded.append(rel)
        except Exception as e:  # noqa: BLE001 — 单文件失败不中断整批
            failed.append({"file": rel, "error": f"{type(e).__name__}: {e}"[:200]})

    result: dict[str, Any] = {
        "output_dir": str(out_root),
        "downloaded_files": downloaded,
        "failed_files": failed,
        "error": None,
    }
    if not downloaded:
        detail = "; ".join(f"{f['file']}: {f['error']}" for f in failed[:3])
        result["error"] = f"all {len(failed)} file(s) failed to download for {dataset_id}: {detail}"[:500]
    return result


# ── Materialize ──────────────────────────────────────────────────────

def materialize_dataset(
    dataset_id: str,
    config: str | None = None,
    split: str | None = None,
    max_rows: int = 2048,
    output_dir: str = "",
) -> dict[str, Any]:
    """Materialize a HuggingFace dataset to JSONL.

    参数名用 ``dataset_id`` (与同模块其他 hf op 及 CLI/工具契约一致) —— 之前叫
    ``dataset`` 导致 ops.execute 以 ``dataset_id`` 关键字分发时报 TypeError。

    Cache 键为 ``{dataset}_{config}_{split}_{max_rows}`` —— 必须含 split,
    否则同 config/max_rows 下不同 split 会串缓存。

    Returns {"local_path": "<path>", "rows": <n>} on success,
    or {"error": "<msg>"} on failure.
    """
    cache_root = _write_root(output_dir, cache_leaf="materialized_datasets")
    cache_root.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset as _load_dataset, get_dataset_config_names, get_dataset_split_names

    config_name = (config or "default").strip()
    split_pref = (split or "").strip()
    split_tag = split_pref or "all"
    max_rows_tag = "all" if max_rows is None else str(max_rows)
    # 缓存键必须含 split: 否则先 materialize train 再请求 test 会命中旧缓存, 静默返回错数据
    cache_key = f"{_safe_source_filename(dataset_id)}_{config_name}_{split_tag}_{max_rows_tag}"
    output_path = cache_root / f"{cache_key}.jsonl"

    if output_path.exists() and output_path.stat().st_size > 0:
        row_count = sum(1 for line in output_path.open() if line.strip())
        if row_count > 0:
            return {"local_path": str(output_path), "rows": row_count}

    config_names: list[str] = []
    try:
        config_names = list(get_dataset_config_names(dataset_id))
    except Exception:
        pass

    if config and config != "default":
        configs_to_try = [config]
    elif config_names:
        configs_to_try = config_names
    else:
        configs_to_try = [""]

    rows: list[dict[str, Any]] = []
    # max_rows=None 表示 "全部": 多 config 分摊预算时不能对 None 做 ceil(None / n)
    if len(configs_to_try) <= 1:
        per_config_budget = max_rows
    else:
        per_config_budget = max_rows if max_rows is None else max(1, ceil(max_rows / len(configs_to_try)))

    load_errors: list[str] = []
    for cfg_name in configs_to_try:
        remaining = max_rows - len(rows) if max_rows else None
        if remaining is not None and remaining <= 0:
            break
        budget = min(remaining, per_config_budget) if remaining is not None else per_config_budget

        try:
            split_names = list(get_dataset_split_names(dataset_id, config_name=cfg_name or None))
        except Exception:
            split_names = []

        chosen_split = _pick_split(split_names, split_pref)
        if not chosen_split:
            chosen_split = split_pref or "train"

        load_args = [dataset_id]
        if cfg_name:
            load_args.append(cfg_name)
        try:
            slice_spec = f"{chosen_split}[:{budget}]" if budget else chosen_split
            ds = _hf_load_with_mirror_fallback(_load_dataset, *load_args, split=slice_spec)
        except Exception:
            try:
                ds = _hf_load_with_mirror_fallback(_load_dataset, *load_args, split=chosen_split)
            except Exception as e:
                LOGGER.warning("Failed to load %s config=%s split=%s: %s", dataset_id, cfg_name, chosen_split, e)
                load_errors.append(f"config={cfg_name or 'default'} split={chosen_split}: {type(e).__name__}: {e}"[:200])
                continue

        for idx, row in enumerate(ds):
            if budget and idx >= budget:
                break
            if isinstance(row, dict):
                if cfg_name:
                    row.setdefault("dataset_config", cfg_name)
                rows.append(row)

    if not rows:
        # 把底层的失败原因带上, 而不是干巴巴一句 "No rows materialized" — 轨迹里
        # agent 对同一个失败源盲目重试十几次, 大多是看不到原因 (镜像重定向 / 分片
        # 不存在 / gated), 带上原因能帮它决定是换源还是重试。
        reason = "; ".join(load_errors[:5]) if load_errors else "no rows in source (empty split / all rows filtered?)"
        return {"error": f"No rows materialized for {dataset_id}: {reason}"[:500]}

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    return {"local_path": str(output_path), "rows": len(rows)}


def _pick_split(split_names: list[str], preferred: str) -> str:
    """Pick the best split from available splits."""
    if preferred and preferred in split_names:
        return preferred
    train_like = ["train", "training", "train_sft"]
    for name in train_like:
        if name in split_names:
            return name
    for name in split_names:
        lower = name.lower()
        if "train" in lower:
            return name
    skip = {"test", "eval", "evaluation", "dev"}
    for name in split_names:
        if name.lower() not in skip:
            return name
    return split_names[0] if split_names else ""
