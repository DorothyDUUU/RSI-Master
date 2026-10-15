#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HuggingFace Search MCP Server – compatibility wrapper.

This file provides backward compatibility with the original MCP-based
search_huggingface.py. It wraps the CLI's hf_ops functions as MCP tools.

Usage:
    python -m huggingface_search_cli.mcp_server

Environment variables:
    HF_TOKEN: HuggingFace API token
    HF_ENDPOINT: HuggingFace mirror endpoint (e.g. https://hf-mirror.com)
    HF_SANDBOX_URL: Sandbox server URL for proxy mode
"""

import os
import sys
import yaml
import logging
from typing import Optional

from mcp.server.fastmcp import FastMCP

logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s - %(name)s - %(message)s',
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

mcp = FastMCP("huggingface-search")

# Configure hf_ops to use sandbox if available
_HF_SANDBOX_URL = os.getenv("HF_SANDBOX_URL", "").rstrip("/")
_HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
_HF_TOKEN = os.getenv("HF_TOKEN", "")


def _sandbox_call(endpoint: str, payload: dict, timeout: int = 30) -> dict | None:
    """Call local sandbox API; return parsed JSON on success, None on failure."""
    if not _HF_SANDBOX_URL:
        return None
    try:
        import httpx
        resp = httpx.post(f"{_HF_SANDBOX_URL}{endpoint}", json=payload, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def _configure_and_call(fn, *args, **kwargs):
    """Configure hf_ops and call the function, with sandbox proxy fallback."""
    from huggingface_search_cli.server.hf_ops import configure
    configure(_HF_ENDPOINT, _HF_TOKEN)
    return fn(*args, **kwargs)


@mcp.tool()
def search_datasets(query: str, limit: int = 100, author: Optional[str] = None):
    """Search for datasets on the HuggingFace Hub.

    Args:
        query: Search keyword(s) for datasets.
        limit: Maximum number of results to return. Default 100.
        author: Filter results by dataset author or organization.
    """
    sb = _sandbox_call("/search", {"query": query, "limit": limit, "author": author or ""})
    if sb is not None:
        return sb.get("text", str(sb))

    from huggingface_search_cli.server.hf_ops import search_datasets as _search
    return _configure_and_call(_search, query, limit, author)


@mcp.tool()
def inspect_dataset(dataset_id: str, config: Optional[str] = None):
    """Retrieve comprehensive metadata and structural information for a HuggingFace dataset.

    Args:
        dataset_id: The unique dataset identifier (e.g. 'openai/gsm8k').
        config: The configuration name to inspect.
    """
    sb = _sandbox_call("/inspect", {"dataset_id": dataset_id, "config": config})
    if sb is not None:
        return sb.get("text", str(sb))

    from huggingface_search_cli.server.hf_ops import inspect_dataset as _inspect
    return _configure_and_call(_inspect, dataset_id, config)


@mcp.tool()
def get_dataset_configs(dataset_id: str):
    """Retrieve all available configurations for a given dataset.

    Args:
        dataset_id: The unique dataset identifier.
    """
    sb = _sandbox_call("/configs", {"dataset_id": dataset_id})
    if sb is not None:
        text = sb.get("text", "")
        if text.startswith("["):
            import json
            try:
                return json.loads(text)
            except Exception:
                pass
        return text

    from huggingface_search_cli.server.hf_ops import get_dataset_configs as _configs
    return _configure_and_call(_configs, dataset_id)


@mcp.tool()
def get_dataset_splits(dataset_id: str, config: Optional[str] = None):
    """Retrieve all available data splits for a dataset configuration.

    Args:
        dataset_id: The unique dataset identifier.
        config: The configuration name.
    """
    sb = _sandbox_call("/splits", {"dataset_id": dataset_id, "config": config})
    if sb is not None:
        return sb.get("text", str(sb))

    from huggingface_search_cli.server.hf_ops import get_dataset_splits as _splits
    return _configure_and_call(_splits, dataset_id, config)


@mcp.tool()
def get_dataset_readme(dataset_id: str):
    """Retrieve the README.md or description for a HuggingFace dataset.

    Args:
        dataset_id: The unique dataset identifier.
    """
    sb = _sandbox_call("/readme", {"dataset_id": dataset_id})
    if sb is not None:
        return sb.get("text", str(sb))

    from huggingface_search_cli.server.hf_ops import get_dataset_readme as _readme
    return _configure_and_call(_readme, dataset_id)


@mcp.tool()
def get_dataset_sample(
    dataset_id: str,
    config: Optional[str] = None,
    split: Optional[str] = None,
    num_samples: int = 5,
):
    """Retrieve sample data records from a specific dataset.

    Args:
        dataset_id: The unique dataset identifier.
        config: The configuration name.
        split: The specific data split.
        num_samples: Number of sample records to retrieve. Default 5.
    """
    sb = _sandbox_call(
        "/sample",
        {"dataset_id": dataset_id, "config": config, "split": split, "num_samples": num_samples},
        timeout=120,
    )
    if sb is not None:
        return sb.get("text", str(sb))

    from huggingface_search_cli.server.hf_ops import get_dataset_sample as _sample
    return _configure_and_call(_sample, dataset_id, config, split, num_samples)


@mcp.tool()
def download_dataset(dataset_id: str, output_dir: str):
    """Download all raw files from a HuggingFace dataset repository.

    Args:
        dataset_id: The unique dataset identifier.
        output_dir: Local directory path for saved files.
    """
    sb = _sandbox_call(
        "/download",
        {"dataset_id": dataset_id, "output_dir": output_dir},
        timeout=600,
    )
    if sb is not None:
        return sb.get("text", str(sb))

    from huggingface_search_cli.server.hf_ops import download_dataset as _download
    return _configure_and_call(_download, dataset_id, output_dir)


if __name__ == "__main__":
    mcp.run()
