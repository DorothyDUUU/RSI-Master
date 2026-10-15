"""HuggingFace Search CLI – unified command-line interface.

Provides both direct HF operations and sandbox-server proxy mode.
Run `hf-search --help` for usage.

Modes:
  1. Server mode   – start a FastAPI service that proxies all HF API calls
                     with rate limiting, caching, and concurrency control.
  2. Client mode   – call HF operations directly from the CLI, or proxy
                     through a running sandbox server (when --url is set).

All CLI commands try the sandbox server first (if --url is provided or
HF_SANDBOX_URL is set), and fall back to direct HF API calls.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

import click

# ── Shared helpers ───────────────────────────────────────────────────


def _get_sandbox_url(ctx: click.Context) -> str:
    """Resolve the sandbox server URL from --url flag or HF_SANDBOX_URL env."""
    return ctx.obj.get("sandbox_url", "")


def _sandbox_call(endpoint: str, payload: dict, timeout: int = 30) -> dict | None:
    """Call sandbox API; return parsed JSON on success, None on failure."""
    url = _get_sandbox_url(click.get_current_context())
    if not url:
        return None
    try:
        import httpx
        resp = httpx.post(f"{url}{endpoint}", json=payload, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def _configure_hf_ops(ctx: click.Context) -> None:
    """Configure hf_ops with endpoint/token from ctx."""
    from huggingface_search_cli.server.hf_ops import configure
    endpoint = ctx.obj.get("hf_endpoint", os.getenv("HF_ENDPOINT", "https://huggingface.co"))
    token = ctx.obj.get("hf_token", os.getenv("HF_TOKEN", ""))
    configure(endpoint.rstrip("/"), token)


def _echo_result(text: str, output: Optional[str]) -> None:
    """Print result to stdout or write to file."""
    if output:
        from pathlib import Path
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(text, encoding="utf-8")
        click.echo(f"Result saved to {output}")
    else:
        click.echo(text)


# ── CLI group ────────────────────────────────────────────────────────

@click.group()
@click.option(
    "--url", envvar="HF_SANDBOX_URL", default="",
    help="Sandbox server URL (e.g. http://localhost:8899). If set, CLI proxies through this server first.",
)
@click.option(
    "--hf-endpoint", envvar="HF_ENDPOINT", default="https://huggingface.co",
    help="HuggingFace API endpoint (default: https://huggingface.co).",
)
@click.option(
    "--hf-token", envvar="HF_TOKEN", default="",
    help="HuggingFace API token for gated datasets.",
)
@click.pass_context
def cli(ctx, url, hf_endpoint, hf_token):
    """HuggingFace Search CLI – search, inspect, and manage HF datasets.

    Supports two modes:
      1. Direct mode: calls HF API directly (default)
      2. Proxy mode: routes requests through a sandbox server (--url)

    Use 'hf-search serve' to start the sandbox server.
    """
    ctx.ensure_object(dict)
    ctx.obj["sandbox_url"] = url.rstrip("/") if url else ""
    ctx.obj["hf_endpoint"] = hf_endpoint.rstrip("/")
    ctx.obj["hf_token"] = hf_token


# ── Server command ───────────────────────────────────────────────────

@cli.command()
@click.option("--host", envvar="HF_SANDBOX_HOST", default="0.0.0.0", help="Server host.")
@click.option("--port", envvar="HF_SANDBOX_PORT", default=8899, type=int, help="Server port.")
@click.option("--workers", default=1, type=int, help="Number of uvicorn workers.")
@click.option("--log-level", default="info", type=click.Choice(["debug", "info", "warning", "error"]))
def serve(host, port, workers, log_level):
    """Start the HF Sandbox Server (FastAPI + rate limiting).

    The server provides HTTP endpoints for all HF operations with built-in
    rate limiting, caching, and concurrency control. Other CLI instances
    can proxy through it using --url http://localhost:<port>.
    """
    import uvicorn
    from huggingface_search_cli.server import config as cfg

    # Override config with CLI flags
    cfg.HOST = host
    cfg.PORT = port

    click.echo(f"Starting HF Sandbox Server on {host}:{port} ...")
    click.echo(f"  Rate limit: {cfg.RATE_LIMIT_RPM} RPM")
    click.echo(f"  Max concurrent: {cfg.MAX_CONCURRENT}")
    click.echo(f"  HF endpoint: {cfg.HF_ENDPOINT}")
    click.echo(f"  Cache TTL: {cfg.SEARCH_CACHE_TTL}s")
    click.echo()

    uvicorn.run(
        "huggingface_search_cli.server.server:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        timeout_keep_alive=620,
    )


# ── Health check ─────────────────────────────────────────────────────

@cli.command()
@click.pass_context
def health(ctx):
    """Check sandbox server health status."""
    url = _get_sandbox_url(ctx)
    if not url:
        click.echo("No sandbox server URL configured. Use --url or set HF_SANDBOX_URL.", err=True)
        sys.exit(1)

    try:
        import httpx
        resp = httpx.get(f"{url}/health", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            click.echo(json.dumps(data, indent=2))
        else:
            click.echo(f"Server returned status {resp.status_code}", err=True)
            sys.exit(1)
    except Exception as e:
        click.echo(f"Failed to connect to sandbox server: {e}", err=True)
        sys.exit(1)


# ── Search ───────────────────────────────────────────────────────────

@cli.command()
@click.argument("query")
@click.option("--limit", default=100, type=int, help="Max number of results.")
@click.option("--author", default=None, help="Filter by author/organization.")
@click.option("-o", "--output", default=None, help="Save result to file instead of stdout.")
@click.pass_context
def search(ctx, query, limit, author, output):
    """Search for datasets on HuggingFace Hub.

    QUERY is the search keyword(s). Use specific terms for better results.
    HuggingFace search does NOT support semantic or fuzzy search.
    """
    # Try sandbox first
    sb = _sandbox_call("/search", {"query": query, "limit": limit, "author": author or ""})
    if sb is not None:
        _echo_result(sb.get("text", str(sb)), output)
        return

    # Fallback: direct HF API
    _configure_hf_ops(ctx)
    from huggingface_search_cli.server.hf_ops import search_datasets
    result = search_datasets(query, limit, author)
    _echo_result(result, output)


# ── Inspect ──────────────────────────────────────────────────────────

@cli.command()
@click.argument("dataset_id")
@click.option("--config", default=None, help="Dataset configuration name.")
@click.option("-o", "--output", default=None, help="Save result to file.")
@click.pass_context
def inspect(ctx, dataset_id, config, output):
    """Inspect a dataset's comprehensive metadata.

    DATASET_ID is the HuggingFace dataset identifier (e.g. 'openai/gsm8k').
    Returns YAML-formatted metadata including features, splits, description, etc.
    """
    sb = _sandbox_call("/inspect", {"dataset_id": dataset_id, "config": config})
    if sb is not None:
        _echo_result(sb.get("text", str(sb)), output)
        return

    _configure_hf_ops(ctx)
    from huggingface_search_cli.server.hf_ops import inspect_dataset
    result = inspect_dataset(dataset_id, config)
    _echo_result(result, output)


# ── Configs ──────────────────────────────────────────────────────────

@cli.command()
@click.argument("dataset_id")
@click.option("-o", "--output", default=None, help="Save result to file.")
@click.pass_context
def configs(ctx, dataset_id, output):
    """List available configurations for a dataset.

    DATASET_ID is the HuggingFace dataset identifier.
    Returns JSON list of config names or a default-config message.
    """
    sb = _sandbox_call("/configs", {"dataset_id": dataset_id})
    if sb is not None:
        text = sb.get("text", "")
        _echo_result(text, output)
        return

    _configure_hf_ops(ctx)
    from huggingface_search_cli.server.hf_ops import get_dataset_configs
    result = get_dataset_configs(dataset_id)
    _echo_result(result, output)


# ── Splits ───────────────────────────────────────────────────────────

@cli.command()
@click.argument("dataset_id")
@click.option("--config", default=None, help="Dataset configuration name.")
@click.option("-o", "--output", default=None, help="Save result to file.")
@click.pass_context
def splits(ctx, dataset_id, config, output):
    """List available splits for a dataset.

    DATASET_ID is the HuggingFace dataset identifier.
    """
    sb = _sandbox_call("/splits", {"dataset_id": dataset_id, "config": config})
    if sb is not None:
        _echo_result(sb.get("text", str(sb)), output)
        return

    _configure_hf_ops(ctx)
    from huggingface_search_cli.server.hf_ops import get_dataset_splits
    result = get_dataset_splits(dataset_id, config)
    _echo_result(result, output)


# ── README ───────────────────────────────────────────────────────────

@cli.command()
@click.argument("dataset_id")
@click.option("-o", "--output", default=None, help="Save result to file.")
@click.pass_context
def readme(ctx, dataset_id, output):
    """Get the README.md or description for a dataset.

    DATASET_ID is the HuggingFace dataset identifier.
    """
    sb = _sandbox_call("/readme", {"dataset_id": dataset_id})
    if sb is not None:
        _echo_result(sb.get("text", str(sb)), output)
        return

    _configure_hf_ops(ctx)
    from huggingface_search_cli.server.hf_ops import get_dataset_readme
    result = get_dataset_readme(dataset_id)
    _echo_result(result, output)


# ── Sample ───────────────────────────────────────────────────────────

@cli.command()
@click.argument("dataset_id")
@click.option("--config", default=None, help="Dataset configuration name.")
@click.option("--split", default=None, help="Data split (e.g. train, test).")
@click.option("-n", "--num-samples", default=5, type=int, help="Number of samples to retrieve.")
@click.option("-o", "--output", default=None, help="Save result to file.")
@click.pass_context
def sample(ctx, dataset_id, config, split, num_samples, output):
    """Get sample data records from a dataset.

    DATASET_ID is the HuggingFace dataset identifier.
    Shows column names, types, and the specified number of sample records.
    """
    sb = _sandbox_call(
        "/sample",
        {"dataset_id": dataset_id, "config": config, "split": split, "num_samples": num_samples},
        timeout=120,
    )
    if sb is not None:
        _echo_result(sb.get("text", str(sb)), output)
        return

    _configure_hf_ops(ctx)
    from huggingface_search_cli.server.hf_ops import get_dataset_sample
    result = get_dataset_sample(dataset_id, config, split, num_samples)
    _echo_result(result, output)


# ── Download ─────────────────────────────────────────────────────────

@cli.command()
@click.argument("dataset_id")
@click.option("-d", "--output-dir", default="./downloaded_datasets", help="Output directory for downloaded files.")
@click.option("-o", "--output", default=None, help="Save summary to file.")
@click.pass_context
def download(ctx, dataset_id, output_dir, output):
    """Download all raw files from a dataset repository.

    DATASET_ID is the HuggingFace dataset identifier.
    Downloads preserve original file format and directory structure.
    """
    sb = _sandbox_call(
        "/download",
        {"dataset_id": dataset_id, "output_dir": output_dir},
        timeout=600,
    )
    if sb is not None:
        _echo_result(sb.get("text", str(sb)), output)
        return

    _configure_hf_ops(ctx)
    from huggingface_search_cli.server.hf_ops import download_dataset
    result = download_dataset(dataset_id, output_dir)
    _echo_result(result, output)


# ── Materialize ──────────────────────────────────────────────────────

@cli.command()
@click.argument("dataset_id")
@click.option("--config", default=None, help="Dataset configuration name.")
@click.option("--split", default=None, help="Data split (e.g. train, test).")
@click.option("--max-rows", default=2048, type=int, help="Maximum number of rows to materialize.")
@click.option("--output-dir", default="", help="Output directory for JSONL file (default: shared cache).")
@click.option("-o", "--output", default=None, help="Save result summary to file.")
@click.pass_context
def materialize(ctx, dataset_id, config, split, max_rows, output_dir, output):
    """Materialize a dataset to JSONL format.

    DATASET_ID is the HuggingFace dataset identifier.
    Converts the dataset to a JSONL file with one JSON object per row.
    Cached results are reused automatically.
    """
    sb = _sandbox_call(
        "/materialize",
        {"dataset": dataset_id, "config": config, "split": split,
         "max_rows": max_rows, "output_dir": output_dir},
        timeout=600,
    )
    if sb is not None:
        _echo_result(json.dumps(sb, indent=2), output)
        return

    _configure_hf_ops(ctx)
    from huggingface_search_cli.server.hf_ops import materialize_dataset
    result = materialize_dataset(dataset_id, config, split, max_rows, output_dir)
    _echo_result(json.dumps(result, indent=2), output)


# ── Entry point ──────────────────────────────────────────────────────

def main():
    cli(obj={})


if __name__ == "__main__":
    main()
