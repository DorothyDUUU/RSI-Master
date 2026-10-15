# HuggingFace Search CLI

Unified CLI + server for HuggingFace dataset search, inspection, download, and materialization.

Converted from `mle_bench_datamaster/hf_sandbox_server` (FastAPI backend) and `mle_bench_datamaster/search_dataset_tools/search_huggingface.py` (MCP tool), combined into a single coherent CLI tool.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  hf-search CLI                                                  │
│  ┌──────────────┐    ┌───────────────────────────────────────┐  │
│  │  CLI Frontend │    │  Server Backend (FastAPI)             │  │
│  │  (click)      │    │  ┌───────────┐  ┌─────────────────┐  │  │
│  │              │    │  │ Rate Limit │  │ Search Cache    │  │  │
│  │  search      │───>│  │ 30 RPM    │  │ TTL 600s        │  │  │
│  │  inspect     │    │  ├───────────┤  ├─────────────────┤  │  │
│  │  configs     │    │  │ Concurrency│  │ In-memory LRU   │  │  │
│  │  splits      │    │  │ max=3     │  │ max 500 entries │  │  │
│  │  readme      │    │  └───────────┘  └─────────────────┘  │  │
│  │  sample      │    │                  ┌─────────────────┐  │  │
│  │  download    │    │                  │ hf_ops.py       │  │  │
│  │  materialize │    │                  │ (actual HF API) │  │  │
│  │              │    │                  └─────────────────┘  │  │
│  │  serve       │    │                                       │  │
│  │  health      │    │  Endpoints:                            │  │
│  └──────────────┘    │  POST /search, /inspect, /configs,    │  │
│                      │        /splits, /readme, /sample,    │  │
│  ──urllocalhost:8899──>        /download, /materialize       │  │
│  (proxy mode)        │  GET  /health                         │  │
│                      └───────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Two Modes

1. **Direct mode** (default): CLI calls HF API directly — no server needed.
2. **Proxy mode** (`--url`): CLI routes requests through a running sandbox server, which provides rate limiting, caching, and concurrency control. Falls back to direct HF API if the server is unavailable.

## Installation

```bash
cd huggingface_search_cli
pip install -e .
# Or just install dependencies:
pip install -r requirements.txt
```

## Quick Start

```bash
# Direct search (no server needed)
hf-search search "sentiment analysis" --limit 5

# Inspect a dataset
hf-search inspect openai/gsm8k

# Get dataset configs
hf-search configs PolyAI/minds14

# Get dataset splits
hf-search splits rotten_tomatoes

# Get README
hf-search readme openai/gsm8k

# Get sample data
hf-search sample rotten_tomatoes --split train -n 3

# Download dataset files
hf-search download glue --output-dir ./data/glue

# Materialize to JSONL
hf-search materialize openai/gsm8k --max-rows 1000 --output-dir ./jsonl_cache
```

## Server Mode

### Start Server

```bash
# Method 1: Using the CLI
hf-search serve --host 0.0.0.0 --port 8899

# Method 2: Using the management script
./start_server.sh start
./start_server.sh status
./start_server.sh stop
./start_server.sh restart

# Method 3: Direct Python
python -m huggingface_search_cli.server.server
```

### Use Server from CLI

```bash
# Set URL via flag (takes priority over env var)
hf-search --url http://localhost:8899 search "math"

# Or set environment variable
export HF_SANDBOX_URL=http://localhost:8899
hf-search search "math"
hf-search inspect openai/gsm8k
hf-search health
```

### API Endpoints

| Endpoint     | Method | Description            | Timeout |
|-------------|--------|------------------------|---------|
| `/health`    | GET   | Health check + metrics  | 5s      |
| `/search`    | POST  | Search datasets (cached)| 30s     |
| `/inspect`   | POST  | Dataset metadata        | 30s     |
| `/configs`   | POST  | List configurations     | 30s     |
| `/splits`    | POST  | List data splits        | 30s     |
| `/readme`    | POST  | Get README/description  | 30s     |
| `/sample`    | POST  | Get sample records      | 120s    |
| `/download`  | POST  | Download raw files      | 600s    |
| `/materialize`| POST | Materialize to JSONL    | 600s    |

### API Examples

```bash
# Health check
curl http://localhost:8899/health

# Search
curl -X POST http://localhost:8899/search \
  -H "Content-Type: application/json" \
  -d '{"query": "math", "limit": 3}'

# Inspect
curl -X POST http://localhost:8899/inspect \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": "openai/gsm8k"}'

# Sample
curl -X POST http://localhost:8899/sample \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": "rotten_tomatoes", "split": "train", "num_samples": 3}'

# Materialize
curl -X POST http://localhost:8899/materialize \
  -H "Content-Type: application/json" \
  -d '{"dataset": "openai/gsm8k", "max_rows": 500}'
```

## Configuration

All configuration is via environment variables:

| Variable              | Default                   | Description                     |
|-----------------------|---------------------------|---------------------------------|
| `HF_ENDPOINT`         | `https://huggingface.co`  | HF API endpoint (use mirror for CN) |
| `HF_TOKEN`            | `""`                      | HuggingFace API token           |
| `HF_HOME`             | `~/.cache/huggingface`    | HF home directory               |
| `HF_DATASETS_CACHE`   | `$HF_HOME/datasets`       | Dataset cache directory         |
| `HF_SANDBOX_URL`      | `""`                      | Sandbox server URL for proxy mode |
| `HF_SANDBOX_HOST`     | `0.0.0.0`                 | Server listen address           |
| `HF_SANDBOX_PORT`     | `8899`                    | Server listen port              |
| `HF_RATE_LIMIT_RPM`   | `30`                      | Max requests per minute (server)|
| `HF_MAX_CONCURRENT`   | `3`                       | Max concurrent HF requests (server) |
| `HF_SEARCH_CACHE_TTL` | `600`                     | Search cache TTL seconds (server) |
| `SHARED_CACHE_DIR`    | `~/.cache/hf_sandbox/materialized` | Materialized JSONL cache dir |

## File Structure

```
huggingface_search_cli/
├── __init__.py              # Package init with version
├── __main__.py              # python -m entry point
├── cli/
│   ├── __init__.py
│   └── main.py              # Click CLI frontend (hf-search commands)
├── server/
│   ├── __init__.py
│   ├── config.py            # Configuration from env vars
│   ├── server.py            # FastAPI server with rate limiting & cache
│   ├── hf_ops.py            # Core HF operations (search/inspect/download/materialize)
│   └── rate_limiter.py      # Token-bucket rate limiter
├── start_server.sh          # Server management script (start/stop/restart/status)
├── setup.py                 # Development install script
├── pyproject.toml           # Modern Python packaging
├── requirements.txt         # Pip dependencies
└── README.md                # This file
```

## Migration from Original Code

| Original Source                                   | CLI Equivalent                    |
|---------------------------------------------------|-----------------------------------|
| `hf_sandbox_server/server.py`                     | `server/server.py` + `hf-search serve` |
| `hf_sandbox_server/hf_ops.py`                     | `server/hf_ops.py` (shared by both) |
| `hf_sandbox_server/config.py`                     | `server/config.py`                |
| `hf_sandbox_server/rate_limiter.py`               | `server/rate_limiter.py`          |
| `hf_sandbox_server/start_sandbox.sh`              | `start_server.sh`                 |
| `search_huggingface.py` (MCP tool, sandbox proxy) | `cli/main.py` (--url proxy mode)  |

Key changes from originals:
- **Unified entry point**: `hf-search` CLI covers both server management and HF operations
- **Proxy fallback**: CLI tries sandbox server first, falls back to direct API (same pattern as search_huggingface.py)
- **No MCP dependency**: Removed FastMCP dependency; CLI uses Click instead
- **Defaults updated**: `HF_ENDPOINT` defaults to `https://huggingface.co` (was `https://hf-mirror.com` in original)
- **Server runs as proper daemon**: `start_server.sh` uses `nohup` instead of `tee` foreground mode
