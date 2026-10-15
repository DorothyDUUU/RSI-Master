"""HF Sandbox Server configuration – all values from env vars with defaults."""

from __future__ import annotations

import os


# ── HuggingFace ──────────────────────────────────────────────────────
HF_ENDPOINT: str = os.getenv("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
HF_TOKEN: str = os.getenv("HF_TOKEN", "")
HF_HOME: str = os.getenv("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
HF_DATASETS_CACHE: str = os.getenv(
    "HF_DATASETS_CACHE", os.path.join(HF_HOME, "datasets")
)

# ── Shared cache (materialized JSONL files) ─────────────────────────
SHARED_CACHE_DIR: str = os.getenv(
    "SHARED_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "hf_sandbox", "materialized"),
)

# ── Rate limiting ────────────────────────────────────────────────────
RATE_LIMIT_RPM: int = int(os.getenv("HF_RATE_LIMIT_RPM", "30"))
MAX_CONCURRENT: int = int(os.getenv("HF_MAX_CONCURRENT", "3"))

# ── Search cache ─────────────────────────────────────────────────────
SEARCH_CACHE_TTL: int = int(os.getenv("HF_SEARCH_CACHE_TTL", "600"))  # seconds

# ── Server ───────────────────────────────────────────────────────────
HOST: str = os.getenv("HF_SANDBOX_HOST", "0.0.0.0")
PORT: int = int(os.getenv("HF_SANDBOX_PORT", "8899"))
