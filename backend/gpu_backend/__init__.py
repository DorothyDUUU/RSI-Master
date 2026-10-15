"""GPU Backend package — HTTP client + CLI + config."""

from .client import BackendClient, BackendError
from .cli import main as cli_main
from .config import load_config

__all__ = ["BackendClient", "BackendError", "cli_main", "load_config"]
