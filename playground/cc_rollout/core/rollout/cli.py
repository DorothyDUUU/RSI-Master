"""Command-line interface: ``python -m playground.cc_rollout.core.rollout {run,health}``."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from .orchestrator import run as orchestrator_run

logger = logging.getLogger(__name__)


def _default_config() -> str:
    """Default config for `run`: rsi_master configs/cc_rollout/config.yaml."""
    from ._paths import rsi_master_root

    default = rsi_master_root() / "configs" / "cc_rollout" / "config.yaml"
    if default.exists():
        return str(default)
    raise SystemExit(f"default config not found: {default} (pass --config)")


def _load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    if not path.exists():
        raise SystemExit(f"config not found: {path}")
    return yaml.safe_load(path.read_text())


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------

def cmd_health(args: argparse.Namespace) -> int:
    _ = _load_config(args.config)
    # 与 submit_pipeline.driver 用同一个 URL 解析 (backend/gpu_backend/config.yaml
    # → http://localhost:8989), health 报告的就是 pipeline 将要打的那个 backend。
    try:
        from backend.gpu_backend.client import BackendClient
        from backend.gpu_backend.config import get_backend_url

        client = BackendClient(base_url=get_backend_url(), timeout=15.0)
        print(json.dumps(client.health_check(), indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001 — health 就是要报告不可达
        print(json.dumps({"status": "unreachable", "error": str(exc)}, indent=2))
        return 1


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    name = args.name or cfg.get("run", {}).get("run_id") or cfg.get("run", {}).get("name") or None
    run_root = orchestrator_run(args.config, run_name=name)
    print(f"\nrun complete: {run_root}\nsummary: {run_root / 'summary.md'}")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cc_rollout", description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name, help_text in [
        ("run", "run a full cc_rollout (one CC; config mode selects naive / with_expos)"),
        ("health", "check GPU backend health"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", default=_default_config())
        if name == "run":
            p.add_argument("--name", default=None,
                           help="run-id override (default: run.run_id from config; else random hash)")

    args = parser.parse_args(argv)

    # verbose can come from the yaml (top-level `verbose:`) or the CLI flag.
    cfg_verbose = False
    if getattr(args, "config", None):
        try:
            cfg_verbose = bool(_load_config(args.config).get("verbose", False))
        except SystemExit:
            cfg_verbose = False  # config missing → rely on the flag
    _setup_logging(args.verbose or cfg_verbose)

    return {"run": cmd_run, "health": cmd_health}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
