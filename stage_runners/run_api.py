"""CLI entry point for local REST API server."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from flight_fare.api_server import run_api_server  # noqa: E402
from flight_fare.settings import ApiSettings  # noqa: E402


def build_args() -> argparse.Namespace:
    """Build and parse command-line arguments for API runs."""
    parser = argparse.ArgumentParser(description="Run local REST API for flight fare predictions.")
    parser.add_argument("--package-dir", type=Path, default=None, help="Path to packaged model directory.")
    parser.add_argument("--host", type=str, default=None, help="Bind host.")
    parser.add_argument("--port", type=int, default=None, help="Bind port.")
    parser.add_argument("--request-timeout-seconds", type=float, default=None, help="Per-request timeout budget.")
    parser.add_argument("--predict-retries", type=int, default=None, help="Retries for packaged prediction loading.")
    parser.add_argument("--predict-timeout-seconds", type=float, default=None, help="Timeout per packaged prediction attempt.")
    parser.add_argument("--predict-retry-delay-seconds", type=float, default=None, help="Delay between prediction retries.")
    parser.add_argument("--log-level", type=str, default=None, help="Optional runtime log level.")
    return parser.parse_args()


def main() -> int:
    """Run API server with optional command-line overrides."""
    args = build_args()
    settings = ApiSettings.from_env()

    # Apply command-line overrides while preserving dataclass immutability.
    overrides: dict[str, object] = {}
    if args.package_dir is not None:
        overrides["package_dir"] = args.package_dir.resolve()
    if args.host is not None:
        overrides["host"] = args.host.strip()
    if args.port is not None:
        overrides["port"] = args.port
    if args.request_timeout_seconds is not None:
        overrides["request_timeout_seconds"] = args.request_timeout_seconds
    if args.predict_retries is not None:
        overrides["predict_retries"] = args.predict_retries
    if args.predict_timeout_seconds is not None:
        overrides["predict_timeout_seconds"] = args.predict_timeout_seconds
    if args.predict_retry_delay_seconds is not None:
        overrides["predict_retry_delay_seconds"] = args.predict_retry_delay_seconds
    if args.log_level is not None:
        overrides["log_level"] = args.log_level.upper()
    if overrides:
        settings = replace(settings, **overrides)

    run_api_server(settings=settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
