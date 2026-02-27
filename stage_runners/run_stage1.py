"""CLI entry point for Stage 1 execution."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from flight_fare import Stage1Settings, run_stage_1  # noqa: E402


def build_args() -> argparse.Namespace:
    """Build and parse command-line arguments for Stage 1 runs."""
    parser = argparse.ArgumentParser(description="Run Stage 1 pipeline for flight fare project.")
    parser.add_argument("--dataset-path", type=Path, default=None, help="Optional path to input dataset CSV.")
    parser.add_argument("--log-level", type=str, default=None, help="Optional runtime log level.")
    parser.add_argument("--retries", type=int, default=3, help="Number of dataset load retries.")
    parser.add_argument("--timeout-seconds", type=float, default=30.0, help="CSV load timeout per attempt.")
    parser.add_argument("--retry-delay-seconds", type=float, default=1.5, help="Delay between load retries.")
    return parser.parse_args()


def main() -> int:
    """Run Stage 1 and print artifact locations on success."""
    args = build_args()
    settings = Stage1Settings.from_env()

    # Apply optional command-line overrides without mutating immutable settings.
    overrides: dict[str, object] = {}
    if args.dataset_path is not None:
        overrides["dataset_path"] = args.dataset_path.resolve()
    if args.log_level is not None:
        overrides["log_level"] = args.log_level.upper()
    if overrides:
        settings = replace(settings, **overrides)

    outputs = run_stage_1(
        settings=settings,
        retries=args.retries,
        timeout_seconds=args.timeout_seconds,
        retry_delay_seconds=args.retry_delay_seconds,
    )
    print(f"Stage 1 completed. Profile: {outputs.profile_path}")
    print(f"Problem definition: {outputs.problem_definition_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
