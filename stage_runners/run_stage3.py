"""CLI entry point for Stage 3 exploratory analysis pipeline."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from flight_fare.settings import Stage3Settings  # noqa: E402
from flight_fare.stage3_eda import run_stage_3  # noqa: E402


def build_args() -> argparse.Namespace:
    """Build and parse command-line arguments for Stage 3 runs."""
    parser = argparse.ArgumentParser(description="Run Stage 3 EDA and KPI pipeline.")
    parser.add_argument("--dataset-path", type=Path, default=None, help="Optional path to Stage 3 input dataset CSV.")
    parser.add_argument("--log-level", type=str, default=None, help="Optional runtime log level.")
    parser.add_argument("--top-n-routes", type=int, default=None, help="Number of top expensive routes to include.")
    parser.add_argument("--max-airlines-boxplot", type=int, default=None, help="Maximum routes to include in fare boxplot.")
    parser.add_argument("--retries", type=int, default=3, help="Number of dataset load retries.")
    parser.add_argument("--timeout-seconds", type=float, default=30.0, help="Dataset load timeout per attempt.")
    parser.add_argument("--retry-delay-seconds", type=float, default=1.5, help="Delay between load retries.")
    parser.add_argument("--io-retries", type=int, default=3, help="Artifact write retries.")
    parser.add_argument("--io-timeout-seconds", type=float, default=30.0, help="Artifact write timeout per attempt.")
    parser.add_argument("--io-retry-delay-seconds", type=float, default=1.0, help="Delay between artifact write retries.")
    return parser.parse_args()


def main() -> int:
    """Run Stage 3 and print generated artifact locations."""
    args = build_args()
    settings = Stage3Settings.from_env()

    # Apply optional command-line overrides while preserving dataclass immutability.
    overrides: dict[str, object] = {}
    if args.dataset_path is not None:
        overrides["dataset_path"] = args.dataset_path.resolve()
    if args.log_level is not None:
        overrides["log_level"] = args.log_level.upper()
    if args.top_n_routes is not None:
        overrides["top_n_routes"] = args.top_n_routes
    if args.max_airlines_boxplot is not None:
        overrides["max_airlines_boxplot"] = args.max_airlines_boxplot
    if overrides:
        settings = replace(settings, **overrides)

    outputs = run_stage_3(
        settings=settings,
        retries=args.retries,
        timeout_seconds=args.timeout_seconds,
        retry_delay_seconds=args.retry_delay_seconds,
        io_retries=args.io_retries,
        io_timeout_seconds=args.io_timeout_seconds,
        io_retry_delay_seconds=args.io_retry_delay_seconds,
    )

    print("Stage 3 completed.")
    print(f"Summary report: {outputs.summary_report_path}")
    print(f"Descriptive stats: {outputs.descriptive_stats_path}")
    print(f"Correlation matrix: {outputs.correlation_matrix_path}")
    print(f"Average fare per route: {outputs.average_fare_by_route_path}")
    print(f"Route frequency: {outputs.route_frequency_path}")
    print(f"Seasonal fare: {outputs.seasonal_fare_path}")
    print(f"Booking window fare: {outputs.booking_window_fare_path}")
    print(f"Top expensive routes: {outputs.top_expensive_routes_path}")
    print(f"Top affordable routes: {outputs.top_affordable_routes_path}")
    print(f"Plots directory: {outputs.plots_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
