"""CLI entry point for Stage 6 interpretation and stakeholder insight pipeline."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from flight_fare.settings import Stage6Settings  # noqa: E402
from flight_fare.stage6_interpretation import run_stage_6  # noqa: E402


def build_args() -> argparse.Namespace:
    """Build and parse command-line arguments for Stage 6 runs."""
    parser = argparse.ArgumentParser(description="Run Stage 6 model interpretation and insight pipeline.")
    parser.add_argument("--cleaned-dataset-path", type=Path, default=None, help="Path to cleaned dataset from Stage 2.")
    parser.add_argument("--stage5-summary-path", type=Path, default=None, help="Path to Stage 5 summary JSON.")
    parser.add_argument("--feature-impact-path", type=Path, default=None, help="Path to Stage 5 feature impact CSV.")
    parser.add_argument("--target-column", type=str, default=None, help="Target fare column in cleaned dataset.")
    parser.add_argument("--top-feature-count", type=int, default=None, help="Number of top features to include in interpretation.")
    parser.add_argument("--top-route-season-count", type=int, default=None, help="Number of top route-season hotspots.")
    parser.add_argument("--min-route-season-flights", type=int, default=None, help="Minimum flights per route-season hotspot.")
    parser.add_argument("--log-level", type=str, default=None, help="Optional runtime log level.")
    parser.add_argument("--retries", type=int, default=3, help="Load/compute retries.")
    parser.add_argument("--timeout-seconds", type=float, default=60.0, help="Load/compute timeout per attempt.")
    parser.add_argument("--retry-delay-seconds", type=float, default=1.5, help="Delay between retries.")
    parser.add_argument("--io-retries", type=int, default=3, help="Artifact write retries.")
    parser.add_argument("--io-timeout-seconds", type=float, default=30.0, help="Artifact write timeout per attempt.")
    parser.add_argument("--io-retry-delay-seconds", type=float, default=1.0, help="Delay between artifact write retries.")
    return parser.parse_args()


def main() -> int:
    """Run Stage 6 and print generated artifact locations."""
    args = build_args()
    settings = Stage6Settings.from_env()

    # Apply command-line overrides without mutating immutable settings.
    overrides: dict[str, object] = {}
    if args.cleaned_dataset_path is not None:
        overrides["cleaned_dataset_path"] = args.cleaned_dataset_path.resolve()
    if args.stage5_summary_path is not None:
        overrides["stage5_summary_path"] = args.stage5_summary_path.resolve()
    if args.feature_impact_path is not None:
        overrides["feature_impact_path"] = args.feature_impact_path.resolve()
    if args.target_column is not None:
        overrides["target_column"] = args.target_column.strip()
    if args.top_feature_count is not None:
        overrides["top_feature_count"] = args.top_feature_count
    if args.top_route_season_count is not None:
        overrides["top_route_season_count"] = args.top_route_season_count
    if args.min_route_season_flights is not None:
        overrides["min_route_season_flights"] = args.min_route_season_flights
    if args.log_level is not None:
        overrides["log_level"] = args.log_level.upper()
    if overrides:
        settings = replace(settings, **overrides)

    outputs = run_stage_6(
        settings=settings,
        retries=args.retries,
        timeout_seconds=args.timeout_seconds,
        retry_delay_seconds=args.retry_delay_seconds,
        io_retries=args.io_retries,
        io_timeout_seconds=args.io_timeout_seconds,
        io_retry_delay_seconds=args.io_retry_delay_seconds,
    )

    print("Stage 6 completed.")
    print(f"Summary report: {outputs.summary_report_path}")
    print(f"Stakeholder report: {outputs.stakeholder_report_path}")
    print(f"Top feature drivers: {outputs.top_feature_table_path}")
    print(f"Airline fare summary: {outputs.airline_fare_summary_path}")
    print(f"Seasonal pricing summary: {outputs.seasonal_pricing_path}")
    print(f"Route-season hotspots: {outputs.route_season_hotspots_path}")
    print(f"Plots directory: {outputs.plots_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
