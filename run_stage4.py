"""CLI entry point for Stage 4 baseline model pipeline."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.flight_fare.settings import Stage4Settings  # noqa: E402
from src.flight_fare.stage4_baseline import run_stage_4  # noqa: E402


def build_args() -> argparse.Namespace:
    """Build and parse command-line arguments for Stage 4 runs."""
    parser = argparse.ArgumentParser(description="Run Stage 4 baseline linear regression pipeline.")
    parser.add_argument("--x-train-path", type=Path, default=None, help="Path to processed training feature CSV.")
    parser.add_argument("--x-test-path", type=Path, default=None, help="Path to processed test feature CSV.")
    parser.add_argument("--y-train-path", type=Path, default=None, help="Path to training target CSV.")
    parser.add_argument("--y-test-path", type=Path, default=None, help="Path to test target CSV.")
    parser.add_argument("--target-column", type=str, default=None, help="Target column name in y CSV files.")
    parser.add_argument("--log-level", type=str, default=None, help="Optional runtime log level.")
    parser.add_argument("--retries", type=int, default=3, help="Input load retries and model fit retries.")
    parser.add_argument("--timeout-seconds", type=float, default=30.0, help="Input/model operation timeout per attempt.")
    parser.add_argument("--retry-delay-seconds", type=float, default=1.5, help="Delay between retries.")
    parser.add_argument("--io-retries", type=int, default=3, help="Artifact write retries.")
    parser.add_argument("--io-timeout-seconds", type=float, default=30.0, help="Artifact write timeout per attempt.")
    parser.add_argument("--io-retry-delay-seconds", type=float, default=1.0, help="Delay between artifact write retries.")
    return parser.parse_args()


def main() -> int:
    """Run Stage 4 and print generated artifact paths."""
    args = build_args()
    settings = Stage4Settings.from_env()

    # Apply optional command-line overrides while preserving immutability.
    overrides: dict[str, object] = {}
    if args.x_train_path is not None:
        overrides["x_train_path"] = args.x_train_path.resolve()
    if args.x_test_path is not None:
        overrides["x_test_path"] = args.x_test_path.resolve()
    if args.y_train_path is not None:
        overrides["y_train_path"] = args.y_train_path.resolve()
    if args.y_test_path is not None:
        overrides["y_test_path"] = args.y_test_path.resolve()
    if args.target_column is not None:
        overrides["target_column"] = args.target_column.strip()
    if args.log_level is not None:
        overrides["log_level"] = args.log_level.upper()
    if overrides:
        settings = replace(settings, **overrides)

    outputs = run_stage_4(
        settings=settings,
        retries=args.retries,
        timeout_seconds=args.timeout_seconds,
        retry_delay_seconds=args.retry_delay_seconds,
        io_retries=args.io_retries,
        io_timeout_seconds=args.io_timeout_seconds,
        io_retry_delay_seconds=args.io_retry_delay_seconds,
    )

    print("Stage 4 completed.")
    print(f"Metrics: {outputs.metrics_path}")
    print(f"Predictions: {outputs.predictions_path}")
    print(f"Model: {outputs.model_path}")
    print(f"Summary: {outputs.summary_path}")
    print(f"Coefficients: {outputs.coefficients_path}")
    print(f"Plots directory: {outputs.plots_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
