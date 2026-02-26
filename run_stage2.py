"""CLI entry point for Stage 2 preprocessing pipeline."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.flight_fare.settings import Stage2Settings  # noqa: E402
from src.flight_fare.stage2_pipeline import run_stage_2  # noqa: E402


def build_args() -> argparse.Namespace:
    """Build and parse command-line arguments for Stage 2 runs."""
    parser = argparse.ArgumentParser(description="Run Stage 2 preprocessing pipeline.")
    parser.add_argument("--dataset-path", type=Path, default=None, help="Optional path to input dataset CSV.")
    parser.add_argument("--log-level", type=str, default=None, help="Optional runtime log level.")
    parser.add_argument("--test-size", type=float, default=None, help="Optional test split size in (0, 1).")
    parser.add_argument("--random-state", type=int, default=None, help="Optional random seed for data split.")
    parser.add_argument("--retries", type=int, default=3, help="Number of dataset load retries.")
    parser.add_argument("--timeout-seconds", type=float, default=30.0, help="CSV load timeout per attempt.")
    parser.add_argument("--retry-delay-seconds", type=float, default=1.5, help="Delay between load retries.")
    parser.add_argument("--io-retries", type=int, default=3, help="Artifact write retries.")
    parser.add_argument("--io-timeout-seconds", type=float, default=30.0, help="Artifact write timeout per attempt.")
    parser.add_argument("--io-retry-delay-seconds", type=float, default=1.0, help="Delay between artifact write retries.")
    return parser.parse_args()


def main() -> int:
    """Run Stage 2 and print generated artifact locations."""
    args = build_args()
    settings = Stage2Settings.from_env()

    # Apply optional command-line overrides while preserving immutability.
    overrides: dict[str, object] = {}
    if args.dataset_path is not None:
        overrides["dataset_path"] = args.dataset_path.resolve()
    if args.log_level is not None:
        overrides["log_level"] = args.log_level.upper()
    if args.test_size is not None:
        overrides["test_size"] = args.test_size
    if args.random_state is not None:
        overrides["random_state"] = args.random_state
    if overrides:
        settings = replace(settings, **overrides)

    outputs = run_stage_2(
        settings=settings,
        retries=args.retries,
        timeout_seconds=args.timeout_seconds,
        retry_delay_seconds=args.retry_delay_seconds,
        io_retries=args.io_retries,
        io_timeout_seconds=args.io_timeout_seconds,
        io_retry_delay_seconds=args.io_retry_delay_seconds,
    )

    print("Stage 2 completed.")
    print(f"Cleaned dataset: {outputs.cleaned_dataset_path}")
    print(f"X train (processed): {outputs.x_train_processed_path}")
    print(f"X test (processed): {outputs.x_test_processed_path}")
    print(f"y train: {outputs.y_train_path}")
    print(f"y test: {outputs.y_test_path}")
    print(f"Preprocessor: {outputs.preprocessor_path}")
    print(f"Report: {outputs.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
