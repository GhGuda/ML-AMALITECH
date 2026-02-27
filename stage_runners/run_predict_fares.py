"""CLI entry point for batch inference using Stage 7 packaged artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd  # noqa: E402

from flight_fare.stage7_delivery import predict_from_package  # noqa: E402


def build_args() -> argparse.Namespace:
    """Build and parse command-line arguments for batch fare prediction."""
    parser = argparse.ArgumentParser(description="Run batch fare prediction from a Stage 7 package.")
    parser.add_argument("--package-dir", type=Path, required=True, help="Path to packaged model directory.")
    parser.add_argument("--input-csv", type=Path, required=True, help="Path to input CSV for predictions.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Path to output CSV for predictions.")
    parser.add_argument("--retries", type=int, default=3, help="Load/prediction retries.")
    parser.add_argument("--timeout-seconds", type=float, default=30.0, help="Load/prediction timeout per attempt.")
    parser.add_argument("--retry-delay-seconds", type=float, default=1.0, help="Delay between retries.")
    return parser.parse_args()


def main() -> int:
    """Load input records, predict fares, and persist prediction output."""
    args = build_args()
    input_df = pd.read_csv(args.input_csv)
    predictions_df = predict_from_package(
        package_dir=args.package_dir.resolve(),
        input_df=input_df,
        retries=args.retries,
        timeout_seconds=args.timeout_seconds,
        retry_delay_seconds=args.retry_delay_seconds,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(args.output_csv, index=False)
    print(f"Predictions saved to: {args.output_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
