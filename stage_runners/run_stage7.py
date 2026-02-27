"""CLI entry point for Stage 7 packaging and delivery pipeline."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from flight_fare.settings import Stage7Settings  # noqa: E402
from flight_fare.stage7_delivery import run_stage_7  # noqa: E402


def build_args() -> argparse.Namespace:
    """Build and parse command-line arguments for Stage 7 runs."""
    parser = argparse.ArgumentParser(description="Run Stage 7 model packaging and delivery pipeline.")
    parser.add_argument("--best-model-path", type=Path, default=None, help="Path to best model pickle from Stage 5.")
    parser.add_argument("--preprocessor-path", type=Path, default=None, help="Path to preprocessor pickle from Stage 2.")
    parser.add_argument("--stage5-summary-path", type=Path, default=None, help="Path to Stage 5 summary JSON.")
    parser.add_argument("--preprocessing-report-path", type=Path, default=None, help="Path to Stage 2 preprocessing report JSON.")
    parser.add_argument("--cleaned-dataset-path", type=Path, default=None, help="Path to cleaned dataset CSV.")
    parser.add_argument("--package-version", type=str, default=None, help="Package version label (for example: v1).")
    parser.add_argument("--smoke-test-rows", type=int, default=None, help="Rows used for package smoke test prediction.")
    parser.add_argument("--log-level", type=str, default=None, help="Optional runtime log level.")
    parser.add_argument("--retries", type=int, default=3, help="Load/prediction retries.")
    parser.add_argument("--timeout-seconds", type=float, default=60.0, help="Load/prediction timeout per attempt.")
    parser.add_argument("--retry-delay-seconds", type=float, default=1.5, help="Delay between retries.")
    parser.add_argument("--io-retries", type=int, default=3, help="Artifact write retries.")
    parser.add_argument("--io-timeout-seconds", type=float, default=30.0, help="Artifact write timeout per attempt.")
    parser.add_argument("--io-retry-delay-seconds", type=float, default=1.0, help="Delay between artifact write retries.")
    return parser.parse_args()


def main() -> int:
    """Run Stage 7 and print generated artifact locations."""
    args = build_args()
    settings = Stage7Settings.from_env()

    # Apply command-line overrides while keeping settings immutable.
    overrides: dict[str, object] = {}
    if args.best_model_path is not None:
        overrides["best_model_path"] = args.best_model_path.resolve()
    if args.preprocessor_path is not None:
        overrides["preprocessor_path"] = args.preprocessor_path.resolve()
    if args.stage5_summary_path is not None:
        overrides["stage5_summary_path"] = args.stage5_summary_path.resolve()
    if args.preprocessing_report_path is not None:
        overrides["preprocessing_report_path"] = args.preprocessing_report_path.resolve()
    if args.cleaned_dataset_path is not None:
        overrides["cleaned_dataset_path"] = args.cleaned_dataset_path.resolve()
    if args.package_version is not None:
        overrides["package_version"] = args.package_version.strip()
    if args.smoke_test_rows is not None:
        overrides["smoke_test_rows"] = args.smoke_test_rows
    if args.log_level is not None:
        overrides["log_level"] = args.log_level.upper()
    if overrides:
        settings = replace(settings, **overrides)

    outputs = run_stage_7(
        settings=settings,
        retries=args.retries,
        timeout_seconds=args.timeout_seconds,
        retry_delay_seconds=args.retry_delay_seconds,
        io_retries=args.io_retries,
        io_timeout_seconds=args.io_timeout_seconds,
        io_retry_delay_seconds=args.io_retry_delay_seconds,
    )

    print("Stage 7 completed.")
    print(f"Package directory: {outputs.package_dir}")
    print(f"Manifest: {outputs.manifest_path}")
    print(f"Input schema: {outputs.schema_path}")
    print(f"Smoke test predictions: {outputs.smoke_test_predictions_path}")
    print(f"Package README: {outputs.package_readme_path}")
    print(f"Summary report: {outputs.summary_report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
