"""CLI entry point for Stage 5 advanced model tuning pipeline."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from flight_fare.settings import Stage5Settings  # noqa: E402
from flight_fare.stage5_advanced import run_stage_5  # noqa: E402


def build_args() -> argparse.Namespace:
    """Build and parse command-line arguments for Stage 5 runs."""
    parser = argparse.ArgumentParser(description="Run Stage 5 advanced model tuning pipeline.")
    parser.add_argument("--x-train-path", type=Path, default=None, help="Path to processed training feature CSV.")
    parser.add_argument("--x-test-path", type=Path, default=None, help="Path to processed test feature CSV.")
    parser.add_argument("--y-train-path", type=Path, default=None, help="Path to training target CSV.")
    parser.add_argument("--y-test-path", type=Path, default=None, help="Path to test target CSV.")
    parser.add_argument("--target-column", type=str, default=None, help="Target column name in y CSV files.")
    parser.add_argument("--cv-folds", type=int, default=None, help="Cross-validation folds for tuning.")
    parser.add_argument("--tuning-sample-fraction", type=float, default=None, help="Fraction of train rows used for hyperparameter search.")
    parser.add_argument("--tree-search-iterations", type=int, default=None, help="Randomized search iterations for decision tree.")
    parser.add_argument("--forest-search-iterations", type=int, default=None, help="Randomized search iterations for random forest.")
    parser.add_argument("--gb-search-iterations", type=int, default=None, help="Randomized search iterations for gradient boosting.")
    parser.add_argument("--n-jobs", type=int, default=None, help="Number of CPU jobs used by scikit-learn searches.")
    parser.add_argument("--log-level", type=str, default=None, help="Optional runtime log level.")
    parser.add_argument("--retries", type=int, default=3, help="Model/load retries.")
    parser.add_argument("--timeout-seconds", type=float, default=300.0, help="Load/search timeout per attempt.")
    parser.add_argument("--retry-delay-seconds", type=float, default=1.5, help="Delay between retries.")
    parser.add_argument("--io-retries", type=int, default=3, help="Artifact write retries.")
    parser.add_argument("--io-timeout-seconds", type=float, default=60.0, help="Artifact write timeout per attempt.")
    parser.add_argument("--io-retry-delay-seconds", type=float, default=1.0, help="Delay between artifact write retries.")
    return parser.parse_args()


def main() -> int:
    """Run Stage 5 and print generated artifact paths."""
    args = build_args()
    settings = Stage5Settings.from_env()

    # Apply command-line overrides without mutating immutable settings.
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
    if args.cv_folds is not None:
        overrides["cv_folds"] = args.cv_folds
    if args.tuning_sample_fraction is not None:
        overrides["tuning_sample_fraction"] = args.tuning_sample_fraction
    if args.tree_search_iterations is not None:
        overrides["tree_search_iterations"] = args.tree_search_iterations
    if args.forest_search_iterations is not None:
        overrides["forest_search_iterations"] = args.forest_search_iterations
    if args.gb_search_iterations is not None:
        overrides["gb_search_iterations"] = args.gb_search_iterations
    if args.n_jobs is not None:
        overrides["n_jobs"] = args.n_jobs
    if args.log_level is not None:
        overrides["log_level"] = args.log_level.upper()
    if overrides:
        settings = replace(settings, **overrides)

    outputs = run_stage_5(
        settings=settings,
        retries=args.retries,
        timeout_seconds=args.timeout_seconds,
        retry_delay_seconds=args.retry_delay_seconds,
        io_retries=args.io_retries,
        io_timeout_seconds=args.io_timeout_seconds,
        io_retry_delay_seconds=args.io_retry_delay_seconds,
    )

    print("Stage 5 completed.")
    print(f"Comparison table: {outputs.comparison_table_path}")
    print(f"Tuning results: {outputs.tuning_results_path}")
    print(f"Best model: {outputs.best_model_path}")
    print(f"Best model predictions: {outputs.best_model_predictions_path}")
    print(f"Summary report: {outputs.summary_report_path}")
    print(f"Regularization effect: {outputs.regularization_effect_path}")
    print(f"Bias-variance table: {outputs.bias_variance_path}")
    print(f"Best model feature impact: {outputs.best_model_importance_path}")
    print(f"Plots directory: {outputs.plots_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
