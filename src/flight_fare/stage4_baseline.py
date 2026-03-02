"""Stage 4 pipeline for baseline linear-regression modeling and diagnostics."""

from __future__ import annotations

import json
import pickle
import time

import joblib
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .logging_utils import configure_logging, shutdown_logger_handlers
from .settings import Stage4Settings

T = TypeVar("T")


class Stage4Error(RuntimeError):
    """Raised when Stage 4 execution fails."""


class Stage4TimeoutError(Stage4Error):
    """Raised when a Stage 4 operation exceeds timeout."""


@dataclass(frozen=True)
class Stage4Outputs:
    """Artifacts produced by Stage 4 pipeline execution."""

    metrics_path: Path
    predictions_path: Path
    model_path: Path
    summary_path: Path
    coefficients_path: Path
    plots_directory: Path


def run_with_retries_and_timeout(
    operation: Callable[[], T],
    operation_name: str,
    logger_name: str,
    retries: int,
    timeout_seconds: float,
    retry_delay_seconds: float,
) -> T:
    """Run any operation with retries and timeout enforcement."""
    if retries < 1:
        raise ValueError("retries must be at least 1.")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than 0.")

    import logging

    logger = logging.getLogger(logger_name)
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(operation)
                return future.result(timeout=timeout_seconds)
        except FutureTimeoutError as exc:
            last_error = Stage4TimeoutError(
                f"Operation '{operation_name}' timed out after {timeout_seconds} seconds."
            )
            logger.warning(
                "Operation '%s' timed out on attempt %s/%s.",
                operation_name,
                attempt,
                retries,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = Stage4Error(f"Operation '{operation_name}' failed: {exc}")
            logger.warning(
                "Operation '%s' failed on attempt %s/%s: %s",
                operation_name,
                attempt,
                retries,
                exc,
            )

        if attempt < retries:
            # Delay briefly to reduce transient contention errors.
            time.sleep(retry_delay_seconds)

    if last_error is None:
        raise Stage4Error(f"Operation '{operation_name}' failed with unknown error.")
    raise last_error


def load_csv_with_retries(
    csv_path: Path,
    logger_name: str,
    retries: int,
    timeout_seconds: float,
    retry_delay_seconds: float,
    read_csv_func: Callable[..., pd.DataFrame] = pd.read_csv,
) -> pd.DataFrame:
    """Load a CSV with retries and timeout protection."""
    return run_with_retries_and_timeout(
        operation=lambda: read_csv_func(csv_path),
        operation_name=f"load_csv:{csv_path.name}",
        logger_name=logger_name,
        retries=retries,
        timeout_seconds=timeout_seconds,
        retry_delay_seconds=retry_delay_seconds,
    )


def select_target_column(dataframe: pd.DataFrame, target_column: str) -> pd.Series:
    """Select target series by preferred name or fallback to first numeric column."""
    if target_column in dataframe.columns:
        return pd.to_numeric(dataframe[target_column], errors="coerce")

    numeric_columns = dataframe.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_columns:
        raise ValueError("No numeric target column found in target dataset.")
    return pd.to_numeric(dataframe[numeric_columns[0]], errors="coerce")


def validate_model_inputs(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> None:
    """Validate baseline model training and evaluation inputs."""
    if x_train.empty or x_test.empty:
        raise ValueError("Feature matrices must not be empty.")
    if len(x_train) != len(y_train):
        raise ValueError("x_train and y_train row counts must match.")
    if len(x_test) != len(y_test):
        raise ValueError("x_test and y_test row counts must match.")
    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError("x_train and x_test must have the same number of features.")
    if x_train.isna().any().any() or x_test.isna().any().any():
        raise ValueError("Feature matrices contain missing values.")
    if y_train.isna().any() or y_test.isna().any():
        raise ValueError("Target vectors contain missing values.")


def fit_linear_regression(x_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Fit and return a linear regression model."""
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model


def compute_regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Compute core regression metrics used in Stage 4 evaluation."""
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mse**0.5),
    }


def build_predictions_table(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    """Build prediction diagnostic table with residual-based columns."""
    predictions_df = pd.DataFrame(
        {
            "actual_total_fare_bdt": y_true.values,
            "predicted_total_fare_bdt": y_pred,
        }
    )
    predictions_df["residual"] = predictions_df["actual_total_fare_bdt"] - predictions_df["predicted_total_fare_bdt"]
    predictions_df["absolute_error"] = predictions_df["residual"].abs()
    predictions_df["squared_error"] = predictions_df["residual"] ** 2
    return predictions_df


def build_coefficients_table(feature_names: list[str], coefficients: list[float]) -> pd.DataFrame:
    """Build coefficient table sorted by absolute impact magnitude."""
    coefficient_df = pd.DataFrame(
        {
            "feature_name": feature_names,
            "coefficient": coefficients,
        }
    )
    coefficient_df["abs_coefficient"] = coefficient_df["coefficient"].abs()
    return coefficient_df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def save_json(payload: dict, output_path: Path) -> Path:
    """Persist dictionary payload as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def save_dataframe_csv(dataframe: pd.DataFrame, output_path: Path) -> Path:
    """Persist dataframe as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    return output_path


def save_model_pickle(model: LinearRegression, output_path: Path) -> Path:
    """Persist trained model as pickle artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return output_path


def create_actual_vs_predicted_plot(predictions_df: pd.DataFrame, output_path: Path) -> Path:
    """Save scatter plot of actual versus predicted fares."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(
        predictions_df["actual_total_fare_bdt"],
        predictions_df["predicted_total_fare_bdt"],
        alpha=0.35,
        s=12,
        color="#2a9d8f",
    )
    diagonal_min = min(predictions_df["actual_total_fare_bdt"].min(), predictions_df["predicted_total_fare_bdt"].min())
    diagonal_max = max(predictions_df["actual_total_fare_bdt"].max(), predictions_df["predicted_total_fare_bdt"].max())
    plt.plot([diagonal_min, diagonal_max], [diagonal_min, diagonal_max], color="#e76f51", linewidth=2)
    plt.title("Actual vs Predicted Total Fare")
    plt.xlabel("Actual Total Fare (BDT)")
    plt.ylabel("Predicted Total Fare (BDT)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def create_residual_distribution_plot(predictions_df: pd.DataFrame, output_path: Path) -> Path:
    """Save residual distribution histogram for diagnostic review."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.hist(predictions_df["residual"], bins=40, color="#264653", alpha=0.85)
    plt.title("Residual Distribution")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def create_residuals_vs_predicted_plot(predictions_df: pd.DataFrame, output_path: Path) -> Path:
    """Save residuals versus predicted scatter plot for heteroscedasticity checks."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(
        predictions_df["predicted_total_fare_bdt"],
        predictions_df["residual"],
        alpha=0.35,
        s=12,
        color="#457b9d",
    )
    plt.axhline(0, color="#e63946", linewidth=1.5)
    plt.title("Residuals vs Predicted")
    plt.xlabel("Predicted Total Fare (BDT)")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def run_stage_4(
    settings: Stage4Settings | None = None,
    retries: int = 3,
    timeout_seconds: float = 30.0,
    retry_delay_seconds: float = 1.5,
    io_retries: int = 3,
    io_timeout_seconds: float = 30.0,
    io_retry_delay_seconds: float = 1.0,
) -> Stage4Outputs:
    """Execute Stage 4 baseline modeling end-to-end."""
    runtime_settings = settings or Stage4Settings.from_env()
    runtime_settings.ensure_directories()

    logger = configure_logging(runtime_settings, logger_name="flight_fare.stage4")
    try:
        logger.info("Starting Stage 4 baseline model pipeline.")

        x_train = load_csv_with_retries(
            csv_path=runtime_settings.x_train_path,
            logger_name=logger.name,
            retries=retries,
            timeout_seconds=timeout_seconds,
            retry_delay_seconds=retry_delay_seconds,
        )
        x_test = load_csv_with_retries(
            csv_path=runtime_settings.x_test_path,
            logger_name=logger.name,
            retries=retries,
            timeout_seconds=timeout_seconds,
            retry_delay_seconds=retry_delay_seconds,
        )
        y_train_df = load_csv_with_retries(
            csv_path=runtime_settings.y_train_path,
            logger_name=logger.name,
            retries=retries,
            timeout_seconds=timeout_seconds,
            retry_delay_seconds=retry_delay_seconds,
        )
        y_test_df = load_csv_with_retries(
            csv_path=runtime_settings.y_test_path,
            logger_name=logger.name,
            retries=retries,
            timeout_seconds=timeout_seconds,
            retry_delay_seconds=retry_delay_seconds,
        )

        y_train = select_target_column(y_train_df, runtime_settings.target_column)
        y_test = select_target_column(y_test_df, runtime_settings.target_column)
        validate_model_inputs(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)

        model = run_with_retries_and_timeout(
            operation=lambda: fit_linear_regression(x_train=x_train, y_train=y_train),
            operation_name="fit_linear_regression",
            logger_name=logger.name,
            retries=retries,
            timeout_seconds=timeout_seconds,
            retry_delay_seconds=retry_delay_seconds,
        )

        y_train_pred = pd.Series(model.predict(x_train), index=y_train.index)
        y_test_pred = pd.Series(model.predict(x_test), index=y_test.index)

        train_metrics = compute_regression_metrics(y_true=y_train, y_pred=y_train_pred)
        test_metrics = compute_regression_metrics(y_true=y_test, y_pred=y_test_pred)
        predictions_df = build_predictions_table(y_true=y_test, y_pred=y_test_pred)
        coefficient_df = build_coefficients_table(feature_names=x_train.columns.tolist(), coefficients=model.coef_.tolist())

        stage_output_dir = runtime_settings.artifacts_dir / "stage_4"
        plots_dir = stage_output_dir / "plots"
        stage_output_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        metrics_payload = {
            "stage": 4,
            "model": "LinearRegression",
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "generalization_gap": {
                "r2_gap": float(train_metrics["r2"] - test_metrics["r2"]),
                "mae_gap": float(test_metrics["mae"] - train_metrics["mae"]),
                "rmse_gap": float(test_metrics["rmse"] - train_metrics["rmse"]),
            },
            "dataset_shapes": {
                "x_train_rows": int(len(x_train)),
                "x_test_rows": int(len(x_test)),
                "feature_count": int(x_train.shape[1]),
            },
        }
        metrics_path = run_with_retries_and_timeout(
            operation=lambda: save_json(metrics_payload, stage_output_dir / "baseline_metrics.json"),
            operation_name="save_baseline_metrics",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )
        predictions_path = run_with_retries_and_timeout(
            operation=lambda: save_dataframe_csv(predictions_df, stage_output_dir / "predictions_test.csv"),
            operation_name="save_predictions_test",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )
        coefficients_path = run_with_retries_and_timeout(
            operation=lambda: save_dataframe_csv(coefficient_df, stage_output_dir / "linear_regression_coefficients.csv"),
            operation_name="save_linear_regression_coefficients",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )
        model_path = run_with_retries_and_timeout(
            operation=lambda: save_model_pickle(model, stage_output_dir / "linear_regression_model.pkl"),
            operation_name="save_linear_regression_model",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )

        actual_vs_predicted_path = run_with_retries_and_timeout(
            operation=lambda: create_actual_vs_predicted_plot(predictions_df, plots_dir / "actual_vs_predicted.png"),
            operation_name="plot_actual_vs_predicted",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )
        residual_distribution_path = run_with_retries_and_timeout(
            operation=lambda: create_residual_distribution_plot(predictions_df, plots_dir / "residual_distribution.png"),
            operation_name="plot_residual_distribution",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )
        residuals_vs_predicted_path = run_with_retries_and_timeout(
            operation=lambda: create_residuals_vs_predicted_plot(predictions_df, plots_dir / "residuals_vs_predicted.png"),
            operation_name="plot_residuals_vs_predicted",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )

        summary_payload = {
            "stage": 4,
            "artifacts": {
                "metrics_path": str(metrics_path),
                "predictions_path": str(predictions_path),
                "coefficients_path": str(coefficients_path),
                "model_path": str(model_path),
                "actual_vs_predicted_plot": str(actual_vs_predicted_path),
                "residual_distribution_plot": str(residual_distribution_path),
                "residuals_vs_predicted_plot": str(residuals_vs_predicted_path),
            },
            "test_metrics": test_metrics,
        }
        summary_path = run_with_retries_and_timeout(
            operation=lambda: save_json(summary_payload, stage_output_dir / "stage4_summary_report.json"),
            operation_name="save_stage4_summary_report",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )

        logger.info("Stage 4 completed successfully.")
        logger.info("Test metrics: R2=%.4f MAE=%.4f RMSE=%.4f", test_metrics["r2"], test_metrics["mae"], test_metrics["rmse"])

        return Stage4Outputs(
            metrics_path=metrics_path,
            predictions_path=predictions_path,
            model_path=model_path,
            summary_path=summary_path,
            coefficients_path=coefficients_path,
            plots_directory=plots_dir,
        )
    finally:
        shutdown_logger_handlers(logger)
