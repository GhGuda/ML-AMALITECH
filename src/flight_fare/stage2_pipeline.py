"""Stage 2 pipeline for cleaning, preprocessing, and train/test artifact generation."""

from __future__ import annotations

import json
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

from .data_loader import load_flight_dataset
from .logging_utils import configure_logging, shutdown_logger_handlers
from .preprocessing import CleaningReport, split_and_preprocess, clean_and_engineer_features
from .settings import Stage2Settings

T = TypeVar("T")


class ArtifactWriteError(RuntimeError):
    """Raised when artifact persistence fails after retry attempts."""


class ArtifactWriteTimeoutError(ArtifactWriteError):
    """Raised when artifact persistence exceeds configured timeout."""


@dataclass(frozen=True)
class Stage2Outputs:
    """Artifacts produced by Stage 2 pipeline execution."""

    cleaned_dataset_path: Path
    x_train_processed_path: Path
    x_test_processed_path: Path
    y_train_path: Path
    y_test_path: Path
    preprocessor_path: Path
    report_path: Path


def run_with_retries_and_timeout(
    operation: Callable[[], T],
    operation_name: str,
    logger_name: str,
    retries: int,
    timeout_seconds: float,
    retry_delay_seconds: float,
) -> T:
    """Execute an operation with retry and timeout guarantees."""
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
            last_error = ArtifactWriteTimeoutError(
                f"Operation '{operation_name}' timed out after {timeout_seconds} seconds."
            )
            logger.warning(
                "Operation '%s' attempt %s/%s timed out.",
                operation_name,
                attempt,
                retries,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = ArtifactWriteError(f"Operation '{operation_name}' failed: {exc}")
            logger.warning(
                "Operation '%s' attempt %s/%s failed: %s",
                operation_name,
                attempt,
                retries,
                exc,
            )

        if attempt < retries:
            # Add a small delay to reduce contention on transient filesystem issues.
            time.sleep(retry_delay_seconds)

    if last_error is None:
        raise ArtifactWriteError(f"Operation '{operation_name}' failed with unknown error.")
    raise last_error


def save_dataframe_csv(dataframe, output_path: Path) -> Path:
    """Save dataframe to CSV and return output path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    return output_path


def save_target_series(target_series, output_path: Path, target_column_name: str = "target") -> Path:
    """Save target series as single-column CSV and return output path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_series.to_frame(name=target_column_name).to_csv(output_path, index=False)
    return output_path


def save_json(payload: dict, output_path: Path) -> Path:
    """Save dictionary payload as JSON with pretty formatting."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def save_preprocessor(preprocessor, output_path: Path) -> Path:
    """Persist fitted preprocessing object for downstream model stages."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file_obj:
        pickle.dump(preprocessor, file_obj)
    return output_path


def build_stage2_report(
    cleaning_report: CleaningReport,
    settings: Stage2Settings,
    x_train_rows: int,
    x_test_rows: int,
    processed_feature_count: int,
    numeric_feature_columns: list[str],
    categorical_feature_columns: list[str],
) -> dict:
    """Build structured Stage 2 report metadata."""
    return {
        "stage": 2,
        "dataset_path": str(settings.dataset_path),
        "split": {
            "test_size": settings.test_size,
            "random_state": settings.random_state,
            "train_rows": x_train_rows,
            "test_rows": x_test_rows,
        },
        "preprocessing": {
            "numeric_impute_strategy": settings.numeric_impute_strategy,
            "categorical_impute_strategy": settings.categorical_impute_strategy,
            "scaler_strategy": settings.scaler_strategy,
            "processed_feature_count": processed_feature_count,
            "numeric_feature_columns": numeric_feature_columns,
            "categorical_feature_columns": categorical_feature_columns,
        },
        "cleaning_report": cleaning_report.to_dict(),
    }


def run_stage_2(
    settings: Stage2Settings | None = None,
    retries: int = 3,
    timeout_seconds: float = 30.0,
    retry_delay_seconds: float = 1.5,
    io_retries: int = 3,
    io_timeout_seconds: float = 30.0,
    io_retry_delay_seconds: float = 1.0,
) -> Stage2Outputs:
    """Execute Stage 2 end-to-end and return generated artifact paths."""
    runtime_settings = settings or Stage2Settings.from_env()
    runtime_settings.ensure_directories()

    logger = configure_logging(runtime_settings, logger_name="flight_fare.stage2")
    try:
        logger.info("Starting Stage 2 pipeline execution.")
        logger.info("Dataset path: %s", runtime_settings.dataset_path)

        raw_dataframe = load_flight_dataset(
            csv_path=runtime_settings.dataset_path,
            logger=logger,
            retries=retries,
            timeout_seconds=timeout_seconds,
            retry_delay_seconds=retry_delay_seconds,
        )
        cleaned_dataframe, cleaning_report = clean_and_engineer_features(raw_dataframe)
        split_output = split_and_preprocess(
            dataframe=cleaned_dataframe,
            test_size=runtime_settings.test_size,
            random_state=runtime_settings.random_state,
            numeric_impute_strategy=runtime_settings.numeric_impute_strategy,
            categorical_impute_strategy=runtime_settings.categorical_impute_strategy,
            scaler_strategy=runtime_settings.scaler_strategy,
        )

        stage_output_dir = runtime_settings.artifacts_dir / "stage_2"
        stage_output_dir.mkdir(parents=True, exist_ok=True)

        cleaned_dataset_path = run_with_retries_and_timeout(
            operation=lambda: save_dataframe_csv(cleaned_dataframe, stage_output_dir / "cleaned_dataset.csv"),
            operation_name="save_cleaned_dataset",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )
        x_train_processed_path = run_with_retries_and_timeout(
            operation=lambda: save_dataframe_csv(split_output.x_train_processed, stage_output_dir / "x_train_processed.csv"),
            operation_name="save_x_train_processed",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )
        x_test_processed_path = run_with_retries_and_timeout(
            operation=lambda: save_dataframe_csv(split_output.x_test_processed, stage_output_dir / "x_test_processed.csv"),
            operation_name="save_x_test_processed",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )
        y_train_path = run_with_retries_and_timeout(
            operation=lambda: save_target_series(split_output.y_train, stage_output_dir / "y_train.csv", "Total Fare (BDT)"),
            operation_name="save_y_train",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )
        y_test_path = run_with_retries_and_timeout(
            operation=lambda: save_target_series(split_output.y_test, stage_output_dir / "y_test.csv", "Total Fare (BDT)"),
            operation_name="save_y_test",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )
        preprocessor_path = run_with_retries_and_timeout(
            operation=lambda: save_preprocessor(split_output.preprocessor, stage_output_dir / "preprocessor.pkl"),
            operation_name="save_preprocessor",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )

        report_payload = build_stage2_report(
            cleaning_report=cleaning_report,
            settings=runtime_settings,
            x_train_rows=len(split_output.x_train_processed),
            x_test_rows=len(split_output.x_test_processed),
            processed_feature_count=split_output.x_train_processed.shape[1],
            numeric_feature_columns=split_output.numeric_feature_columns,
            categorical_feature_columns=split_output.categorical_feature_columns,
        )
        report_path = run_with_retries_and_timeout(
            operation=lambda: save_json(report_payload, stage_output_dir / "preprocessing_report.json"),
            operation_name="save_preprocessing_report",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )

        logger.info("Stage 2 completed successfully.")
        logger.info("Cleaned dataset: %s", cleaned_dataset_path)
        logger.info("Preprocessor: %s", preprocessor_path)
        logger.info("Report: %s", report_path)

        return Stage2Outputs(
            cleaned_dataset_path=cleaned_dataset_path,
            x_train_processed_path=x_train_processed_path,
            x_test_processed_path=x_test_processed_path,
            y_train_path=y_train_path,
            y_test_path=y_test_path,
            preprocessor_path=preprocessor_path,
            report_path=report_path,
        )
    finally:
        shutdown_logger_handlers(logger)
